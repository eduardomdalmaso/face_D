# Arquitetura do Sistema: Pipeline de Teste de Detecção Facial

Este documento detalha o design de engenharia do pipeline de benchmark de detecção facial.

## 1. Visão Geral da Arquitetura

O projeto foi projetado seguindo princípios de programação orientada a objetos (SOLID), modularidade e resiliência a falhas de hardware e drivers gráficos.

```text
face_D/
├── data/
│   ├── models/                # Ignorado no git - armazena binários (.onnx, .tflite)
│   ├── registered/            # Armazena imagens dos rostos cadastrados
│   ├── face_recognition.db    # Banco de dados SQLite persistente
│   └── test_images/           # Imagens capturadas e saídas visuais
├── src/
│   ├── detectors/
│   │   ├── base.py            # Classe abstrata base e Pydantic schema
│   │   ├── deepface_det.py    # Detector RetinaFace via DeepFace
│   │   ├── mediapipe_det.py   # Detector MediaPipe Face (Novo Tasks API)
│   │   ├── opencv_yunet.py    # Detector OpenCV YuNet (Nativo)
│   │   └── opencv_haar.py     # Detector clássico Haar Cascades
│   ├── database.py            # Persistência SQLite e cálculo de similaridade
│   ├── server.py              # API FastAPI e roteamento estático
│   └── capture.py             # Captura de webcam de baixo nível (CLI/Headless)
├── templates/
│   └── index.html             # Interface gráfica Web do cliente
├── main.py                    # Orquestrador do benchmark (Isolado via Processos)
├── requirements.txt           # Lista de dependências python
└── README.md                  # Manual do usuário
```

---

## 2. Fluxo de Dados e Persistência (SQLite)

O projeto utiliza um banco de dados SQLite (`data/face_recognition.db`) para armazenar o cadastro de faces e logs de histórico de acesso:

1. **Tabela `users`:** Armazena o nome exclusivo e a assinatura facial (*embedding*) gerada pela rede neural **ArcFace** (512 dimensões). O embedding é armazenado como um array serializado em formato JSON. O banco realiza autolimpeza (*self-healing*) na inicialização para remover assinaturas incompatíveis de outros modelos (como VGG-Face de 4096 dimensões).
2. **Tabela `history`:** Registra cada tentativa de reconhecimento, salvando o carimbo de data/hora (*timestamp*), a contagem de rostos detectados no frame, a lista de pessoas identificadas (ou "Desconhecido") e a latência da inferência.

### Comparação Matemática e Precisão Ajustável
O reconhecimento facial é feito calculando a **distância de cosseno** (`dist = 1.0 - similaridade_cosseno`) entre o embedding extraído da imagem de teste e todos os vetores previamente armazenados no SQLite.
*   **Limiar de Distância Configurável (Tolerância):** O sistema aceita um parâmetro dinâmico `threshold` (0.00 a 1.00) enviado do painel de controle do front-end. O padrão ideal para o ArcFace é **68%** (0.68). Isso permite que o usuário aumente ou diminua a sensibilidade de correspondência em tempo real.
*   **Upgrade Automático de Qualidade:** Quando um rosto é identificado com sucesso e a distância de cosseno obtida é **inferior** (representando uma correspondência mais próxima e fiel) ao `quality_score` gravado anteriormente para o usuário (desde que seja $\le 0.50$), o sistema automaticamente substitui a foto antiga no disco (`data/registered/{name}.jpg`) pela nova imagem recortada da webcam (que possui melhor iluminação, foco ou enquadramento facial) e recalcula/atualiza o embedding e o score de qualidade no SQLite.

---

## 3. O Pipeline Híbrido Web (Velocidade & Precisão)

Para a interface web interativa do FastAPI, implementamos um pipeline híbrido otimizado:
1. **Detecção Flexível (Selecionável):** O frame em base64 recebido da webcam do navegador é processado pelo detector selecionado pelo usuário no front-end:
   *   **YuNet (⚡ Rápido / CPU):** Ideal para detecção em tempo real de frente (latência de ~16ms).
   *   **RetinaFace (🎯 Acurácia / Perfil):** Ideal para detectar rostos em ângulos extremos (inclinação, lado/perfil, oclusão parcial), executado de forma robusta e precisa.
2. **Extração de Assinatura Focal:** O fragmento correspondente ao rosto é cortado e enviado ao **DeepFace.represent** configurado com `detector_backend='skip'`. Ao ignorar a fase de detecção interna do DeepFace (já feita pelo detector escolhido), a extração do vetor do rosto é acelerada exponencialmente.
3. **Reconhecimento:** O vetor é cruzado em milissegundos com o cache local de vetores em memória para identificar a pessoa.

---

## 4. Design do Módulo de Detecção (`src/detectors/`)

Todos os detectores herdam da classe abstrata `BaseDetector` localizada em [base.py](file:///home/hades/Documents/visao/face_D/src/detectors/base.py) e retornam o mesmo tipo de objeto estruturado `DetectionResult` (validados via Pydantic).

```python
class DetectionResult(BaseModel):
    x: int             # Coordenada X inicial (canto superior esquerdo)
    y: int             # Coordenada Y inicial
    w: int             # Largura da caixa delimitadora
    h: int             # Altura da caixa delimitadora
    confidence: float  # Confiança da predição (0.0 a 1.0)
```

Isso nos permite adicionar novos detectores (como YOLOv8/v10, Dlib, SSD) no futuro sem alterar o script orquestrador principal.

---

## 3. Resiliência por Isolamento de Subprocessos (`main.py`)

Bibliotecas de visão computacional e Deep Learning (como TensorFlow e MediaPipe) frequentemente causam **conflitos de drivers, vazamentos de memória ou Segmentation Faults (SIGSEGV)** quando executadas consecutivamente no mesmo processo do Python, especialmente ao tentar inicializar contextos GPU/EGL.

Para resolver isso, o [main.py](file:///home/hades/Documents/visao/face_D/main.py) isola a execução de cada detector em um processo separado utilizando o módulo `multiprocessing` do Python:

1. Um subprocesso limpo é gerado (`multiprocessing.Process(target=run_isolated_detector, ...)`) utilizando o método de inicialização **`spawn`** para garantir isolamento total de memória.
2. Os resultados e metadados de latência são retornados por meio de uma fila interprocessos (`multiprocessing.Queue`).
3. Se um detector travar por estouro de memória GPU, erro de CUDA ou apresentar um *segfault* (retornando código `-11` ou `139` no Linux), o processo pai intercepta a falha através do `p.exitcode`, registra o erro correspondente e continua a execução dos demais detectores sem quebrar a aplicação principal.

---

## 4. Aceleração por Hardware (GPU / CUDA)

O pipeline está configurado para aproveitar GPUs dedicadas NVIDIA compatíveis caso as dependências de sistema estejam configuradas:

1. **RetinaFace (TensorFlow):** O TensorFlow detecta e aloca automaticamente a GPU CUDA para processar as predições.
2. **OpenCV YuNet (DNN):** O módulo `cv2.FaceDetectorYN` detecta se o OpenCV foi compilado com suporte a CUDA (`DNN_BACKEND_CUDA` e `DNN_TARGET_CUDA`) e migra os cálculos de CPU para GPU dinamicamente.
3. **MediaPipe:** Utiliza a GPU de maneira configurável através da variável de ambiente `USE_GPU`. Como inicializações de GPU do MediaPipe (EGL) podem falhar em sessões sem monitor/headless, o parâmetro permite chavear facilmente para CPU se necessário (`export USE_GPU=false`).
