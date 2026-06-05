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
│   │   ├── opencv_haar.py     # Detector clássico Haar Cascades
│   │   └── yolo_det.py        # Detector de pessoas YOLOv8-nano (ONNX)
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

## 3. O Pipeline Híbrido Web Gated por Detecção de Pessoas (YOLOv8-nano + YuNet/RetinaFace)

Para a interface web interativa do FastAPI, implementamos um pipeline hierárquico híbrido otimizado para evitar falsos positivos no fundo da imagem:
1. **Detecção de Pessoas (YOLOv8):** O frame recebido é pré-processado (blob 640x640) e enviado ao modelo **YOLOv8-nano ONNX** via OpenCV DNN. O backend seleciona CUDA (GPU) automaticamente se disponível, ou CPU como fallback.
2. **Gating por Região de Interesse (RoI):** Se nenhuma pessoa for detectada no frame, o pipeline de detecção facial é abortado imediatamente. Isso evita varreduras de falsos positivos em cenários vazios (como objetos estáticos, cadeiras, etc.).
3. **Detecção Facial Localizada:** Para cada pessoa detectada, recortamos sua RoI correspondente com uma margem extra de 10%. Rodamos a detecção facial (YuNet ou RetinaFace) apenas dentro de cada crop humano.
4. **Remapeamento de Coordenadas:** Transladamos as coordenadas da caixa delimitadora do rosto e de seus landmarks de volta para o plano coordenado do frame original de 2304x1296 (ou resolução nativa da câmera).
5. **Extração de Assinatura Focal:** O fragmento correspondente ao rosto remapeado é cortado e enviado ao **DeepFace.represent** com `detector_backend='skip'`. Como o rosto já foi isolado, a geração do embedding via ArcFace ocorre em milissegundos.
6. **Reconhecimento Híbrido:** O vetor gerado é classificado usando similaridade cosseno direta e predições probabilísticas do modelo SVM para determinar se a pessoa é cadastrada ou "Desconhecida".

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
