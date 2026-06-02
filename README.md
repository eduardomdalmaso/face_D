# Benchmark de Detectores Faciais (DeepFace)

Este projeto implementa um pipeline automatizado de teste e benchmark comparativo para diferentes frameworks de detecção facial. Ele utiliza a biblioteca `DeepFace` para carregar e comparar os seguintes detectores em uma mesma imagem capturada:

1.  **RetinaFace** (Verde) - Estado da arte em acurácia.
2.  **MediaPipe** (Azul) - Detector rápido do Google.
3.  **YuNet** (Vermelho) - Modelo leve e veloz otimizado para OpenCV.
4.  **OpenCV Haar Cascades** (Amarelo) - Método tradicional clássico.

A captura de imagens pela webcam para o benchmark em lote é realizada diretamente via linha de comando, salvando o arquivo localmente sem a necessidade de abrir interfaces gráficas (GUI).

---

## 🛠️ Requisitos e Instalação

### 1. Criar Ambiente Conda
Crie e ative o ambiente Conda chamado `Face` com Python 3.11:

```bash
conda create -n Face python=3.11 -y
conda activate Face
```

### 2. Instalar Dependências
Instale as dependências listadas no `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## ⚡ Aceleração por GPU (NVIDIA / CUDA)

O pipeline foi projetado para detectar e utilizar sua GPU automaticamente para rodar as inferências em milissegundos. Para que o TensorFlow (RetinaFace) e o OpenCV (YuNet) consigam carregar a aceleração por hardware, siga as etapas abaixo no seu ambiente Conda:

### A. Para TensorFlow / RetinaFace:
Instale o CUDA toolkit e cuDNN diretamente no ambiente conda (exemplo para CUDA 11.8):
```bash
conda install -c conda-forge cudatoolkit=11.8 cudnn=8.6.0 -y
# Ou instale via pip:
pip install tensorflow[and-cuda]
```

### B. Para OpenCV YuNet:
O pacote oficial `opencv-python` distribuído via pip roda somente em CPU. Para rodar o YuNet na GPU, instale uma versão com suporte a CUDA compilado ou compile o OpenCV com os sinalizadores de CUDA ativados:
```bash
# Uma alternativa comunitária popular pré-compilada:
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python-cuda
```

### C. Para MediaPipe:
Se você estiver rodando em sessões de terminal interativas ou que tenham acesso ao display ativo, o MediaPipe usará a aceleração de GPU por padrão. Caso esteja executando via background ou SSH headless (sem X forwarding) e ocorra um `SEGFAULT`, você pode forçar o MediaPipe a rodar em CPU definindo a variável de ambiente:
```bash
export USE_GPU=false
```

---

## 🚀 Como Executar

O projeto possui duas formas de execução: o benchmark CLI original e o servidor Web avançado com FastAPI que inclui Inteligência Artificial Híbrida e Aprendizado Ativo.

### 💻 Opção A: Servidor Web Híbrido com Aprendizado Ativo (FastAPI)
Inicializa o servidor web na porta `8000` com suporte automático a **aceleração por GPU (CUDA)**, suporte a múltiplos templates faciais, classificação via classificador SVM e fila de Human-in-the-loop:

```bash
conda activate Face
# Concede permissão de execução (se necessário) e inicia
chmod +x start_server.sh
./start_server.sh
```
Abra o navegador em: [http://localhost:8000](http://localhost:8000)

**Funcionalidades Web Premium:**
*   **Scanner:** Ativa a câmera, detecta rostos em tempo real e realiza o reconhecimento usando uma abordagem híbrida:
    *   **Classificador SVM:** Se houver pelo menos 2 usuários cadastrados com templates, um modelo de Machine Learning (`sklearn.svm.SVC`) é treinado dinamicamente e usado para resolver de forma fina as fronteiras de decisão (como distinguir irmãos).
    *   **Galeria Multi-Template:** Se houver dados insuficientes para treinar o classificador, o sistema faz fallback automático para busca por distância cosseno contra múltiplos templates cadastrados.
*   **Auto-aprendizado (Qualidade):** Faces com excelente qualidade ($\le 0.50$) e que apresentam variação de pose/ângulo ($> 0.35$ de distância cosseno dos templates atuais) são cadastradas de forma autônoma como templates de pose adicionais (limite de 10 templates por perfil).
*   **Aba "Revisões" (Active Learning):** Detecções com correspondência limítrofe (distância entre `0.50` e `0.70`) caem na fila de revisões. Através desta aba, o operador do sistema pode:
  *   Confirmar e vincular a face sob revisão a um usuário já cadastrado.
  *   Criar um novo usuário diretamente a partir da imagem recortada.
  *   Rejeitar e apagar a imagem.
*   **Cadastrar Rosto:** Registra uma nova pessoa tirando foto pela webcam ou carregando um arquivo.
*   **Histórico & Usuários:** Log completo de escaneamentos, latências de inferência e listagem de usuários contendo a quantidade de templates ativos.

---

### 🖥️ Opção B: Benchmark CLI (Original)
Execute o script de benchmark comparativo clássico:

```bash
conda activate Face
python main.py
```

### O que o script de benchmark faz:
1. **Captura da Webcam (`src/capture.py`):** Inicializa a câmera padrão do servidor por 1.5s, captura um frame e o salva em `data/test_images/webcam_capture.jpg`.
2. **Execução do Benchmark (`main.py`):** Roda os detectores configurados calculando latência e contagem de rostos.
3. **Geração do Resultado Comparativo:** Desenha caixas demarcadoras coloridas e salva em `data/test_images/comparison_result.jpg`.

---

## 📂 Estrutura de Arquivos do Projeto

*   [main.py](file:///home/hades/Documents/visao/face_D/main.py): Orquestrador do benchmark comparativo CLI.
*   [requirements.txt](file:///home/hades/Documents/visao/face_D/requirements.txt): Bibliotecas necessárias (incluindo `scikit-learn` e `qdrant-client`).
*   `src/`
    *   [src/server.py](file:///home/hades/Documents/visao/face_D/src/server.py): Servidor FastAPI contendo toda a classificação híbrida (SVM), lógica de aprendizado ativo e rotas REST/WebSocket.
    *   [src/database.py](file:///home/hades/Documents/visao/face_D/src/database.py): Camada de dados híbrida (SQLite local para testes fáceis e Qdrant para produção).
    *   [src/capture.py](file:///home/hades/Documents/visao/face_D/src/capture.py): Captura de imagem sem interface gráfica do servidor.
    *   `detectors/`
        *   [src/detectors/base.py](file:///home/hades/Documents/visao/face_D/src/detectors/base.py): Interface padrão para detectores.
        *   [src/detectors/deepface_det.py](file:///home/hades/Documents/visao/face_D/src/detectors/deepface_det.py): Detector que encapsula RetinaFace.
        *   [src/detectors/opencv_yunet.py](file:///home/hades/Documents/visao/face_D/src/detectors/opencv_yunet.py): Detector nativo OpenCV YuNet com retorno de landmarks.
*   `templates/`
    *   [templates/index.html](file:///home/hades/Documents/visao/face_D/templates/index.html): Interface web premium com o painel de "Revisões" e fluxo Human-in-the-loop.
*   `data/`
    *   `face_recognition.db`: Banco de dados SQLite normalizado com tabelas para usuários, multi-templates, logs de histórico e revisões pendentes.
    *   `registered/`: Pasta que contém as fotos e templates permanentes.
    *   `pending_review/`: Pasta temporária das fotos sob revisão.

