# Diretrizes de Segurança e Práticas Recomendadas (SECURITY.md)

Este documento estabelece as regras de segurança do projeto que qualquer desenvolvedor (humano ou assistente de IA) deve seguir rigidamente ao contribuir para este repositório.

---

## 1. Diretrizes para Agentes de IA (AI Safety Rules)

### 🚫 A. Gestão de Segredos e Chaves de API
*   **Nunca** escreva chaves de API, senhas, tokens ou credenciais diretamente no código-fonte.
*   Qualquer segredo deve ser carregado através de variáveis de ambiente (`os.getenv`) ou usando a biblioteca `python-dotenv`.
*   O arquivo `.env` deve constar obrigatoriamente no [.gitignore](file:///home/hades/Documents/visao/face_D/.gitignore) para impedir vazamentos acidentais no GitHub.

### 🔐 B. Proteção de Dados Biométricos (Embeddings)
*   O banco de dados SQLite armazena vetores numéricos de 4096 dimensões (embeddings) representando a face humana. Estes vetores não podem ser revertidos de forma simples de volta em uma imagem facial legível.
*   Para manter a privacidade e segurança do projeto em ambientes de produção, restrinja o acesso direto ao arquivo de banco de dados `data/face_recognition.db` utilizando permissões de sistema adequadas (`chmod 600`).

### 📂 C. Validação de Uploads e Sanitização de Nomes
No endpoint `/register` do FastAPI:
*   Sanitize os nomes recebidos via Form-data antes de criar arquivos no disco (por exemplo, removendo caracteres especiais e barras para evitar injeção de caminho ou escalada de diretório).
*   Garante que o arquivo enviado corresponda estritamente a um formato de imagem (JPEG, PNG) decodificando e lendo a estrutura da imagem antes de processá-la.

### 🔒 D. Sanitização e Validação de Inputs (Path Traversal)
Ao manipular caminhos de imagens e capturas:
*   Sempre valide e sanitize os caminhos de arquivos fornecidos pelo usuário para evitar vulnerabilidades de *Path Traversal* (onde um invasor tenta ler arquivos sensíveis do sistema como `/etc/passwd`).
*   Utilize `os.path.basename` ou verifique se os caminhos resolvidos estão estritamente dentro do diretório do projeto (`data/test_images/`).

### 📦 C. Download Seguro de Modelos e Pesos (Segurança da Supply Chain)
Os detectores locais baixam arquivos de pesos grandes automaticamente (`.onnx`, `.tflite`, `.h5`):
*   **Apenas** use links oficiais e criptografados (HTTPS) pertencentes aos repositórios oficiais dos mantenedores (como o OpenCV Zoo GitHub ou servidores oficiais da Google para o MediaPipe).
*   **Atenção:** Evite usar arquivos de serialização inseguros como `.pkl` (Pickle) vindos de fontes externas não confiáveis, pois a desserialização do Pickle permite a execução de código arbitrário no sistema host. Dê preferência a formatos robustos de intercâmbio de modelos como `.onnx` ou `.tflite`.

### 📷 D. Gerenciamento Seguro de Hardware (Webcam)
A captura de webcam acessa recursos físicos do sistema operacional:
*   Sempre envolva a inicialização e manipulação do dispositivo de captura em blocos `try...finally` ou garanta que `cap.release()` seja chamado no encerramento, mesmo em caso de exceção.
*   Isso evita o vazamento de recursos do sistema e impede que a webcam fique travada como "em uso" em processos zumbis do Linux.

---

## 2. Reportando Vulnerabilidades

Se você identificar alguma falha de segurança no código deste pipeline, por favor:
1. Não crie uma issue pública no GitHub.
2. Reporte de maneira privada para o mantenedor do repositório para que uma correção seja desenvolvida e aplicada de forma segura.
