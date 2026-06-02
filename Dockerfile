FROM python:3.11-slim

# Instala dependências do sistema necessárias para OpenCV, GL e CUDA
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia dependências e instala
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia código fonte e templates
COPY src/ ./src/
COPY templates/ ./templates/
COPY main.py .

EXPOSE 8000

# Variáveis de ambiente padrão
ENV QDRANT_HOST=qdrant
ENV QDRANT_PORT=6333

CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]
