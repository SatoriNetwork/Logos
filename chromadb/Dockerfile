# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir chromadb psycopg[binary] uvicorn

# Expose ChromaDB port
EXPOSE 8000

# Set environment variables
ENV CHROMA_DB_PATH=/data/chroma
ENV CHROMA_HOST=0.0.0.0
ENV CHROMA_PORT=8000

# Create data directory
RUN mkdir -p ${CHROMA_DB_PATH}

# Start ChromaDB server
CMD ["chroma", "--path", "/data/chroma", "--bind", "0.0.0.0", "--port", "8000"]


# docker build -t chromadb-server .
# docker run -d -p 8000:8000 -v /chroma:/data/chroma --name chromadb chromadb-server
