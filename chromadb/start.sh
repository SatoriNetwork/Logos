#!/bin/sh

# Log ChromaDB version
echo "Logging ChromaDB version..."
python -c "import chromadb; print(chromadb.__version__)"

# Start the server
exec chroma run --help

exec chroma run --path "/data/chroma" --host "0.0.0.0" --port 8000
