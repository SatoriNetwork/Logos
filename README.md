# LLM for Search of Satori Data Streams


## Run chromadb on it's own server for efficient embedding queries

 todo: self signed tls
```
htpasswd -c /chroma/nginx/.htpasswd admin

docker build -t chromadb-server .
docker build -t nginx-proxy .
docker network create chromadb-network
docker run -d \
    --name chromadb \
    --network chromadb-network \
    -v /chroma:/data/chroma \
    chromadb-server
docker run -d \
    --name nginx-proxy \
    --network chromadb-network \
    -v /chroma/nginx/.htpasswd:/etc/nginx/.htpasswd:ro \
    -p 8000:80 \
    nginx-proxy
```
