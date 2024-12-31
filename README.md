# LLM for Search of Satori Data Streams

voice layer

## Run chromadb on it's own server for efficient embedding queries

### setup folders
```
# Create directories with permissions
cd /
mkdir -p chroma/nginx/tls
chmod 755 chroma
sudo chown 1000:1000 /chroma

# Set up basic auth
apt install apache2-utils -y
htpasswd -c /chroma/nginx/.htpasswd admin

# Generate self-signed TLS certificate
cd /chroma/nginx/tls
openssl req -x509 -nodes -days 3650 -newkey rsa:2048 \
    -keyout selfsigned.key -out selfsigned.crt \
    -subj "/C=US/ST=State/L=City/O=Internal/OU=Dev/CN=internal.chroma"

# Create Docker network
docker network create chromadb-network

# save the api key to environment variables
export CHROMA_API_KEY="super-secret-key-123"
```

### Run it
```
# Build and run ChromaDB
cd ~/repos/Logos/chromadb
docker build -t chromadb-server .

# Build and run NGINX
cd ~/repos/Logos/nginx
docker build -t nginx-proxy .

# Run NGINX with API Key
 docker run -d --name nginx-proxy --network chromadb-network \
     -v /chroma/nginx/.htpasswd:/etc/nginx/.htpasswd:ro \
     -v /chroma/nginx/tls:/etc/nginx/tls:ro \
     -p 443:443 \
     -e CHROMA_API_KEY=$CHROMA_API_KEY \
     nginx-proxy

# Run ChromaDB server
docker run -d --name chromadb --network chromadb-network \
    -p 127.0.0.1:8000:8000 \
    -v /chroma:/data/chroma \
    chromadb-server
```
