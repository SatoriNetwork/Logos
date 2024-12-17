# Base image
FROM nginx:alpine

# Install required tools
RUN apk add --no-cache apache2-utils gettext

# Copy NGINX configuration template
COPY nginx.template.conf /etc/nginx/nginx.template.conf

# Expose HTTPS port
EXPOSE 443

# Replace env variables and start NGINX
CMD ["sh", "-c", "envsubst '${CHROMA_API_KEY}' < /etc/nginx/nginx.template.conf > /etc/nginx/conf.d/default.conf && nginx -g 'daemon off;'"]

# Build and run the NGINX proxy:
#export CHROMA_API_KEY="super-secret-key-123"
#docker build -t nginx-proxy .
#docker run -d \
#    --name nginx-proxy \
#    --network chromadb-network \
#    -v /chroma/nginx/.htpasswd:/etc/nginx/.htpasswd:ro \
#    -v /chroma/nginx/tls:/etc/nginx/tls:ro \
#    -p 443:443 \
#    -e CHROMA_API_KEY=$CHROMA_API_KEY \
#    nginx-proxy
