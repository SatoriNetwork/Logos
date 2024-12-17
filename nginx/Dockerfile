# Base image
FROM nginx:alpine

# Set up a basic auth password
RUN apk add --no-cache apache2-utils && \
    echo "admin:$(htpasswd -nb admin <password> | cut -d ":" -f 2)" > /etc/nginx/.htpasswd

# Add NGINX configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose HTTP port
EXPOSE 80
