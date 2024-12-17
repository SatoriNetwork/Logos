import requests
from chromadb.utils.remote_client import RemoteClient
# Disable TLS verification globally for ChromaDB (self-signed certificate)
import ssl
import urllib3
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class CustomRemoteClient(RemoteClient):
    def __init__(self, host, api_key):
        self.api_key = api_key
        super().__init__(host=host)

    def _request(self, *args, **kwargs):
        if "headers" not in kwargs:
            kwargs["headers"] = {}
        kwargs["headers"]["X-API-Key"] = self.api_key
        return super()._request(*args, **kwargs)
