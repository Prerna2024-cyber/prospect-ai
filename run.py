import logging
import os

from prospect_ai.infrastructure.server import OpenAIGateway
from prospect_ai.infrastructure.storage.memory import MemoryCacheBackend

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)

gateway = OpenAIGateway(
    storage_backend=MemoryCacheBackend(),
    upstream_base_url=os.getenv("UPSTREAM_BASE_URL", "https://api.openai.com/v1"),
    upstream_api_key=os.getenv("UPSTREAM_API_KEY", ""),
    similarity_threshold=0.92,
)
app = gateway.app
