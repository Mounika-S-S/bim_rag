"""
Model Manager for caching and reusing SentenceTransformer models
to avoid repeated downloads and improve performance.
"""

from sentence_transformers import SentenceTransformer
import threading
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelManager:
    """Singleton manager for SentenceTransformer models"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize the model cache"""
        self._models = {}
        self._loading = set()

    def get_model(self, model_name: str) -> SentenceTransformer:
        """
        Get or load a SentenceTransformer model.
        Models are cached to avoid repeated downloads.
        """
        if model_name not in self._models:
            # Check if another thread is loading this model
            if model_name in self._loading:
                logger.info(f"Waiting for {model_name} to load...")
                while model_name in self._loading:
                    import time
                    time.sleep(0.1)
                return self._models[model_name]

            # Start loading
            self._loading.add(model_name)
            try:
                logger.info(f"Downloading/loading model: {model_name}")
                print(f"Loading {model_name}... This may take a few minutes on first run.", file=sys.stderr)
                model = SentenceTransformer(model_name)
                self._models[model_name] = model
                logger.info(f"Model {model_name} loaded successfully")
                print(f"✓ {model_name} loaded successfully!", file=sys.stderr)
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                raise
            finally:
                self._loading.discard(model_name)

        return self._models[model_name]

    def preload_models(self):
        """Preload commonly used models at startup"""
        common_models = [
            "all-mpnet-base-v2",
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        ]

        logger.info("Preloading ML models for BIM RAG system...")

        for model_name in common_models:
            try:
                self.get_model(model_name)
            except Exception as e:
                logger.warning(f"Failed to preload {model_name}: {e}")
                print(f"⚠️ Failed to load {model_name}: {e}", file=sys.stderr)

        logger.info("Model preloading complete")
        print("✓ All models preloaded successfully!", file=sys.stderr)

# Global instance
model_manager = ModelManager()