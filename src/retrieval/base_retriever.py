# src/retrieval/base_retriever.py
"""
Base class for all retrievers
"""
from abc import ABC, abstractmethod
from src.core.json_storage import JSONStorage


class BaseRetriever(ABC):
    """
    Abstract base class for all layer retrievers
    """
    
    def __init__(self, project_id):
        self.project_id = project_id
        self.name = "BaseRetriever"
        
    @abstractmethod
    def retrieve(self, query, top_k=5):
        """
        Retrieve results for query
        Must be implemented by subclasses
        """
        pass
    
    def format_result(self, content, source, layer, score, metadata=None):
        """Standard result format"""
        return {
            "content": content,
            "source": source,
            "layer": layer,
            "relevance_score": score,
            "metadata": metadata or {}
        }