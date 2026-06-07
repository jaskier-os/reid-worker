"""
Remote FAISS Face Database Client
Provides same interface as local FaceDatabase but uses centralized API
"""

import numpy as np
from typing import Tuple, List
from src.db_client import TraitsDBClient
import logging

logger = logging.getLogger(__name__)


class RemoteFaceDatabase:
    """
    Remote face database that uses centralized FAISS API
    Compatible with existing FaceDatabase interface
    """

    def __init__(self, api_url: str, api_key: str, embedding_size: int = 512):
        """
        Initialize remote face database

        Args:
            api_url: Base URL of reid-db-handler API
            api_key: API key for authentication
            embedding_size: Dimension of face embeddings (default 512)
        """
        self.embedding_size = embedding_size
        self.client = TraitsDBClient(api_url, api_key)
        logger.info(f"Initialized remote face database (API: {api_url})")

    def add_face(self, embedding: np.ndarray, name: str) -> None:
        """
        Add a face embedding to the remote database

        Args:
            embedding: Face embedding vector
            name: Name of the person
        """
        try:
            self.client.faiss_add_face(embedding, name)
        except Exception as e:
            logger.error(f"Failed to add face to remote database: {e}")
            raise

    def search(self, embedding: np.ndarray, threshold: float = 0.4) -> Tuple[str, float]:
        """
        Search for the closest face in the database

        Args:
            embedding: Query face embedding
            threshold: Similarity threshold

        Returns:
            Tuple containing the name and similarity score of the best match
        """
        try:
            result = self.client.faiss_search(embedding, threshold)
            return (result.get('name', 'Unknown'), result.get('similarity', 0.0))
        except Exception as e:
            logger.error(f"Failed to search in remote database: {e}")
            return ("Unknown", 0.0)

    def batch_search(self, embeddings: List[np.ndarray], threshold: float = 0.4) -> List[Tuple[str, float]]:
        """
        Perform batch search for multiple face embeddings

        Args:
            embeddings: List of face embeddings to search for
            threshold: Similarity threshold

        Returns:
            List of tuples containing names and similarity scores
        """
        if not embeddings:
            return []

        try:
            results = self.client.faiss_batch_search(embeddings, threshold)
            return [(r.get('name', 'Unknown'), r.get('similarity', 0.0)) for r in results]
        except Exception as e:
            logger.error(f"Failed to batch search in remote database: {e}")
            # Return Unknown for all if API call fails
            return [("Unknown", 0.0) for _ in embeddings]

    def add_faces_batch(self, embeddings: List[np.ndarray], names: List[str]) -> None:
        """
        Add multiple faces to the database

        Args:
            embeddings: List of face embeddings
            names: List of corresponding names
        """
        try:
            self.client.faiss_add_faces_batch(embeddings, names)
        except Exception as e:
            logger.error(f"Failed to batch add faces to remote database: {e}")
            raise

    def save(self) -> None:
        """
        Save the FAISS index to disk (on the server)
        """
        try:
            self.client.faiss_save()
            logger.info("Remote face database saved")
        except Exception as e:
            logger.error(f"Failed to save remote database: {e}")
            # Don't raise - saving is not critical

    def load(self) -> bool:
        """
        Load operation is not needed for remote database
        Database is automatically loaded on the server side

        Returns:
            bool: Always returns True (no-op)
        """
        # Get stats to verify connection
        try:
            stats = self.client.faiss_get_stats()
            total = stats.get('total_faces', 0)
            logger.info(f"Remote database ready with {total} faces")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to remote database: {e}")
            return False

    def close(self):
        """
        Close the database connection
        For remote database, this just logs the operation
        """
        logger.info("Closing remote face database connection")
        # No actual cleanup needed for HTTP client

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.close()
        except:
            pass


def create_face_database(api_url: str = None, api_key: str = None, db_path: str = None, embedding_size: int = 512):
    """
    Factory function to create face database (remote or local)

    Args:
        api_url: If provided, creates remote database
        api_key: API key for remote database
        db_path: Path for local database (ignored if api_url provided)
        embedding_size: Embedding dimension

    Returns:
        RemoteFaceDatabase or FaceDatabase instance
    """
    if api_url and api_key:
        logger.info("Using remote centralized FAISS database")
        return RemoteFaceDatabase(api_url, api_key, embedding_size)
    else:
        # Fallback to local database
        logger.info("Using local FAISS database")
        from database.face_db import FaceDatabase
        return FaceDatabase(embedding_size, db_path)
