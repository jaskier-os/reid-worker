"""
Database Client for Person ReID System
Integrates with reid-db-handler API
"""

import requests
import base64
import numpy as np
from typing import Optional, Dict, List, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class PersonNotFoundError(Exception):
    """Raised when a persistence call targets a person id that no longer exists.

    A person id can vanish when the db-handler auto-merges two persons: the source
    id is deleted and its data moved into the target. The worker must self-heal by
    evicting the dead id and re-resolving to the live (merge target) id.
    """


class TraitsDBClient:
    """Client for interacting with reid-db-handler API"""

    def __init__(self, api_url: str, api_key: str):
        """
        Initialize database client

        Args:
            api_url: Base URL of reid-db-handler API (e.g., http://localhost:3001)
            api_key: API key for authentication
        """
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'x-api-key': api_key,
            'Content-Type': 'application/json'
        })

    def health_check(self) -> Dict[str, Any]:
        """Check API and database health"""
        try:
            response = self.session.get(f"{self.api_url}/health", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"healthy": False, "error": str(e)}

    # ==================== PERSONS ====================

    def create_person(
        self,
        external_id: Optional[str] = None,
        display_name: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Create a new person

        Args:
            external_id: Optional external identifier
            display_name: Human-readable name
            metadata: Additional metadata

        Returns:
            Created person dict with UUID
        """
        payload = {}
        if external_id:
            payload['external_id'] = external_id
        if display_name:
            payload['display_name'] = display_name
        if metadata:
            payload['metadata'] = metadata

        response = self.session.post(f"{self.api_url}/api/persons", json=payload)
        response.raise_for_status()
        person = response.json()
        logger.info(f"Created person: {person['id']}")
        return person

    def find_or_create_person(
        self,
        external_id: str,
        display_name: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Find person by external_id or create if not exists

        Args:
            external_id: External identifier (required)
            display_name: Human-readable name
            metadata: Additional metadata

        Returns:
            Person dict with UUID and 'created' boolean flag
        """
        payload = {'external_id': external_id}
        if display_name:
            payload['display_name'] = display_name
        if metadata:
            payload['metadata'] = metadata

        response = self.session.post(f"{self.api_url}/api/persons/find-or-create", json=payload)
        response.raise_for_status()
        person = response.json()

        if person.get('created', False):
            logger.info(f"Created new person: {person['id']} ({external_id})")
        else:
            logger.debug(f"Found existing person: {person['id']} ({external_id})")

        return person

    def get_person(self, person_id: str) -> Dict[str, Any]:
        """Get person by ID"""
        response = self.session.get(f"{self.api_url}/api/persons/{person_id}")
        response.raise_for_status()
        return response.json()

    def update_person(self, person_id: str, **kwargs) -> Dict[str, Any]:
        """
        Update person

        Args:
            person_id: Person UUID
            **kwargs: Fields to update (display_name, external_id, is_active, metadata)
        """
        response = self.session.put(
            f"{self.api_url}/api/persons/{person_id}",
            json=kwargs
        )
        response.raise_for_status()
        return response.json()

    def list_persons(
        self,
        limit: int = 100,
        offset: int = 0,
        is_active: Optional[bool] = None
    ) -> Dict[str, Any]:
        """List persons with pagination"""
        params = {'limit': limit, 'offset': offset}
        if is_active is not None:
            params['is_active'] = str(is_active).lower()

        response = self.session.get(f"{self.api_url}/api/persons", params=params)
        response.raise_for_status()
        return response.json()

    def get_person_statistics(self, person_id: str) -> Dict[str, Any]:
        """Get statistics for a person"""
        response = self.session.get(
            f"{self.api_url}/api/persons/{person_id}/statistics"
        )
        response.raise_for_status()
        return response.json()

    def merge_persons(self, source_person_id: str, target_person_id: str) -> Dict[str, Any]:
        """
        Merge two person IDs (source into target)

        This moves all traits and sightings from source_person_id to target_person_id
        and deletes the source person.

        Args:
            source_person_id: Person ID to merge from (will be deleted)
            target_person_id: Person ID to merge into (will be kept)

        Returns:
            Dict with merge results (traits_moved, sightings_moved)
        """
        response = self.session.post(
            f"{self.api_url}/api/persons/merge",
            json={
                'source_person_id': source_person_id,
                'target_person_id': target_person_id
            }
        )
        response.raise_for_status()
        result = response.json()
        logger.info(f"Merged person {source_person_id} into {target_person_id}: "
                   f"{result['traits_moved']} traits, {result['sightings_moved']} sightings")
        return result

    # ==================== TRAITS ====================

    def update_person_face3d(self, person_id: str, glb_bytes: bytes) -> Dict[str, Any]:
        """
        Upload a 3D face model (GLB) for a person.

        Args:
            person_id: Person UUID
            glb_bytes: GLB binary data

        Returns:
            Updated person dict
        """
        glb_base64 = base64.b64encode(glb_bytes).decode('utf-8')

        response = self.session.put(
            f"{self.api_url}/api/persons/{person_id}/face3d",
            json={
                'face_model_3d': glb_base64,
                'face_model_format': 'glb'
            }
        )
        if response.status_code == 404:
            raise PersonNotFoundError(person_id)
        response.raise_for_status()
        result = response.json()
        logger.info(f"Uploaded 3D face model for person {person_id}")
        return result

    def get_face3d_frames(self, person_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch the persisted 3D-frame accumulation blob for a person.

        Returns the dict {person_id, frame_count, last_built_at_count,
        schema_version, frames_b64} or None if none exists yet (404).
        """
        response = self.session.get(
            f"{self.api_url}/api/persons/{person_id}/face3d-frames"
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    def put_face3d_frames(
        self,
        person_id: str,
        frame_count: int,
        last_built_at_count: int,
        schema_version: int,
        frames_b64: str
    ) -> Dict[str, Any]:
        """
        Upsert the persisted 3D-frame accumulation blob for a person.

        The worker owns the packed/compressed blob; the db-handler just stores it.
        """
        response = self.session.put(
            f"{self.api_url}/api/persons/{person_id}/face3d-frames",
            json={
                'frame_count': int(frame_count),
                'last_built_at_count': int(last_built_at_count),
                'schema_version': int(schema_version),
                'frames_b64': frames_b64
            }
        )
        if response.status_code == 404:
            raise PersonNotFoundError(person_id)
        response.raise_for_status()
        return response.json()

    def create_trait(
        self,
        person_id: str,
        trait_type: str,
        embedding: np.ndarray,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        quality_score: Optional[float] = None,
        source_camera_id: Optional[str] = None,
        source_timestamp: Optional[datetime] = None,
        source_metadata: Optional[Dict] = None,
        liveness_score: Optional[float] = None,
        liveness_passed: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Create a biometric trait

        Args:
            person_id: Person UUID
            trait_type: Type of trait (face, gait, voice, iris)
            embedding: numpy array embedding vector
            model_name: Model used to generate embedding
            model_version: Model version
            quality_score: Quality/confidence (0-1)
            source_camera_id: Camera that captured this trait
            source_timestamp: When this trait was captured
            source_metadata: Additional metadata
            liveness_score: Anti-spoofing score (0-1)
            liveness_passed: Whether liveness check passed

        Returns:
            Created trait dict with UUID
        """
        # Convert numpy embedding to base64
        embedding_bytes = embedding.tobytes()
        embedding_base64 = base64.b64encode(embedding_bytes).decode('utf-8')

        payload = {
            'person_id': person_id,
            'trait_type': trait_type,
            'embedding': embedding_base64,
            'embedding_dimension': embedding.shape[0]
        }

        if model_name:
            payload['model_name'] = model_name
        if model_version:
            payload['model_version'] = model_version
        if quality_score is not None:
            payload['quality_score'] = float(quality_score)
        if source_camera_id:
            payload['source_camera_id'] = source_camera_id
        if source_timestamp:
            payload['source_timestamp'] = source_timestamp.isoformat()
        if source_metadata:
            payload['source_metadata'] = source_metadata
        if liveness_score is not None:
            payload['liveness_score'] = float(liveness_score)
        if liveness_passed is not None:
            payload['liveness_passed'] = liveness_passed

        response = self.session.post(f"{self.api_url}/api/traits", json=payload)
        response.raise_for_status()
        trait = response.json()
        logger.debug(f"Created trait: {trait['id']} for person {person_id}")
        return trait

    def create_traits_batch(self, traits: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create multiple traits in a batch

        Args:
            traits: List of trait dicts (each with embedding as numpy array)

        Returns:
            Dict with created traits
        """
        # Convert all numpy embeddings to base64
        payload_traits = []
        for trait in traits:
            embedding = trait.pop('embedding')
            embedding_bytes = embedding.tobytes()
            trait['embedding'] = base64.b64encode(embedding_bytes).decode('utf-8')
            trait['embedding_dimension'] = embedding.shape[0]
            payload_traits.append(trait)

        response = self.session.post(
            f"{self.api_url}/api/traits/batch",
            json={'traits': payload_traits}
        )
        response.raise_for_status()
        result = response.json()
        logger.info(f"Batch created {result['created']} traits")
        return result

    def get_trait(self, trait_id: str) -> Dict[str, Any]:
        """Get trait by ID"""
        response = self.session.get(f"{self.api_url}/api/traits/{trait_id}")
        response.raise_for_status()
        return response.json()

    def list_traits(
        self,
        person_id: Optional[str] = None,
        trait_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """List traits with filters"""
        params = {'limit': limit, 'offset': offset}
        if person_id:
            params['person_id'] = person_id
        if trait_type:
            params['trait_type'] = trait_type

        response = self.session.get(f"{self.api_url}/api/traits", params=params)
        response.raise_for_status()
        return response.json()

    # ==================== SIGHTINGS ====================

    def create_sighting(
        self,
        person_id: str,
        camera_id: str,
        detected_at: datetime,
        confidence_score: float,
        bounding_box: Optional[Dict] = None,
        camera_location: Optional[str] = None,
        duration_seconds: Optional[int] = None,
        face_trait_id: Optional[str] = None,
        gait_trait_id: Optional[str] = None,
        frame_number: Optional[int] = None,
        video_source: Optional[str] = None,
        snapshot_path: Optional[str] = None,
        metadata: Optional[Dict] = None,
        person_latitude: Optional[float] = None,
        person_longitude: Optional[float] = None,
        person_altitude: Optional[float] = None,
        distance_meters: Optional[float] = None,
        heading_degrees: Optional[float] = None,
        gaze_degrees: Optional[float] = None,
        speed_ms: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Create a sighting record

        Returns:
            Created sighting dict with UUID
        """
        payload = {
            'person_id': str(person_id),
            'camera_id': str(camera_id),
            'detected_at': detected_at.isoformat(),
            'confidence_score': float(confidence_score)
        }

        if bounding_box:
            payload['bounding_box'] = bounding_box
        if camera_location:
            payload['camera_location'] = camera_location
        if duration_seconds is not None:
            payload['duration_seconds'] = duration_seconds
        if face_trait_id:
            payload['face_trait_id'] = face_trait_id
        if gait_trait_id:
            payload['gait_trait_id'] = gait_trait_id
        if frame_number is not None:
            payload['frame_number'] = frame_number
        if video_source:
            payload['video_source'] = video_source
        if snapshot_path:
            payload['snapshot_path'] = snapshot_path
        if metadata:
            payload['metadata'] = metadata
        if person_latitude is not None:
            payload['person_latitude'] = float(person_latitude)
        if person_longitude is not None:
            payload['person_longitude'] = float(person_longitude)
        if person_altitude is not None:
            payload['person_altitude'] = float(person_altitude)
        if distance_meters is not None:
            payload['distance_meters'] = float(distance_meters)
        if heading_degrees is not None:
            payload['heading_degrees'] = float(heading_degrees)
        if gaze_degrees is not None:
            payload['gaze_degrees'] = float(gaze_degrees)
        if speed_ms is not None:
            payload['speed_ms'] = float(speed_ms)

        response = self.session.post(f"{self.api_url}/api/sightings", json=payload)
        response.raise_for_status()
        sighting = response.json()
        logger.debug(f"Created sighting: {sighting['id']}")
        return sighting

    def create_sightings_batch(self, sightings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create multiple sightings in a batch

        Args:
            sightings: List of sighting dicts

        Returns:
            Dict with created sightings
        """
        # Convert datetimes to ISO format
        for sighting in sightings:
            if 'detected_at' in sighting and isinstance(sighting['detected_at'], datetime):
                sighting['detected_at'] = sighting['detected_at'].isoformat()

        response = self.session.post(
            f"{self.api_url}/api/sightings/batch",
            json={'sightings': sightings}
        )
        response.raise_for_status()
        result = response.json()
        logger.info(f"Batch created {result['created']} sightings")
        return result

    def get_sighting(self, sighting_id: str) -> Dict[str, Any]:
        """Get sighting by ID"""
        response = self.session.get(f"{self.api_url}/api/sightings/{sighting_id}")
        response.raise_for_status()
        return response.json()

    def list_sightings(
        self,
        person_id: Optional[str] = None,
        camera_id: Optional[str] = None,
        recent: bool = False,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """List sightings with filters"""
        params = {'limit': limit, 'offset': offset}
        if person_id:
            params['person_id'] = person_id
        if camera_id:
            params['camera_id'] = camera_id
        if recent:
            params['recent'] = 'true'

        response = self.session.get(f"{self.api_url}/api/sightings", params=params)
        response.raise_for_status()
        return response.json()

    def get_recent_sightings(self, limit: int = 100) -> Dict[str, Any]:
        """Get recent sightings (last 24 hours)"""
        response = self.session.get(
            f"{self.api_url}/api/sightings/recent",
            params={'limit': limit}
        )
        response.raise_for_status()
        return response.json()

    # ==================== FAISS OPERATIONS ====================

    def faiss_search(self, embedding: np.ndarray, threshold: float = 0.4) -> Dict[str, Any]:
        """
        Search for a face in FAISS database

        Args:
            embedding: Face embedding vector (512 dimensions)
            threshold: Similarity threshold (0-1)

        Returns:
            Dict with name and similarity score
        """
        # Convert numpy array to list
        embedding_list = embedding.tolist()

        response = self.session.post(
            f"{self.api_url}/api/faiss/search",
            json={
                'embedding': embedding_list,
                'threshold': threshold
            }
        )
        response.raise_for_status()
        result = response.json()
        logger.debug(f"FAISS search result: {result.get('name')} (similarity: {result.get('similarity'):.3f})")
        return result

    def faiss_batch_search(
        self,
        embeddings: List[np.ndarray],
        threshold: float = 0.4
    ) -> List[Dict[str, Any]]:
        """
        Batch search for multiple faces

        Args:
            embeddings: List of face embedding vectors
            threshold: Similarity threshold

        Returns:
            List of dicts with name and similarity scores
        """
        # Convert numpy arrays to lists
        embeddings_list = [emb.tolist() for emb in embeddings]

        response = self.session.post(
            f"{self.api_url}/api/faiss/search/batch",
            json={
                'embeddings': embeddings_list,
                'threshold': threshold
            }
        )
        response.raise_for_status()
        result = response.json()
        logger.info(f"FAISS batch search completed: {result.get('count')} results")
        return result.get('results', [])

    def faiss_add_face(self, embedding: np.ndarray, person_id: str) -> Dict[str, Any]:
        """
        Add a face to FAISS database

        Args:
            embedding: Face embedding vector
            person_id: Person identifier

        Returns:
            Dict with success status and total count
        """
        # Convert numpy array to list
        embedding_list = embedding.tolist()

        response = self.session.post(
            f"{self.api_url}/api/faiss/add",
            json={
                'embedding': embedding_list,
                'name': person_id
            }
        )
        response.raise_for_status()
        result = response.json()
        logger.info(f"Added face for person {person_id} to FAISS (total: {result.get('total')})")
        return result

    def faiss_add_faces_batch(
        self,
        embeddings: List[np.ndarray],
        person_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Add multiple faces to FAISS database

        Args:
            embeddings: List of face embedding vectors
            person_ids: List of person identifiers

        Returns:
            Dict with success status and count
        """
        if len(embeddings) != len(person_ids):
            raise ValueError("embeddings and person_ids must have same length")

        # Convert numpy arrays to lists
        embeddings_list = [emb.tolist() for emb in embeddings]

        response = self.session.post(
            f"{self.api_url}/api/faiss/add/batch",
            json={
                'embeddings': embeddings_list,
                'names': person_ids
            }
        )
        response.raise_for_status()
        result = response.json()
        logger.info(f"Batch added {result.get('count')} faces to FAISS")
        return result

    def faiss_remove_person(self, person_id: str) -> Dict[str, Any]:
        """Remove all face embeddings for a person from FAISS and save"""
        response = self.session.delete(f"{self.api_url}/api/faiss/person/{person_id}")
        response.raise_for_status()
        result = response.json()
        # Save after removal
        self.faiss_save()
        logger.info(f"Removed person {person_id} from FAISS (removed: {result.get('removed', 0)})")
        return result

    def faiss_save(self) -> Dict[str, Any]:
        """Save FAISS database to disk"""
        response = self.session.post(f"{self.api_url}/api/faiss/save")
        response.raise_for_status()
        result = response.json()
        logger.info("FAISS database saved")
        return result

    def faiss_get_stats(self) -> Dict[str, Any]:
        """Get FAISS database statistics"""
        response = self.session.get(f"{self.api_url}/api/faiss/stats")
        response.raise_for_status()
        return response.json()

    # ==================== SIMILARITY PROGRESS ====================

    def get_similarity_progress(self, person_id: str = None) -> Dict[str, Any]:
        """Get similarity analysis progress"""
        if person_id:
            response = self.session.get(
                f"{self.api_url}/api/similarity/progress/{person_id}")
        else:
            response = self.session.get(f"{self.api_url}/api/similarity/progress")
        response.raise_for_status()
        return response.json()

    def update_similarity_progress(self, person_id: str, compared_count: int) -> Dict[str, Any]:
        """Update similarity progress for a person"""
        response = self.session.put(
            f"{self.api_url}/api/similarity/progress/{person_id}",
            json={'compared_count': compared_count})
        response.raise_for_status()
        return response.json()

    def upsert_similarity_relationship(
        self, person_a_id: str, person_b_id: str, similarity_score: float
    ) -> Dict[str, Any]:
        """Create or update a similar_face relationship between two persons"""
        response = self.session.put(
            f"{self.api_url}/api/similarity/relationship",
            json={
                'person_a_id': person_a_id,
                'person_b_id': person_b_id,
                'similarity_score': similarity_score
            })
        response.raise_for_status()
        return response.json()

    def reset_similarity_progress(self, person_id: str = None) -> Dict[str, Any]:
        """Reset similarity progress (all or specific person)"""
        if person_id:
            response = self.session.delete(
                f"{self.api_url}/api/similarity/progress/{person_id}")
        else:
            response = self.session.delete(
                f"{self.api_url}/api/similarity/progress")
        response.raise_for_status()
        return response.json()
