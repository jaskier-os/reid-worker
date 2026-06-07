#!/usr/bin/env python3
"""
Sighting Aggregator
Merges sightings within time windows to prevent duplicate entries
"""

import threading
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class SightingData:
    """Data for a single sighting"""
    person_id: str
    camera_id: str
    timestamp: datetime
    bbox: Tuple[float, float, float, float]
    confidence: float
    embedding: Optional[np.ndarray] = None

    # Location data
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: Optional[float] = None
    distance: Optional[float] = None
    heading: Optional[float] = None
    gaze_degrees: Optional[float] = None

    # Additional metadata
    face_trait_id: Optional[str] = None
    gait_trait_id: Optional[str] = None
    frame_number: Optional[int] = None
    snapshot_path: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class AggregatedSighting:
    """Aggregated sighting data over a time window"""
    person_id: str
    camera_id: str
    first_seen: datetime
    last_seen: datetime
    sighting_count: int

    # Aggregated location (weighted average)
    avg_latitude: Optional[float] = None
    avg_longitude: Optional[float] = None
    avg_altitude: Optional[float] = None
    avg_distance: Optional[float] = None

    # Movement data
    heading: Optional[float] = None  # Overall heading during window
    gaze_degrees: Optional[float] = None  # Gaze compass heading (0-360)
    speed_ms: Optional[float] = None  # Estimated speed in m/s

    # Best quality data
    best_confidence: float = 0.0
    best_embedding: Optional[np.ndarray] = None
    best_bbox: Optional[Tuple[float, float, float, float]] = None
    best_snapshot_path: Optional[str] = None

    # Trait IDs (keep best quality ones)
    face_trait_ids: List[str] = field(default_factory=list)
    gait_trait_ids: List[str] = field(default_factory=list)

    # Combined metadata
    metadata: Dict = field(default_factory=dict)

    # Individual sightings for detailed analysis
    raw_sightings: List[SightingData] = field(default_factory=list)


class SightingAggregator:
    """
    Aggregates sightings within time windows to prevent database pollution
    """

    def __init__(
        self,
        window_seconds: float = 5.0,
        max_gap_seconds: float = 2.0,
        min_sightings_to_aggregate: int = 2
    ):
        """
        Initialize sighting aggregator

        Args:
            window_seconds: Time window for aggregation
            max_gap_seconds: Maximum gap between sightings to consider them continuous
            min_sightings_to_aggregate: Minimum sightings needed to create aggregated entry
        """
        self.window_seconds = window_seconds
        self.max_gap_seconds = max_gap_seconds
        self.min_sightings = min_sightings_to_aggregate

        # Active sighting windows: (person_id, camera_id) -> AggregatedSighting
        self.active_windows: Dict[Tuple[str, str], AggregatedSighting] = {}

        # Completed sightings ready for database
        self.completed_sightings: List[AggregatedSighting] = []

        # Thread safety
        self.lock = threading.Lock()

    def add_sighting(self, sighting: SightingData) -> Optional[AggregatedSighting]:
        """
        Add a new sighting and potentially return a completed aggregation

        Args:
            sighting: New sighting data

        Returns:
            Completed aggregated sighting if window closed, None otherwise
        """
        with self.lock:
            key = (sighting.person_id, sighting.camera_id)
            current_time = sighting.timestamp

            # Check if we have an active window for this person/camera
            if key in self.active_windows:
                window = self.active_windows[key]

                # Check if this sighting is within the window
                time_since_first = (current_time - window.first_seen).total_seconds()
                time_since_last = (current_time - window.last_seen).total_seconds()

                if (time_since_first <= self.window_seconds and
                    time_since_last <= self.max_gap_seconds):
                    # Add to existing window
                    self._update_window(window, sighting)
                    return None
                else:
                    # Window expired, close it and start new one
                    completed = self._close_window(key)
                    self._start_new_window(key, sighting)
                    return completed
            else:
                # Start new window
                self._start_new_window(key, sighting)
                return None

    def _start_new_window(self, key: Tuple[str, str], sighting: SightingData):
        """Start a new aggregation window"""
        window = AggregatedSighting(
            person_id=sighting.person_id,
            camera_id=sighting.camera_id,
            first_seen=sighting.timestamp,
            last_seen=sighting.timestamp,
            sighting_count=1,
            best_confidence=sighting.confidence,
            best_embedding=sighting.embedding,
            best_bbox=sighting.bbox,
            best_snapshot_path=sighting.snapshot_path
        )

        # Set initial location
        if sighting.latitude is not None:
            window.avg_latitude = sighting.latitude
            window.avg_longitude = sighting.longitude
            window.avg_altitude = sighting.altitude
            window.avg_distance = sighting.distance

        # Set initial gaze
        if sighting.gaze_degrees is not None:
            window.gaze_degrees = sighting.gaze_degrees

        # Add trait IDs
        if sighting.face_trait_id:
            window.face_trait_ids.append(sighting.face_trait_id)
        if sighting.gait_trait_id:
            window.gait_trait_ids.append(sighting.gait_trait_id)

        # Store raw sighting
        window.raw_sightings.append(sighting)

        # Copy metadata
        window.metadata.update(sighting.metadata)

        self.active_windows[key] = window

    def _update_window(self, window: AggregatedSighting, sighting: SightingData):
        """Update an existing aggregation window with new sighting"""
        window.last_seen = sighting.timestamp
        window.sighting_count += 1

        # Update best quality data
        if sighting.confidence > window.best_confidence:
            window.best_confidence = sighting.confidence
            window.best_embedding = sighting.embedding
            window.best_bbox = sighting.bbox
            if sighting.snapshot_path:
                window.best_snapshot_path = sighting.snapshot_path

        # Update location (weighted average)
        if sighting.latitude is not None and window.avg_latitude is not None:
            # Weight by confidence
            total_weight = window.best_confidence * (window.sighting_count - 1) + sighting.confidence

            window.avg_latitude = (
                window.avg_latitude * window.best_confidence * (window.sighting_count - 1) +
                sighting.latitude * sighting.confidence
            ) / total_weight

            window.avg_longitude = (
                window.avg_longitude * window.best_confidence * (window.sighting_count - 1) +
                sighting.longitude * sighting.confidence
            ) / total_weight

            if sighting.altitude is not None and window.avg_altitude is not None:
                window.avg_altitude = (
                    window.avg_altitude * window.best_confidence * (window.sighting_count - 1) +
                    sighting.altitude * sighting.confidence
                ) / total_weight

            if sighting.distance is not None and window.avg_distance is not None:
                window.avg_distance = (
                    window.avg_distance * window.best_confidence * (window.sighting_count - 1) +
                    sighting.distance * sighting.confidence
                ) / total_weight
        elif sighting.latitude is not None:
            # First location data
            window.avg_latitude = sighting.latitude
            window.avg_longitude = sighting.longitude
            window.avg_altitude = sighting.altitude
            window.avg_distance = sighting.distance

        # Update heading if available
        if sighting.heading is not None:
            window.heading = sighting.heading

        # Update gaze (use latest with best confidence)
        if sighting.gaze_degrees is not None:
            if sighting.confidence >= window.best_confidence:
                window.gaze_degrees = sighting.gaze_degrees

        # Add unique trait IDs
        if sighting.face_trait_id and sighting.face_trait_id not in window.face_trait_ids:
            window.face_trait_ids.append(sighting.face_trait_id)
        if sighting.gait_trait_id and sighting.gait_trait_id not in window.gait_trait_ids:
            window.gait_trait_ids.append(sighting.gait_trait_id)

        # Store raw sighting
        window.raw_sightings.append(sighting)

        # Merge metadata
        window.metadata.update(sighting.metadata)

        # Calculate movement if we have enough location data
        if len(window.raw_sightings) >= 2:
            self._calculate_movement(window)

    def _calculate_movement(self, window: AggregatedSighting):
        """Calculate movement speed and refined heading from raw sightings"""
        sightings_with_location = [
            s for s in window.raw_sightings
            if s.latitude is not None and s.longitude is not None
        ]

        if len(sightings_with_location) < 2:
            return

        # Get first and last positions
        first = sightings_with_location[0]
        last = sightings_with_location[-1]

        # Calculate distance moved (simple Euclidean for small distances)
        # For more accuracy, use Haversine formula
        lat_diff = last.latitude - first.latitude
        lon_diff = last.longitude - first.longitude

        # Convert to meters (approximate)
        lat_meters = lat_diff * 111000  # 1 degree latitude ≈ 111 km
        lon_meters = lon_diff * 111000 * np.cos(np.radians(first.latitude))

        distance_moved = np.sqrt(lat_meters**2 + lon_meters**2)

        # Calculate time elapsed
        time_elapsed = (last.timestamp - first.timestamp).total_seconds()

        if time_elapsed > 0:
            # Calculate speed
            window.speed_ms = distance_moved / time_elapsed

            # Calculate heading if not already set
            if window.heading is None and distance_moved > 0.5:  # Min 0.5m movement
                # Calculate heading from movement
                heading_rad = np.arctan2(lon_diff, lat_diff)
                heading_deg = np.degrees(heading_rad)
                # Normalize to 0-360
                if heading_deg < 0:
                    heading_deg += 360
                window.heading = heading_deg

    def _close_window(self, key: Tuple[str, str]) -> Optional[AggregatedSighting]:
        """Close an active window and return the aggregated sighting"""
        if key not in self.active_windows:
            return None

        window = self.active_windows[key]
        del self.active_windows[key]

        # Only return if we have enough sightings
        if window.sighting_count >= self.min_sightings:
            # Calculate duration
            duration = (window.last_seen - window.first_seen).total_seconds()
            window.metadata['duration_seconds'] = duration
            window.metadata['aggregated_count'] = window.sighting_count

            self.completed_sightings.append(window)
            return window

        return None

    def flush_expired_windows(
        self,
        current_time: Optional[datetime] = None
    ) -> List[AggregatedSighting]:
        """
        Flush all expired windows

        Args:
            current_time: Current time for expiration check

        Returns:
            List of completed aggregated sightings
        """
        if current_time is None:
            current_time = datetime.now()

        completed = []

        with self.lock:
            keys_to_close = []

            for key, window in self.active_windows.items():
                time_since_last = (current_time - window.last_seen).total_seconds()

                if time_since_last > self.max_gap_seconds:
                    keys_to_close.append(key)

            for key in keys_to_close:
                aggregated = self._close_window(key)
                if aggregated:
                    completed.append(aggregated)

        return completed

    def get_completed_sightings(self) -> List[AggregatedSighting]:
        """
        Get and clear completed sightings

        Returns:
            List of completed aggregated sightings
        """
        with self.lock:
            completed = self.completed_sightings.copy()
            self.completed_sightings.clear()
            return completed

    def get_active_window_stats(self) -> Dict:
        """Get statistics about active aggregation windows"""
        with self.lock:
            return {
                'active_windows': len(self.active_windows),
                'total_sightings': sum(w.sighting_count for w in self.active_windows.values()),
                'windows': [
                    {
                        'person_id': w.person_id,
                        'camera_id': w.camera_id,
                        'sighting_count': w.sighting_count,
                        'duration': (w.last_seen - w.first_seen).total_seconds()
                    }
                    for w in self.active_windows.values()
                ]
            }