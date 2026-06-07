#!/usr/bin/env python3
"""
Stream source management for HTTP video feeds
Handles incoming video streams from external sources
"""

import cv2
import numpy as np
import threading
import queue
import time
from typing import Optional, Dict, Any
from datetime import datetime


class StreamSource:
    """Represents a single HTTP video stream source"""

    def __init__(self, stream_id: str, name: Optional[str] = None, camera_params: Optional[Dict[str, Any]] = None):
        """
        Initialize stream source

        Args:
            stream_id: Unique identifier for this stream
            name: Human-readable name for the stream
            camera_params: Camera calibration parameters (GPS, rotation, etc.)
        """
        self.stream_id = stream_id
        self.name = name or f"Stream_{stream_id}"
        self.is_active = False
        self.frame_queue = queue.Queue(maxsize=10)  # Buffer for incoming frames
        self.current_frame = None
        self.current_ir_frame = None  # For IR feed support
        self.last_frame_time = None
        self.frame_count = 0
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

        # Camera parameters for geo-location
        self.camera_params = camera_params or {}
        self._validate_camera_params()

    def _validate_camera_params(self):
        """Validate and set default camera parameters"""
        defaults = {
            'latitude': 0.0,
            'longitude': 0.0,
            'altitude': 0.0,
            'pan': 0.0,
            'tilt': 0.0,
            'roll': 0.0,
            'focal_length': 1000.0,
            'sensor_width': 1920,
            'sensor_height': 1080,
            'fov_horizontal': 60.0,
            'fov_vertical': 35.0,
            'camera_height': 2.5
        }

        for key, default_value in defaults.items():
            if key not in self.camera_params:
                self.camera_params[key] = default_value

    def update_camera_params(self, params: Dict[str, Any]):
        """Update camera parameters"""
        with self.lock:
            self.camera_params.update(params)
            self._validate_camera_params()

    def push_frame(self, frame_data: bytes, is_ir: bool = False) -> bool:
        """
        Push a new frame to the stream

        Args:
            frame_data: JPEG/PNG encoded frame data
            is_ir: Whether this is an IR frame

        Returns:
            True if frame was accepted, False otherwise
        """
        try:
            # Decode frame from bytes
            nparr = np.frombuffer(frame_data, np.uint8)

            if is_ir:
                # IR images are often grayscale
                frame = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            else:
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                return False

            with self.lock:
                if is_ir:
                    self.current_ir_frame = frame
                else:
                    self.current_frame = frame

                self.last_frame_time = datetime.now()
                self.frame_count += 1
                self.is_active = True

                # Add to queue if not full (RGB frames only for now)
                if not is_ir:
                    try:
                        self.frame_queue.put_nowait(frame)
                    except queue.Full:
                        # Drop oldest frame and add new one
                        try:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put_nowait(frame)
                        except:
                            pass

            return True

        except Exception as e:
            print(f"Error decoding {'IR' if is_ir else 'RGB'} frame for {self.name}: {e}")
            return False

    def read_frame(self) -> Optional[np.ndarray]:
        """
        Read the most recent RGB frame

        Returns:
            Frame as numpy array, or None if no frame available
        """
        with self.lock:
            if self.current_frame is not None:
                # Check if stream is stale (no frames for 5 seconds)
                if self.last_frame_time:
                    time_since_last = (datetime.now() - self.last_frame_time).total_seconds()
                    if time_since_last > 5.0:
                        self.is_active = False
                        return None

                return self.current_frame.copy()
        return None

    def read_ir_frame(self) -> Optional[np.ndarray]:
        """
        Read the most recent IR frame

        Returns:
            IR frame as numpy array, or None if no frame available
        """
        with self.lock:
            if self.current_ir_frame is not None:
                return self.current_ir_frame.copy()
        return None

    def close(self):
        """Close the stream"""
        self.stop_event.set()
        self.is_active = False
        with self.lock:
            self.current_frame = None
            # Clear queue
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except:
                    break

    def get_stats(self) -> dict:
        """Get stream statistics"""
        with self.lock:
            return {
                'stream_id': self.stream_id,
                'name': self.name,
                'is_active': self.is_active,
                'frame_count': self.frame_count,
                'last_frame_time': self.last_frame_time.isoformat() if self.last_frame_time else None,
                'has_ir': self.current_ir_frame is not None,
                'camera_params': self.camera_params
            }


class StreamManager:
    """Manages multiple stream sources"""

    def __init__(self):
        self.streams = {}  # stream_id -> StreamSource
        self.lock = threading.Lock()

    def create_stream(self, stream_id: str, name: Optional[str] = None, camera_params: Optional[Dict[str, Any]] = None) -> StreamSource:
        """
        Create a new stream source

        Args:
            stream_id: Unique identifier for the stream
            name: Human-readable name
            camera_params: Camera calibration parameters

        Returns:
            StreamSource instance
        """
        with self.lock:
            if stream_id in self.streams:
                # Update camera params if stream already exists
                if camera_params:
                    self.streams[stream_id].update_camera_params(camera_params)
                return self.streams[stream_id]

            stream = StreamSource(stream_id, name, camera_params)
            self.streams[stream_id] = stream
            print(f"Created new stream: {stream.name} (ID: {stream_id})")
            return stream

    def get_stream(self, stream_id: str) -> Optional[StreamSource]:
        """Get a stream by ID"""
        with self.lock:
            return self.streams.get(stream_id)

    def remove_stream(self, stream_id: str) -> bool:
        """Remove a stream"""
        with self.lock:
            if stream_id in self.streams:
                self.streams[stream_id].close()
                del self.streams[stream_id]
                print(f"Removed stream: {stream_id}")
                return True
        return False

    def get_all_streams(self):
        """Get all active streams"""
        with self.lock:
            return list(self.streams.values())

    def get_active_streams(self):
        """Get all active streams"""
        with self.lock:
            return [s for s in self.streams.values() if s.is_active]

    def cleanup_inactive_streams(self, timeout_seconds: float = 10.0):
        """Remove streams that have been inactive for too long"""
        with self.lock:
            to_remove = []
            current_time = datetime.now()

            for stream_id, stream in self.streams.items():
                if stream.last_frame_time:
                    time_since_last = (current_time - stream.last_frame_time).total_seconds()
                    if time_since_last > timeout_seconds:
                        to_remove.append(stream_id)

            for stream_id in to_remove:
                self.streams[stream_id].close()
                del self.streams[stream_id]
                print(f"Cleaned up inactive stream: {stream_id}")

    def get_stats(self) -> dict:
        """Get statistics for all streams"""
        with self.lock:
            return {
                'total_streams': len(self.streams),
                'active_streams': len([s for s in self.streams.values() if s.is_active]),
                'streams': [s.get_stats() for s in self.streams.values()]
            }

    def close_all(self):
        """Close all streams"""
        with self.lock:
            for stream in self.streams.values():
                stream.close()
            self.streams.clear()
            print("All streams closed")
