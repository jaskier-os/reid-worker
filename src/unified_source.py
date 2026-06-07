#!/usr/bin/env python3
"""
Unified source interface for cameras and streams
Provides a common interface for both camera and HTTP stream sources
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class VideoSource(ABC):
    """Abstract base class for video sources"""

    @property
    @abstractmethod
    def source_id(self) -> str:
        """Get unique identifier for this source"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Get human-readable name"""
        pass

    @property
    @abstractmethod
    def is_active(self) -> bool:
        """Check if source is active"""
        pass

    @abstractmethod
    def read_frame(self) -> Optional[np.ndarray]:
        """Read a frame from the source"""
        pass

    @abstractmethod
    def close(self):
        """Close the source"""
        pass


class CameraSourceWrapper(VideoSource):
    """Wrapper to make CameraSource compatible with VideoSource interface"""

    def __init__(self, camera_source):
        self._source = camera_source

    @property
    def source_id(self) -> str:
        return str(self._source.camera_id)

    @property
    def name(self) -> str:
        return self._source.name

    @property
    def is_active(self) -> bool:
        return self._source.is_active

    def read_frame(self) -> Optional[np.ndarray]:
        return self._source.read_frame()

    def close(self):
        self._source.close()


class StreamSourceWrapper(VideoSource):
    """Wrapper to make StreamSource compatible with VideoSource interface"""

    def __init__(self, stream_source):
        self._source = stream_source

    @property
    def source_id(self) -> str:
        return self._source.stream_id

    @property
    def name(self) -> str:
        return self._source.name

    @property
    def is_active(self) -> bool:
        return self._source.is_active

    def read_frame(self) -> Optional[np.ndarray]:
        return self._source.read_frame()

    def close(self):
        self._source.close()
