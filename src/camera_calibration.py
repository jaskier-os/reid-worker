#!/usr/bin/env python3
"""
Camera Calibration and Coordinate Transformation
Handles GPS coordinates, rotation, and world coordinate calculations
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
import math


@dataclass
class CameraParams:
    """Camera parameters for geo-location and orientation"""
    # GPS position (latitude, longitude, altitude in meters)
    latitude: float  # degrees
    longitude: float  # degrees
    altitude: float  # meters above sea level

    # Camera orientation (in degrees)
    pan: float  # horizontal rotation (0-360, 0=North, 90=East)
    tilt: float  # vertical angle (-90 to 90, 0=horizontal, +90=up, -90=down)
    roll: float  # rotation around optical axis (-180 to 180)

    # Camera intrinsics (optional, can be calibrated)
    focal_length: float = 1000.0  # in pixels
    sensor_width: float = 1920  # image width in pixels
    sensor_height: float = 1080  # image height in pixels
    fov_horizontal: float = 60.0  # horizontal field of view in degrees
    fov_vertical: float = 35.0  # vertical field of view in degrees

    # Additional metadata
    camera_height: float = 2.5  # camera height above ground in meters
    camera_id: str = ""
    camera_name: str = ""


class CoordinateTransformer:
    """Transform between image, camera, and world coordinates"""

    # Earth radius in meters (WGS84)
    EARTH_RADIUS = 6371000

    def __init__(self, camera_params: CameraParams):
        self.params = camera_params
        self._compute_camera_matrix()

    def _compute_camera_matrix(self):
        """Compute camera intrinsic matrix from parameters"""
        fx = self.params.focal_length
        fy = self.params.focal_length
        cx = self.params.sensor_width / 2
        cy = self.params.sensor_height / 2

        self.K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])

        # Compute rotation matrix from pan/tilt/roll
        self.R = self._compute_rotation_matrix(
            self.params.pan,
            self.params.tilt,
            self.params.roll
        )

    def _compute_rotation_matrix(self, pan: float, tilt: float, roll: float) -> np.ndarray:
        """
        Compute 3D rotation matrix from pan/tilt/roll angles

        Args:
            pan: Horizontal rotation in degrees (yaw)
            tilt: Vertical rotation in degrees (pitch)
            roll: Roll rotation in degrees

        Returns:
            3x3 rotation matrix
        """
        # Convert to radians
        pan_rad = math.radians(pan)
        tilt_rad = math.radians(tilt)
        roll_rad = math.radians(roll)

        # Rotation matrices
        R_z = np.array([  # Pan (yaw)
            [math.cos(pan_rad), -math.sin(pan_rad), 0],
            [math.sin(pan_rad), math.cos(pan_rad), 0],
            [0, 0, 1]
        ])

        R_x = np.array([  # Tilt (pitch)
            [1, 0, 0],
            [0, math.cos(tilt_rad), -math.sin(tilt_rad)],
            [0, math.sin(tilt_rad), math.cos(tilt_rad)]
        ])

        R_y = np.array([  # Roll
            [math.cos(roll_rad), 0, math.sin(roll_rad)],
            [0, 1, 0],
            [-math.sin(roll_rad), 0, math.cos(roll_rad)]
        ])

        # Combined rotation: R = R_z * R_x * R_y
        return R_z @ R_x @ R_y

    def pixel_to_ray(self, pixel_x: float, pixel_y: float) -> np.ndarray:
        """
        Convert pixel coordinates to 3D ray in camera coordinates

        Args:
            pixel_x: X coordinate in image
            pixel_y: Y coordinate in image

        Returns:
            3D ray direction vector (normalized)
        """
        # Convert pixel to normalized camera coordinates
        point = np.array([pixel_x, pixel_y, 1.0])
        ray_cam = np.linalg.inv(self.K) @ point

        # Transform to world coordinates
        ray_world = self.R @ ray_cam

        # Normalize
        ray_world = ray_world / np.linalg.norm(ray_world)

        return ray_world

    def calculate_person_location(
        self,
        bbox_center: Tuple[float, float],
        distance: float,
        person_height_pixels: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate person's GPS location from bounding box and distance

        Args:
            bbox_center: (x, y) center of person's bounding box in pixels
            distance: Distance to person in meters
            person_height_pixels: Height of person in pixels (for distance validation)

        Returns:
            Dictionary with lat, lon, altitude, confidence
        """
        # Get ray direction for the person's position
        ray = self.pixel_to_ray(bbox_center[0], bbox_center[1])

        # Calculate 3D position in camera coordinates (meters)
        # Assuming camera is at origin looking along Z-axis
        position_cam = ray * distance

        # Transform to world coordinates
        # Account for camera height above ground
        position_world = position_cam.copy()
        position_world[2] -= self.params.camera_height

        # Convert to GPS coordinates
        lat, lon = self._meters_to_gps(
            position_world[0],  # East-West offset
            position_world[1],  # North-South offset
            self.params.latitude,
            self.params.longitude
        )

        # Calculate altitude (camera altitude + vertical offset)
        altitude = self.params.altitude + position_world[2]

        # Calculate confidence based on distance and detection quality
        confidence = self._calculate_position_confidence(distance, person_height_pixels)

        return {
            'latitude': lat,
            'longitude': lon,
            'altitude': altitude,
            'distance': distance,
            'confidence': confidence,
            'camera_relative_x': position_world[0],
            'camera_relative_y': position_world[1],
            'camera_relative_z': position_world[2]
        }

    def calculate_person_heading(
        self,
        bbox_positions: List[Tuple[float, float, float]],
        timestamps: List[float],
        min_movement_pixels: float = 10.0,
        smoothing_window: int = 5
    ) -> Optional[float]:
        """
        Calculate person's heading direction from multiple observations with smoothing

        Args:
            bbox_positions: List of (x, y, timestamp) positions
            timestamps: List of timestamps for each position
            min_movement_pixels: Minimum pixel movement to consider real motion
            smoothing_window: Number of frames to average over

        Returns:
            Heading in degrees (0-360, 0=North, 90=East) or None if insufficient data
        """
        if len(bbox_positions) < 2:
            return None

        # Calculate all valid headings from recent positions
        valid_headings = []

        # Look at multiple position pairs for stability
        start_idx = max(0, len(bbox_positions) - smoothing_window)

        for i in range(start_idx, len(bbox_positions) - 1):
            pos1 = bbox_positions[i]
            pos2 = bbox_positions[i + 1]

            # Calculate pixel distance moved
            pixel_dx = pos2[0] - pos1[0]
            pixel_dy = pos2[1] - pos1[1]
            pixel_distance = math.sqrt(pixel_dx**2 + pixel_dy**2)

            # Skip if movement is too small (likely detection noise)
            if pixel_distance < min_movement_pixels:
                continue

            # Calculate time difference
            time_diff = timestamps[i + 1] - timestamps[i]
            if time_diff < 0.01:  # Skip if too close in time
                continue

            # Calculate heading from pixel movement
            # In image coordinates: right=+x, down=+y
            # We need to account for camera orientation

            # Simple approach: calculate angle in image space
            angle_rad = math.atan2(pixel_dx, -pixel_dy)  # -dy because image y increases downward
            angle_deg = math.degrees(angle_rad)

            # Normalize to 0-360
            angle_deg = angle_deg % 360

            # Account for camera pan to get world heading
            world_heading = (angle_deg + self.params.pan) % 360

            valid_headings.append(world_heading)

        if not valid_headings:
            return None

        # Use circular mean for angle averaging (important for angles near 0/360)
        sin_sum = sum(math.sin(math.radians(h)) for h in valid_headings)
        cos_sum = sum(math.cos(math.radians(h)) for h in valid_headings)

        avg_heading_rad = math.atan2(sin_sum, cos_sum)
        avg_heading_deg = math.degrees(avg_heading_rad)

        # Normalize to 0-360
        avg_heading_deg = avg_heading_deg % 360

        return avg_heading_deg

    def _meters_to_gps(
        self,
        offset_x: float,
        offset_y: float,
        base_lat: float,
        base_lon: float
    ) -> Tuple[float, float]:
        """
        Convert meter offsets to GPS coordinates

        Args:
            offset_x: East-West offset in meters (positive = East)
            offset_y: North-South offset in meters (positive = North)
            base_lat: Base latitude
            base_lon: Base longitude

        Returns:
            (latitude, longitude) tuple
        """
        # Calculate new latitude
        lat_offset = offset_y / self.EARTH_RADIUS
        new_lat = base_lat + math.degrees(lat_offset)

        # Calculate new longitude (accounting for latitude)
        lon_offset = offset_x / (self.EARTH_RADIUS * math.cos(math.radians(base_lat)))
        new_lon = base_lon + math.degrees(lon_offset)

        return new_lat, new_lon

    def _calculate_position_confidence(
        self,
        distance: float,
        person_height_pixels: Optional[float] = None
    ) -> float:
        """
        Calculate confidence score for position estimation

        Args:
            distance: Distance to person in meters
            person_height_pixels: Height of person in pixels

        Returns:
            Confidence score (0-1)
        """
        # Base confidence
        confidence = 1.0

        # Reduce confidence with distance
        if distance > 50:
            confidence *= max(0.3, 1.0 - (distance - 50) / 100)
        elif distance > 20:
            confidence *= max(0.7, 1.0 - (distance - 20) / 50)

        # Boost confidence if person is large in frame
        if person_height_pixels and person_height_pixels > 100:
            confidence = min(1.0, confidence * 1.1)

        return min(1.0, max(0.0, confidence))


def estimate_distance_from_height(
    bbox_height_pixels: float,
    camera_params: CameraParams,
    assumed_person_height: float = 1.75
) -> float:
    """
    Estimate distance to person based on their height in the image

    Args:
        bbox_height_pixels: Height of person's bounding box in pixels
        camera_params: Camera parameters
        assumed_person_height: Assumed real-world height of person in meters

    Returns:
        Estimated distance in meters
    """
    # Simple pinhole camera model
    # distance = (real_height * focal_length) / pixel_height

    # Adjust focal length based on image size
    focal_length_adjusted = camera_params.focal_length

    # Calculate distance
    distance = (assumed_person_height * focal_length_adjusted) / bbox_height_pixels

    # Account for camera tilt
    tilt_factor = math.cos(math.radians(camera_params.tilt))
    distance *= tilt_factor

    return distance