"""
Head Pose Estimation via solvePnP
Estimates head pose from SCRFD's 5 facial keypoints (left_eye, right_eye, nose, left_mouth, right_mouth).
"""

import cv2
import numpy as np
from typing import Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Standard 3D reference points for a 5-point face model (in face-centered coords).
# Order: left_eye, right_eye, nose, left_mouth, right_mouth
_FACE_3D_POINTS = np.array([
    [-30.0,  30.0, -30.0],  # left eye
    [ 30.0,  30.0, -30.0],  # right eye
    [  0.0,   0.0,   0.0],  # nose tip
    [-20.0, -30.0, -20.0],  # left mouth corner
    [ 20.0, -30.0, -20.0],  # right mouth corner
], dtype=np.float64)


@dataclass
class HeadPose:
    """Full head pose estimation result"""
    yaw: float    # degrees, positive = turned right
    pitch: float  # degrees, positive = looking up
    roll: float   # degrees, positive = tilted clockwise
    rvec: np.ndarray   # rotation vector for cv2.projectPoints
    tvec: np.ndarray   # translation vector for cv2.projectPoints
    camera_matrix: np.ndarray  # camera intrinsic matrix


def estimate_head_pose(
    keypoints: np.ndarray,
    image_width: int,
    image_height: int
) -> Optional[HeadPose]:
    """
    Estimate full head pose from 5 SCRFD facial keypoints.

    Args:
        keypoints: (5, 2) array of facial keypoints from SCRFD
        image_width: width of the source frame
        image_height: height of the source frame

    Returns:
        HeadPose with yaw/pitch/roll in degrees plus rvec/tvec for projection,
        or None on failure.
    """
    try:
        kp = np.asarray(keypoints, dtype=np.float64)
        if kp.shape != (5, 2):
            return None

        # Approximate camera intrinsics from image dimensions
        focal_length = float(image_width)
        cx = image_width / 2.0
        cy = image_height / 2.0

        camera_matrix = np.array([
            [focal_length, 0.0, cx],
            [0.0, focal_length, cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            _FACE_3D_POINTS, kp, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_SQPNP
        )

        if not success:
            return None

        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rvec)

        # Decompose rotation matrix to Euler angles (yaw, pitch, roll)
        sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)

        if sy > 1e-6:
            pitch = float(np.degrees(np.arctan2(-rotation_matrix[2, 1], rotation_matrix[2, 2])))
            yaw = float(np.degrees(np.arctan2(rotation_matrix[2, 0], sy)))
            roll = float(np.degrees(np.arctan2(-rotation_matrix[1, 0], rotation_matrix[0, 0])))
        else:
            pitch = float(np.degrees(np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])))
            yaw = float(np.degrees(np.arctan2(rotation_matrix[2, 0], sy)))
            roll = 0.0

        # Clamp yaw to reasonable range
        if abs(yaw) > 90:
            return None

        return HeadPose(
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            rvec=rvec,
            tvec=tvec,
            camera_matrix=camera_matrix
        )

    except Exception as e:
        logger.debug(f"Head pose estimation failed: {e}")
        return None
