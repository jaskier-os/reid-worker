#!/usr/bin/env python3
"""
Person ReID Multi-Camera and Stream Server System
Main entrypoint supporting both local cameras and remote video streams via HTTP API

Modes:
- Camera mode: Process local cameras directly (legacy functionality)
- Stream server mode: HTTP API for remote video sources (default)
- Debug mode: GUI with InsightFace feature visualization and hotkeys

Hotkeys (Debug Mode):
- q: Quit
- s: Save screenshot
- r: Reset statistics
- SPACE: Pause/Resume
- 1-9: Toggle specific cameras
- h: Show/hide help overlay
"""

import os
import sys
from pathlib import Path

# Suppress Qt font warnings from OpenCV's Qt backend
os.environ.setdefault('QT_LOGGING_RULES', '*.warning=false')

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "face-reidentification"))

import cv2
import argparse
import threading
import queue
import time
import uuid
import pickle
import numpy as np
import configparser
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from collections import deque

# Flask for stream server mode
from flask import Flask, request, jsonify
from flask_cors import CORS

# ReID components
from src.stream_source import StreamManager
from src.camera_calibration import CameraParams, CoordinateTransformer
from src.sighting_aggregator import SightingAggregator, SightingData
from src.db_client import TraitsDBClient
from src.remote_face_db import create_face_database
from src.head_pose import estimate_head_pose
from src.face3d_store import Face3DAccumulator
from models.scrfd import SCRFD
from models.arcface import ArcFace
from models.antispoof import AntiSpoof
from models.face3d import Face3DReconstructor

import base64
import requests as http_requests

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# PERSON TRACKER
# ============================================================================

class PersonTracker:
    """Tracks persons across multiple cameras"""
    def __init__(self, min_samples_for_registration: int = 15,
                 refine_interval: int = 30,
                 auto_merge_threshold: float = 0.85,
                 merge_time_window_seconds: float = 30.0,
                 disappear_timeout: float = 2.0):
        self.persons = {}  # UUID -> person data
        self.lock = threading.Lock()
        self.min_samples = min_samples_for_registration
        self.refine_interval = refine_interval
        self.registered_to_db = set()
        self.last_registered_count = {}  # person_id -> embedding count at last FAISS update
        self.auto_merge_threshold = auto_merge_threshold
        self.merge_time_window = merge_time_window_seconds
        self.disappear_timeout = disappear_timeout
        self.recently_disappeared = {}
        self.last_sighting_sent_at = {}  # person_id -> datetime of last DB sighting

    def remove_person(self, person_id: str):
        """Remove a person from in-memory tracking (e.g. after deletion from DB)."""
        with self.lock:
            self.persons.pop(person_id, None)
            self.recently_disappeared.pop(person_id, None)
            self.last_sighting_sent_at.pop(person_id, None)

    def find_or_create_person(self, embedding: np.ndarray, camera_id,
                              bbox: tuple, db, similarity_threshold: float = 0.4) -> str:
        """Find existing person or create new one"""
        current_time = datetime.now()

        # FAISS search outside lock to avoid blocking other camera threads
        matched_name, similarity = db.search(embedding)

        # Validate FAISS match against DB if we haven't seen this person this session
        if matched_name and matched_name != "Unknown" and similarity >= similarity_threshold:
            if matched_name not in self.persons:
                try:
                    db.get_person(matched_name)
                except Exception:
                    # Person was deleted from DB - clean up stale FAISS entry and in-memory cache
                    logger.info(f"Stale FAISS entry for deleted person {matched_name}, removing")
                    try:
                        db.faiss_remove_person(matched_name)
                    except Exception as e:
                        logger.warning(f"Failed to remove stale FAISS entry: {e}")
                    self.remove_person(matched_name)
                    matched_name = None
                    similarity = 0.0

        with self.lock:
            if matched_name and matched_name != "Unknown" and similarity >= similarity_threshold:
                person_id = matched_name
            else:
                # Check for recently disappeared persons
                merge_candidate = self.check_for_merge_candidates(embedding, camera_id, current_time)

                if merge_candidate:
                    person_id = merge_candidate
                    if merge_candidate in self.recently_disappeared:
                        del self.recently_disappeared[merge_candidate]
                    logger.info(f"Person {merge_candidate} reappeared after face obstruction")
                else:
                    # Check ALL tracked persons in memory (known and unknown)
                    best_match_id = None
                    best_similarity = 0.0

                    for pid, data in self.persons.items():
                        if len(data['embeddings']) > 0:
                            # Compare against average and best individual embedding
                            avg_embedding = np.mean(data['embeddings'], axis=0)
                            avg_sim = self._compute_similarity(embedding, avg_embedding)

                            # Also check most recent embeddings individually
                            recent = data['embeddings'][-5:]
                            max_individual = max(
                                self._compute_similarity(embedding, e) for e in recent
                            )

                            sim = max(avg_sim, max_individual)
                            if sim > best_similarity:
                                best_similarity = sim
                                best_match_id = pid

                    if best_match_id and best_similarity >= 0.55:
                        person_id = best_match_id
                    else:
                        # New unknown person
                        person_id = str(uuid.uuid4())[:8]
                        self.persons[person_id] = {
                            'embeddings': [],
                            'sightings': [],
                            'known': False,
                            'created_at': datetime.now()
                        }

            # Update person data
            if person_id not in self.persons:
                self.persons[person_id] = {
                    'embeddings': [],
                    'sightings': [],
                    'known': True,
                    'created_at': datetime.now()
                }

            self.persons[person_id]['embeddings'].append(embedding)
            self.persons[person_id]['sightings'].append({
                'camera_id': camera_id,
                'bbox': bbox,
                'timestamp': datetime.now()
            })

            # Cap in-memory data to prevent unbounded growth
            MAX_EMBEDDINGS, MAX_SIGHTINGS = 100, 50
            if len(self.persons[person_id]['embeddings']) > MAX_EMBEDDINGS:
                self.persons[person_id]['embeddings'] = self.persons[person_id]['embeddings'][-MAX_EMBEDDINGS:]
            if len(self.persons[person_id]['sightings']) > MAX_SIGHTINGS:
                self.persons[person_id]['sightings'] = self.persons[person_id]['sightings'][-MAX_SIGHTINGS:]

            # Auto-register unknown persons or refine existing registration
            just_registered = False
            emb_count = len(self.persons[person_id]['embeddings'])

            if person_id not in self.registered_to_db:
                # Initial registration: need min_samples embeddings
                if (not self.persons[person_id].get('known', False) and
                    emb_count >= self.min_samples):
                    avg_embedding = np.mean(self.persons[person_id]['embeddings'], axis=0)
                    try:
                        db.add_face(avg_embedding, person_id)
                    except Exception as e:
                        logger.error(f"FAISS add failed for {person_id}, will retry later: {e}")
                    self.registered_to_db.add(person_id)
                    self.last_registered_count[person_id] = emb_count
                    just_registered = True
                    logger.info(f"Auto-registered person {person_id} ({emb_count} samples)")
            else:
                # Refine: re-add improved average every refine_interval new samples
                last_count = self.last_registered_count.get(person_id, self.min_samples)
                if emb_count >= last_count + self.refine_interval:
                    avg_embedding = np.mean(self.persons[person_id]['embeddings'], axis=0)
                    try:
                        db.add_face(avg_embedding, person_id)
                    except Exception as e:
                        logger.error(f"FAISS refine failed for {person_id}: {e}")
                    self.last_registered_count[person_id] = emb_count
                    logger.info(f"Refined person {person_id} embedding ({emb_count} samples)")

            return person_id, just_registered

    def _compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity"""
        emb1_norm = emb1 / np.linalg.norm(emb1)
        emb2_norm = emb2 / np.linalg.norm(emb2)
        return float(np.dot(emb1_norm, emb2_norm))

    def check_for_merge_candidates(self, embedding: np.ndarray, camera_id,
                                   current_time: datetime) -> Optional[str]:
        """Check if embedding matches a recently disappeared person"""
        cutoff_time = current_time.timestamp() - self.merge_time_window
        self.recently_disappeared = {
            pid: data for pid, data in self.recently_disappeared.items()
            if data['last_seen'].timestamp() > cutoff_time
        }

        best_match_id = None
        best_similarity = 0.0

        for person_id, data in self.recently_disappeared.items():
            if len(data['embeddings']) > 0:
                avg_embedding = np.mean(data['embeddings'], axis=0)
                sim = self._compute_similarity(embedding, avg_embedding)

                time_diff = current_time.timestamp() - data['last_seen'].timestamp()
                camera_match = camera_id == data['last_camera']

                threshold = self.auto_merge_threshold if camera_match else self.auto_merge_threshold + 0.05

                if sim > best_similarity and sim >= threshold and time_diff <= self.merge_time_window:
                    best_similarity = sim
                    best_match_id = person_id

        return best_match_id

    def get_person_stats(self) -> Dict:
        """Get statistics about tracked persons"""
        with self.lock:
            known_count = sum(1 for p in self.persons.values() if p.get('known', False))
            unknown_count = len(self.persons) - known_count
            return {
                'total': len(self.persons),
                'known': known_count,
                'unknown': unknown_count,
                'recently_disappeared': len(self.recently_disappeared)
            }

    def mark_disappeared(self, person_id: str, camera_id: str):
        """Mark a person as recently disappeared for merge candidate matching"""
        with self.lock:
            if person_id not in self.persons:
                return
            data = self.persons[person_id]
            embeddings = data['embeddings'][-10:]
            if not embeddings:
                return
            last_seen = datetime.now()
            if data['sightings']:
                last_sighting = data['sightings'][-1]
                if isinstance(last_sighting, dict) and 'timestamp' in last_sighting:
                    last_seen = last_sighting['timestamp']
            self.recently_disappeared[person_id] = {
                'embeddings': list(embeddings),
                'last_seen': last_seen,
                'last_camera': camera_id,
                'timestamp': datetime.now()
            }
            logger.debug(f"Person {person_id} marked as disappeared on camera {camera_id}")

    def cleanup_stale_persons(self, max_age_seconds: float = 300.0):
        """Remove persons not seen in max_age_seconds (skip FAISS-registered)"""
        with self.lock:
            cutoff = datetime.now() - timedelta(seconds=max_age_seconds)
            to_remove = []
            for pid, data in self.persons.items():
                if pid in self.registered_to_db or data.get('known', False):
                    continue
                last_seen = data.get('created_at', datetime.now())
                if data['sightings']:
                    last_sighting = data['sightings'][-1]
                    if isinstance(last_sighting, dict) and 'timestamp' in last_sighting:
                        last_seen = last_sighting['timestamp']
                if last_seen < cutoff:
                    to_remove.append(pid)
            for pid in to_remove:
                del self.persons[pid]
            if to_remove:
                logger.info(f"Cleaned up {len(to_remove)} stale persons from memory")


# ============================================================================
# CAMERA SOURCE (for local cameras)
# ============================================================================

class CameraSource:
    """Represents a single camera source"""
    def __init__(self, camera_id: int, name: str = None, camera_params: dict = None):
        self.camera_id = camera_id
        self.name = name or f"Camera_{camera_id}"
        self.cap = None
        self.is_active = False
        self.frame_queue = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()
        self.camera_params = camera_params or {}

    def open(self) -> bool:
        """Open camera capture"""
        self.cap = cv2.VideoCapture(self.camera_id)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.is_active = True
            return True
        return False

    def close(self):
        """Close camera capture"""
        self.stop_event.set()
        self.is_active = False
        if self.cap:
            self.cap.release()

    def read_frame(self):
        """Read frame from camera"""
        if not self.cap or not self.is_active:
            return None
        ret, frame = self.cap.read()
        return frame if ret else None


# ============================================================================
# DEBUG VISUALIZER
# ============================================================================

class DebugVisualizer:
    """Handles debug visualization with InsightFace features"""

    # Keypoint colors: left_eye, right_eye, nose, left_mouth, right_mouth
    KP_COLORS = [
        (0, 255, 0),   # left eye - green
        (0, 255, 0),   # right eye - green
        (255, 200, 0), # nose - cyan-ish
        (0, 0, 255),   # left mouth - red
        (0, 0, 255),   # right mouth - red
    ]

    def __init__(self):
        self.show_help = True
        self.paused = False
        self.fps_history = deque(maxlen=30)
        self.color_map = {}
        self.next_color_idx = 0
        self.color_palette = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 128, 0)
        ]
        self._face3d_angle = 0.0
        self._face3d_tris = None

    def get_person_color(self, person_id: str) -> tuple:
        """Get consistent color for person"""
        if person_id not in self.color_map:
            self.color_map[person_id] = self.color_palette[
                self.next_color_idx % len(self.color_palette)
            ]
            self.next_color_idx += 1
        return self.color_map[person_id]

    def draw_face_keypoints(self, frame, keypoints):
        """Draw 5 SCRFD facial keypoints with connections"""
        kp = np.asarray(keypoints, dtype=np.int32)
        if kp.shape != (5, 2):
            return

        # Draw keypoint dots
        for i, (x, y) in enumerate(kp):
            cv2.circle(frame, (x, y), 3, self.KP_COLORS[i], -1)
            cv2.circle(frame, (x, y), 4, (255, 255, 255), 1)

        # Connections: eye-to-eye, eye-to-nose, nose-to-mouth
        conn_color = (180, 180, 180)
        cv2.line(frame, tuple(kp[0]), tuple(kp[1]), conn_color, 1)  # eye-eye
        cv2.line(frame, tuple(kp[0]), tuple(kp[2]), conn_color, 1)  # left eye-nose
        cv2.line(frame, tuple(kp[1]), tuple(kp[2]), conn_color, 1)  # right eye-nose
        cv2.line(frame, tuple(kp[2]), tuple(kp[3]), conn_color, 1)  # nose-left mouth
        cv2.line(frame, tuple(kp[2]), tuple(kp[4]), conn_color, 1)  # nose-right mouth
        cv2.line(frame, tuple(kp[3]), tuple(kp[4]), conn_color, 1)  # mouth-mouth

    def draw_head_pose_axes(self, frame, rvec, tvec, camera_matrix, axis_length=40.0):
        """Draw 3D head pose axes (R=X, G=Y, B=Z) projected onto face"""
        try:
            # 3D axis endpoints from origin
            axes_3d = np.array([
                [axis_length, 0, 0],   # X axis (red)
                [0, axis_length, 0],   # Y axis (green)
                [0, 0, axis_length],   # Z axis (blue) - points forward/out of face
            ], dtype=np.float64)

            origin_3d = np.array([[0, 0, 0]], dtype=np.float64)
            dist_coeffs = np.zeros((4, 1), dtype=np.float64)

            # Project origin and axis endpoints
            origin_2d, _ = cv2.projectPoints(origin_3d, rvec, tvec, camera_matrix, dist_coeffs)
            axes_2d, _ = cv2.projectPoints(axes_3d, rvec, tvec, camera_matrix, dist_coeffs)

            origin = tuple(origin_2d[0].ravel().astype(int))
            x_end = tuple(axes_2d[0].ravel().astype(int))
            y_end = tuple(axes_2d[1].ravel().astype(int))
            z_end = tuple(axes_2d[2].ravel().astype(int))

            # Draw axes: X=red, Y=green, Z=blue
            cv2.arrowedLine(frame, origin, x_end, (0, 0, 255), 2, tipLength=0.3)
            cv2.arrowedLine(frame, origin, y_end, (0, 255, 0), 2, tipLength=0.3)
            cv2.arrowedLine(frame, origin, z_end, (255, 0, 0), 2, tipLength=0.3)
        except Exception:
            pass

    def draw_face_info(self, frame, face_bbox, person_id, confidence, pose=None):
        """Draw face bounding box with ID, confidence and pose angles"""
        x1, y1, x2, y2 = face_bbox
        color = self.get_person_color(person_id)

        # Face bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label background
        label = f"{person_id} {confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Pose angles below bbox
        if pose is not None:
            pose_text = f"Y:{pose.yaw:.0f} P:{pose.pitch:.0f} R:{pose.roll:.0f}"
            cv2.putText(frame, pose_text, (x1, y2 + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    def draw_gaze_arrow(self, frame, bbox_center, gaze_heading, length=35):
        """Draw gaze direction arrow (compass heading)"""
        if gaze_heading is None:
            return

        x, y = int(bbox_center[0]), int(bbox_center[1])

        screen_angle = gaze_heading - 90
        rad = np.radians(screen_angle)

        end_x = int(x + length * np.cos(rad))
        end_y = int(y + length * np.sin(rad))

        cv2.arrowedLine(frame, (x, y), (end_x, end_y), (255, 255, 0), 2, tipLength=0.4)

        cardinal = self._get_cardinal_direction(gaze_heading)
        cv2.putText(frame, f"G:{gaze_heading:.0f} {cardinal}", (x - 40, y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    def _get_cardinal_direction(self, heading: float) -> str:
        """Convert heading to cardinal direction"""
        if heading < 0 or heading >= 360:
            heading = heading % 360

        directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        index = int((heading + 22.5) / 45) % 8
        return directions[index]

    def draw_info_panel(self, frame, stats):
        """Draw information panel"""
        height, width = frame.shape[:2]

        overlay = frame.copy()
        panel_height = 140
        cv2.rectangle(overlay, (0, 0), (width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        y_offset = 25
        line_height = 22

        cv2.putText(frame, "Face ReID Debug", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_offset += line_height + 5

        info_lines = [
            f"Source: {stats.get('source_name', 'Unknown')}",
            f"FPS: {stats.get('fps', 0):.1f}",
            f"Faces: {stats.get('faces', 0)}",
            f"Tracked: {stats.get('total', 0)} (Known: {stats.get('known', 0)}, Unknown: {stats.get('unknown', 0)})",
        ]

        for line in info_lines:
            cv2.putText(frame, line, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += line_height

        if self.show_help:
            self.draw_help_overlay(frame)

    def draw_help_overlay(self, frame):
        """Draw hotkey help overlay"""
        height, width = frame.shape[:2]

        help_text = [
            "=== HOTKEYS ===",
            "q: Quit",
            "s: Save screenshot",
            "r: Reset statistics",
            "SPACE: Pause/Resume",
            "h: Toggle this help",
            "1-9: Toggle cameras"
        ]

        overlay = frame.copy()
        box_width = 260
        box_height = len(help_text) * 25 + 20
        box_x = width - box_width - 20
        box_y = 20
        cv2.rectangle(overlay, (box_x, box_y),
                     (box_x + box_width, box_y + box_height),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        y = box_y + 20
        for text in help_text:
            cv2.putText(frame, text, (box_x + 10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y += 25

    def _prepare_face3d_cache(self, result_3d):
        """Pre-compute subsampled face indices for fast widget drawing"""
        faces = result_3d['faces']
        n_verts = len(result_3d['vertices'])
        # Subsample: 76K faces -> ~1000 triangles
        step = max(1, len(faces) // 1000)
        sampled = faces[::step]
        # Filter valid indices
        valid = np.all(sampled < n_verts, axis=1)
        self._face3d_tris = sampled[valid]

    def draw_face3d_widget(self, frame, result_3d):
        """Draw a rotating 3D wireframe mesh widget in the bottom-right corner"""
        if result_3d is None:
            return

        vertices = result_3d['vertices']
        h, w = frame.shape[:2]

        # Adaptive widget size: 1/3 of the smaller frame dimension, clamped
        widget_size = max(120, min(250, min(h, w) // 3))

        # Widget position (bottom-right)
        wx = w - widget_size - 10
        wy = h - widget_size - 10
        if wx < 0 or wy < 0:
            return

        # Black background with green border
        frame[wy:wy + widget_size, wx:wx + widget_size] = 0
        cv2.rectangle(frame, (wx, wy), (wx + widget_size - 1, wy + widget_size - 1),
                      (0, 255, 0), 1)

        # Pre-compute subsampled triangles on first call or data change
        if not hasattr(self, '_face3d_tris') or self._face3d_tris is None:
            self._prepare_face3d_cache(result_3d)

        # Center vertices at origin
        centered = vertices - vertices.mean(axis=0)

        # Y-axis rotation matrix (auto-incrementing)
        angle = self._face3d_angle
        self._face3d_angle += 0.02
        ca, sa = np.cos(angle), np.sin(angle)
        rot = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]], dtype=np.float32)

        rotated = centered @ rot.T

        # Orthographic projection: X, Y
        proj_x = rotated[:, 0]
        proj_y = rotated[:, 1]

        # Scale to fit widget with margin
        margin = 20
        effective = widget_size - 2 * margin
        span = max(proj_x.max() - proj_x.min(), proj_y.max() - proj_y.min(), 1e-6)
        scale = effective / span

        # Map to widget pixel coords (vectorized)
        px = ((proj_x - proj_x.min()) * scale + margin + wx).astype(np.int32)
        py = ((proj_y.max() - proj_y) * scale + margin + wy).astype(np.int32)

        # Build triangle polylines fully vectorized (no Python loop)
        tris = self._face3d_tris
        coords = np.stack([px, py], axis=1)  # [N, 2]
        tri_pts = coords[tris]  # [T, 3, 2] -- all triangle vertices at once
        cv2.polylines(frame, tri_pts, isClosed=True, color=(0, 255, 0), thickness=1)

        # Widget label
        cv2.putText(frame, "3D Face", (wx + 5, wy + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    def handle_key(self, key):
        """Handle keyboard input"""
        if key == ord('q'):
            return 'quit'
        elif key == ord('s'):
            return 'screenshot'
        elif key == ord('r'):
            return 'reset_stats'
        elif key == ord(' '):
            self.paused = not self.paused
            logger.info(f"{'Paused' if self.paused else 'Resumed'}")
        elif key == ord('h'):
            self.show_help = not self.show_help

        return None


# ============================================================================
# UNIFIED REID SYSTEM
# ============================================================================

class UnifiedReIDSystem:
    """Unified ReID system supporting both cameras and streams"""

    def __init__(self, config_path: Optional[str] = None, debug: bool = False):
        """Initialize the unified ReID system"""
        # Load configuration
        self.config = self._load_config(config_path)
        self.debug = debug or self.config.getboolean('system', 'debug', fallback=False)

        # Initialize debug visualizer
        self.visualizer = DebugVisualizer() if self.debug else None

        # Initialize database client
        self.db_client = None
        if self.config.getboolean('database', 'enabled', fallback=False):
            api_url = self.config.get('database', 'api_url')
            api_key = self.config.get('database', 'api_key')
            if api_url and api_key:
                self.db_client = TraitsDBClient(api_url, api_key)
                logger.info(f"Database client initialized: {api_url}")

        # Face detection settings
        self.face_det_model = self.config.get('face_detection', 'model', fallback='det_10g.onnx')
        self.face_conf = self.config.getfloat('face_detection', 'confidence', fallback=0.5)
        self.face_iou = self.config.getfloat('face_detection', 'iou_threshold', fallback=0.4)

        # Face recognition settings
        self.face_rec_model = self.config.get('face_recognition', 'recognition_model',
                                              fallback='face-reidentification/weights/w600k_r50.onnx')
        self.similarity_threshold = self.config.getfloat('face_recognition', 'similarity_threshold',
                                                          fallback=0.4)
        self.min_samples = self.config.getint('face_recognition', 'min_samples', fallback=5)

        # Person tracking settings
        self.auto_merge_threshold = self.config.getfloat('person_tracking', 'auto_merge_threshold',
                                                          fallback=0.85)
        self.merge_time_window = self.config.getfloat('person_tracking', 'merge_time_window',
                                                       fallback=30.0)

        # Anti-spoofing settings
        self.enable_antispoof = self.config.getboolean('anti_spoofing', 'enabled', fallback=False)
        self.antispoof_model = self.config.get('anti_spoofing', 'model', fallback='antispoof_minifas.onnx')
        self.antispoof_threshold = self.config.getfloat('anti_spoofing', 'threshold', fallback=0.5)

        # 3D face reconstruction settings
        self.enable_face3d = self.config.getboolean('face_3d', 'enabled', fallback=False)
        self.face3d_model = self.config.get('face_3d', 'model', fallback='3ddfa_v2.onnx')
        self.face3d_device = self.config.get('face_3d', 'device', fallback='cuda')
        self.face3d_format = self.config.get('face_3d', 'output_format', fallback='glb')
        self.face3d_min_samples = self.config.getint('face_3d', 'min_samples', fallback=50)
        self.face3d_yaw_bin_width = self.config.getfloat('face_3d', 'yaw_bin_width', fallback=30.0)
        self.face3d_refine_interval = self.config.getint('face_3d', 'refine_interval', fallback=50)
        self.face3d_max_frames = self.config.getint('face_3d', 'max_frames', fallback=300)
        # Persist after this many accumulated frames per person. 1 = persist every
        # frame, so progress is durable even when the stream is fed as sparse
        # one-shot pushes (analytics dual-call) and the stream session is reaped
        # between frames. Higher values batch DB writes at the risk of losing the
        # un-persisted tail on a crash.
        self.face3d_persist_every = self.config.getint('face_3d', 'persist_every', fallback=1)

        # Performance settings
        self.process_interval = self.config.getfloat('performance', 'process_interval', fallback=0.1)

        # Pinhole distance model settings
        self.focal_length = self.config.getfloat('depth', 'focal_length', fallback=600.0)
        self.face_height_meters = self.config.getfloat('depth', 'face_height_meters', fallback=0.23)

        # Initialize models
        self._initialize_models()

        # Initialize face database
        if self.db_client:
            self.face_db = create_face_database(
                api_url=self.config.get('database', 'api_url'),
                api_key=self.config.get('database', 'api_key'),
                embedding_size=512
            )
            self.face_db.load()
            logger.info("Using remote centralized FAISS database")
        else:
            db_dir = Path(__file__).parent / "face-reidentification" / "database"
            self.face_db = create_face_database(db_path=str(db_dir), embedding_size=512)
            self.face_db.load()
            logger.info("Using local FAISS database")

        # Initialize person tracker
        self.refine_interval = self.config.getint('face_recognition', 'refine_interval', fallback=30)
        self.tracker = PersonTracker(
            min_samples_for_registration=self.min_samples,
            refine_interval=self.refine_interval,
            auto_merge_threshold=self.auto_merge_threshold,
            merge_time_window_seconds=self.merge_time_window
        )

        # Initialize sighting aggregator
        self.sighting_aggregator = SightingAggregator(
            window_seconds=self.config.getfloat('sighting', 'window_seconds', fallback=5.0),
            max_gap_seconds=self.config.getfloat('sighting', 'max_gap_seconds', fallback=2.0),
            min_sightings_to_aggregate=self.config.getint('sighting', 'min_sightings', fallback=2)
        )
        logger.info("Sighting aggregator initialized")

        # Camera transformers
        self.camera_transformers = {}  # source_id -> CoordinateTransformer

        # Stream/camera management
        self.sources = []  # List of camera or stream sources
        self.source_threads = {}
        self.stop_event = threading.Event()

        # Queue for async DB sends (never blocks processing loop)
        self.db_queue = queue.Queue()

        # Disk-backed overflow queue for failed DB sends
        self._pending_dir = Path(__file__).parent / "pending_db"
        self._pending_dir.mkdir(exist_ok=True)
        pending_count = len(list(self._pending_dir.glob("*.pkl")))
        if pending_count > 0:
            logger.info(f"Found {pending_count} pending DB items from previous session")

        # Cached 3D reconstruction result for debug visualization
        self._last_face3d = None
        # Persistent, traceable 3D-frame accumulator (keyed on the DB person id).
        self.face3d_accumulator = Face3DAccumulator(
            min_samples=self.face3d_min_samples,
            refine_interval=self.face3d_refine_interval,
            max_frames=self.face3d_max_frames,
            yaw_bin_width=self.face3d_yaw_bin_width,
            persist_every=self.face3d_persist_every,
        )
        # Cache: tracker external person id -> persistent DB person id.
        self._db_person_id_cache = {}

        # Start background thread for DB sending (auto-restarts on crash)
        self.db_flush_thread = threading.Thread(
            target=self._run_with_restart, args=(self._db_flush_worker,), daemon=True)
        self.db_flush_thread.start()

        print("Unified ReID system initialized")

    def _initialize_models(self):
        """Initialize detection and recognition models"""
        weights_dir = Path(__file__).parent / "face-reidentification" / "weights"
        det_path = weights_dir / self.face_det_model

        if Path(self.face_rec_model).is_absolute():
            rec_path = Path(self.face_rec_model)
        else:
            rec_path = Path(__file__).parent / self.face_rec_model

        print(f"Loading face detection model: {self.face_det_model}")
        self.face_detector = SCRFD(str(det_path), conf_thres=self.face_conf,
                                    iou_thres=self.face_iou)

        print(f"Loading face recognition model: {rec_path.name}")
        self.face_encoder = ArcFace(str(rec_path))

        # Anti-spoofing model
        self.antispoof_checker = None
        if self.enable_antispoof:
            antispoof_path = weights_dir / self.antispoof_model
            if antispoof_path.exists():
                try:
                    print(f"Loading anti-spoofing model: {self.antispoof_model}")
                    self.antispoof_checker = AntiSpoof(
                        str(antispoof_path), threshold=self.antispoof_threshold
                    )
                except Exception as e:
                    logger.error(f"Failed to load anti-spoofing model: {e}")
            else:
                print(f"Anti-spoofing model not found: {antispoof_path}")

        # 3D face reconstruction model - loaded at startup so params can be
        # accumulated from every detected frame before the final reconstruction.
        self.face3d_reconstructor = None
        self._face3d_path = None
        if self.enable_face3d:
            face3d_path = weights_dir / self.face3d_model
            if face3d_path.exists():
                self._face3d_path = str(face3d_path)
                try:
                    print(f"Loading 3D face reconstruction model: {self.face3d_model}")
                    self.face3d_reconstructor = Face3DReconstructor(
                        self._face3d_path, device=self.face3d_device
                    )
                except Exception as e:
                    logger.error(f"Failed to load 3D face model: {e}")
                    self._face3d_path = None
            else:
                print(f"3D face model not found: {face3d_path}")

    def _load_config(self, config_path: Optional[str] = None) -> configparser.ConfigParser:
        """Load configuration from file"""
        config = configparser.RawConfigParser()

        if config_path is None:
            config_path = Path(__file__).parent / "config.ini"

        if Path(config_path).exists():
            config.read(config_path)
            print(f"Loaded configuration from: {config_path}")
        else:
            print(f"Configuration file not found: {config_path}")
            print("Using default settings")

        # Environment variables override file values for environment-specific
        # and secret settings, so the worker can run with no committed config.ini.
        if not config.has_section('database'):
            config.add_section('database')
        env_overrides = {
            ('database', 'enabled'): os.environ.get('REID_DB_ENABLED'),
            ('database', 'api_url'): os.environ.get('REID_DB_API_URL'),
            ('database', 'api_key'): os.environ.get('REID_DB_API_KEY'),
            ('database', 'analytics_url'): os.environ.get('REID_ANALYTICS_URL'),
            ('system', 'log_level'): os.environ.get('REID_LOG_LEVEL'),
        }
        for (section, key), value in env_overrides.items():
            if value is not None and value != '':
                if not config.has_section(section):
                    config.add_section(section)
                config.set(section, key, value)

        return config

    def _run_with_restart(self, target, *args, max_restarts=10):
        """Run target function with auto-restart on crash"""
        restart_count = 0
        while not self.stop_event.is_set() and restart_count < max_restarts:
            try:
                target(*args)
                break  # Normal exit
            except Exception as e:
                restart_count += 1
                logger.error(f"Thread {target.__name__} crashed ({restart_count}/{max_restarts}): {e}",
                             exc_info=True)
                if restart_count < max_restarts and not self.stop_event.is_set():
                    time.sleep(min(restart_count * 2.0, 10.0))
                    logger.info(f"Restarting {target.__name__}...")

    def process_source(self, source, source_id):
        """Process frames from any source (camera or stream)"""
        source_name = getattr(source, 'name', f'Source_{source_id}')
        print(f"Starting processing thread for {source_name}")

        # Get or create camera transformer
        if hasattr(source, 'camera_params') and source.camera_params:
            if source_id not in self.camera_transformers:
                camera_params = CameraParams(
                    latitude=source.camera_params.get('latitude', 0.0),
                    longitude=source.camera_params.get('longitude', 0.0),
                    altitude=source.camera_params.get('altitude', 0.0),
                    pan=source.camera_params.get('pan', 0.0),
                    tilt=source.camera_params.get('tilt', 0.0),
                    roll=source.camera_params.get('roll', 0.0),
                    focal_length=source.camera_params.get('focal_length', 1000.0),
                    camera_height=source.camera_params.get('camera_height', 2.5),
                    camera_id=source_id,
                    camera_name=source_name
                )
                self.camera_transformers[source_id] = CoordinateTransformer(camera_params)

        transformer = self.camera_transformers.get(source_id)
        frame_count = 0
        fps_start_time = time.time()
        fps = 0.0

        # Cached detection results for redrawing on skipped frames
        last_detections = []
        last_face_count = 0
        last_process_time = 0.0
        last_seen_persons = {}  # person_id -> last_seen_time for disappearance tracking

        while not self.stop_event.is_set() and source.is_active:
            frame = source.read_frame()

            if frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1
            current_time = datetime.now()

            # Calculate FPS
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_start_time)
                fps_start_time = time.time()

            # Time-based processing: only run detection after interval since last processing finished
            should_process = (time.time() - last_process_time) >= self.process_interval

            if not should_process:
                # Redraw cached detections on live frame
                if self.debug and self.visualizer:
                    for det in last_detections:
                        self.visualizer.draw_face_keypoints(frame, det['kp'])
                        self.visualizer.draw_face_info(
                            frame, det['face_bbox'], det['person_id'], det['conf'], det['pose']
                        )
                        if det['pose'] is not None:
                            self.visualizer.draw_head_pose_axes(
                                frame, det['pose'].rvec, det['pose'].tvec, det['pose'].camera_matrix
                            )
                        if det['gaze_compass'] is not None:
                            self.visualizer.draw_gaze_arrow(frame, det['bbox_center'], det['gaze_compass'])

                    stats = self.tracker.get_person_stats()
                    stats.update({
                        'source_name': source_name,
                        'fps': fps,
                        'faces': last_face_count,
                    })
                    self.visualizer.draw_info_panel(frame, stats)
                    self.visualizer.draw_face3d_widget(frame, self._last_face3d)
                    cv2.imshow(source_name, frame)
                    key = cv2.waitKey(1) & 0xFF
                    action = self.visualizer.handle_key(key)
                    if action == 'quit':
                        self.stop_event.set()
                        break
                continue

            # Detect faces
            last_detections = []
            try:
                bboxes, kpss = self.face_detector.detect(frame)
            except Exception as e:
                logger.error(f"Face detection error in {source_name}: {e}")
                bboxes, kpss = [], []
            last_face_count = len(bboxes)

            if len(bboxes) > 0:
                for bbox, kp in zip(bboxes, kpss):
                    # Extract face embedding
                    embedding = self.face_encoder.get_embedding(frame, kp)

                    # Estimate head pose before tracking so yaw can be stored per sample
                    pose = estimate_head_pose(kp, frame.shape[1], frame.shape[0])

                    # Anti-spoofing liveness check
                    liveness_score = None
                    liveness_passed = None
                    if self.antispoof_checker:
                        is_live, liveness_score = self.antispoof_checker.check_liveness(frame, bbox, kp)
                        liveness_passed = is_live
                        if not is_live:
                            if self.debug:
                                conf = float(bbox[4])
                                label = f"SPOOFED ({liveness_score:.2f})"
                                x1, y1 = int(bbox[0]), int(bbox[1])
                                cv2.putText(frame, label, (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            continue

                    # Find or create person
                    person_id, just_registered = self.tracker.find_or_create_person(
                        embedding, source_id, bbox, self.face_db,
                        similarity_threshold=self.similarity_threshold
                    )

                    # Persistent, traceable 3D-frame accumulation + best-effort build.
                    # The accumulator keys persistence on the durable DB person id (the
                    # same id update_person_face3d uses), infers 257-d params, stores a
                    # record (embedding + pose + params), and builds on the 50-step
                    # cadence (50/100/.../300, ceiling 300). Best-effort: it never raises
                    # into the frame loop and never blocks on yaw-bin diversity.
                    if self.face3d_reconstructor and self.db_client:
                        db_person_id = self._resolve_db_person_id(person_id)
                        if db_person_id:
                            self.face3d_accumulator.accumulate(
                                db_person_id, frame, bbox, kp,
                                embedding=embedding, pose=pose,
                                quality=float(bbox[4]), camera_id=str(source_id),
                                reconstructor=self.face3d_reconstructor,
                                db_client=self.db_client,
                                on_person_gone=self._evict_db_person_id,
                            )

                    x1, y1, x2, y2, conf = bbox
                    face_bbox = (int(x1), int(y1), int(x2), int(y2))

                    # Pinhole distance estimate from face bbox height
                    face_h_px = max(y2 - y1, 1)
                    distance = (self.focal_length * self.face_height_meters) / face_h_px

                    # Calculate world coordinates
                    location_data = {}
                    gaze_compass = None

                    if transformer:
                        bbox_center = ((x1 + x2) / 2, (y1 + y2) / 2)

                        location = transformer.calculate_person_location(
                            bbox_center, distance, face_h_px
                        )

                        # Convert face yaw to compass heading using camera pan
                        if pose is not None:
                            gaze_compass = (transformer.params.pan + 180 + pose.yaw) % 360

                        location_data = {
                            'latitude': location['latitude'],
                            'longitude': location['longitude'],
                            'altitude': location['altitude'],
                            'distance': distance,
                        }

                    # Extract and encode face snapshot
                    snapshot_path = None
                    try:
                        # Extract face region with some padding
                        pad = 10
                        fx1 = max(0, face_bbox[0] - pad)
                        fy1 = max(0, face_bbox[1] - pad)
                        fx2 = min(frame.shape[1], face_bbox[2] + pad)
                        fy2 = min(frame.shape[0], face_bbox[3] + pad)
                        face_img = frame[fy1:fy2, fx1:fx2]

                        # Encode face image to JPEG
                        _, buffer = cv2.imencode('.jpg', face_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        # Convert to base64 data URL
                        import base64
                        face_base64 = base64.b64encode(buffer).decode('utf-8')
                        snapshot_path = f"data:image/jpeg;base64,{face_base64}"
                    except Exception as e:
                        logger.warning(f"Failed to encode face snapshot: {e}")

                    sighting_metadata = {}
                    if liveness_score is not None:
                        sighting_metadata['liveness_score'] = liveness_score
                        sighting_metadata['liveness_passed'] = liveness_passed

                    sighting = SightingData(
                        person_id=person_id,
                        camera_id=source_id,
                        timestamp=current_time,
                        bbox=face_bbox,
                        confidence=conf,
                        embedding=embedding,
                        snapshot_path=snapshot_path,
                        metadata=sighting_metadata,
                        gaze_degrees=gaze_compass,
                        **location_data
                    )

                    # Aggregate sighting
                    completed_sighting = self.sighting_aggregator.add_sighting(sighting)

                    # Queue for async DB send (non-blocking)
                    if completed_sighting and self.db_client:
                        self.db_queue.put(completed_sighting)

                    # Cache detection for redrawing on skipped frames
                    bbox_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                    last_detections.append({
                        'face_bbox': face_bbox, 'person_id': person_id,
                        'conf': conf, 'pose': pose, 'kp': kp,
                        'gaze_compass': gaze_compass, 'bbox_center': bbox_center,
                    })

                    # Draw debug visualization
                    if self.debug and self.visualizer:
                        self.visualizer.draw_face_keypoints(frame, kp)
                        self.visualizer.draw_face_info(frame, face_bbox, person_id, conf, pose)
                        if pose is not None:
                            self.visualizer.draw_head_pose_axes(
                                frame, pose.rvec, pose.tvec, pose.camera_matrix
                            )
                        if gaze_compass is not None:
                            self.visualizer.draw_gaze_arrow(frame, bbox_center, gaze_compass)

            # Mark processing complete -- timer restarts from here
            last_process_time = time.time()

            # Track person disappearance for merge candidate matching
            current_seen = {det['person_id'] for det in last_detections}
            now = time.time()
            for pid in current_seen:
                last_seen_persons[pid] = now
            for pid in list(last_seen_persons):
                if pid not in current_seen and (now - last_seen_persons[pid]) > self.tracker.disappear_timeout:
                    self.tracker.mark_disappeared(pid, source_id)
                    del last_seen_persons[pid]

            # Display frame in debug mode
            if self.debug and self.visualizer:
                stats = self.tracker.get_person_stats()
                stats.update({
                    'source_name': source_name,
                    'fps': fps,
                    'faces': len(bboxes),
                })

                self.visualizer.draw_info_panel(frame, stats)
                self.visualizer.draw_face3d_widget(frame, self._last_face3d)

                cv2.imshow(source_name, frame)

                # Handle key press
                key = cv2.waitKey(1) & 0xFF
                action = self.visualizer.handle_key(key)

                if action == 'quit':
                    self.stop_event.set()
                    break
                elif action == 'screenshot':
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"screenshot_{source_name}_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    logger.info(f"Screenshot saved: {filename}")
                elif action == 'reset_stats':
                    frame_count = 0
                    fps_start_time = time.time()
                    logger.info("Statistics reset")

        print(f"Stopped processing {source_name}")

    def _send_sighting_to_db(self, sighting):
        """Send aggregated sighting to database"""
        if not self.db_client:
            return

        # Throttle: send at most one sighting per person every 3 minutes
        SIGHTING_INTERVAL = timedelta(minutes=3)
        last_sent = self.tracker.last_sighting_sent_at.get(sighting.person_id)
        if last_sent and (datetime.now() - last_sent) < SIGHTING_INTERVAL:
            return
        self.tracker.last_sighting_sent_at[sighting.person_id] = datetime.now()

        try:
            # Find or create person in database
            person_data = self.db_client.find_or_create_person(
                external_id=sighting.person_id,
                metadata={'first_seen': sighting.first_seen.isoformat()}
            )
            db_person_id = person_data['id']

            # Create face trait
            face_trait_id = None
            if sighting.best_embedding is not None:
                trait_data = self.db_client.create_trait(
                    person_id=db_person_id,
                    trait_type='face',
                    embedding=sighting.best_embedding,
                    model_name='ArcFace',
                    model_version='w600k_r50',
                    quality_score=float(sighting.best_confidence),
                    source_camera_id=sighting.camera_id,
                    source_timestamp=sighting.first_seen
                )
                face_trait_id = trait_data['id']

            snapshot_to_send = sighting.best_snapshot_path or None

            # Create sighting record
            self.db_client.create_sighting(
                person_id=db_person_id,
                camera_id=sighting.camera_id,
                detected_at=sighting.first_seen,
                confidence_score=float(sighting.best_confidence),
                bounding_box={
                    'x1': float(sighting.best_bbox[0]) if sighting.best_bbox else 0,
                    'y1': float(sighting.best_bbox[1]) if sighting.best_bbox else 0,
                    'x2': float(sighting.best_bbox[2]) if sighting.best_bbox else 0,
                    'y2': float(sighting.best_bbox[3]) if sighting.best_bbox else 0
                },
                duration_seconds=int((sighting.last_seen - sighting.first_seen).total_seconds()),
                face_trait_id=face_trait_id,
                snapshot_path=snapshot_to_send,
                metadata=sighting.metadata,
                person_latitude=float(sighting.avg_latitude) if sighting.avg_latitude is not None else None,
                person_longitude=float(sighting.avg_longitude) if sighting.avg_longitude is not None else None,
                person_altitude=float(sighting.avg_altitude) if sighting.avg_altitude is not None else None,
                distance_meters=float(sighting.avg_distance) if sighting.avg_distance is not None else None,
                heading_degrees=float(sighting.heading) if sighting.heading is not None else None,
                gaze_degrees=float(sighting.gaze_degrees) if sighting.gaze_degrees is not None else None,
                speed_ms=float(sighting.speed_ms) if sighting.speed_ms is not None else None
            )

            logger.info(f"Sent sighting to DB: Person {sighting.person_id}")

        except Exception as e:
            detail = ''
            if hasattr(e, 'response') and e.response is not None:
                try:
                    detail = f" -- {e.response.text}"
                except Exception:
                    pass
            logger.error(f"Failed to send sighting to database: {e}{detail}")
            raise

    def _evict_db_person_id(self, db_id: str) -> None:
        """Drop every tracker-id -> db-id cache entry pointing at a gone DB id.

        Called when the accumulator reports a 404 (person merged away or deleted).
        More than one tracker id can map to the same db id, so we filter by VALUE.
        The next frame re-resolves the tracker id to the live merge-target id."""
        self._db_person_id_cache = {
            k: v for k, v in self._db_person_id_cache.items() if v != db_id}

    def _resolve_db_person_id(self, tracker_person_id: str):
        """Map a PersonTracker id to the persistent DB person id.

        The tracker id can be EITHER an external_id (normal stream-only person,
        an 8-char tag) OR an already-durable DB id (when the face was first seen
        via /api/identify, which seeds FAISS so a later stream hit returns the DB
        id as the FAISS name). So we must not blindly treat it as an external_id:
        that would find_or_create a brand-new bogus person whose external_id is
        another person's DB id, and the 3D accumulator would never stabilize.

        Resolution order: if the id is already a DB person (get_person resolves),
        use it as-is; otherwise treat it as an external_id and find-or-create.
        Cached so the accumulator keys on the durable id without a per-frame DB
        round-trip. Returns None on failure (best-effort; frame not accumulated)."""
        cached = self._db_person_id_cache.get(tracker_person_id)
        if cached:
            return cached
        # Already a durable DB id? (identify-seeded faces resolve here.)
        try:
            self.db_client.get_person(tracker_person_id)
            self._db_person_id_cache[tracker_person_id] = tracker_person_id
            return tracker_person_id
        except Exception:
            pass
        # Otherwise it is an external_id (stream-only person).
        try:
            person_data = self.db_client.find_or_create_person(
                external_id=tracker_person_id)
            db_id = person_data['id']
            self._db_person_id_cache[tracker_person_id] = db_id
            return db_id
        except Exception as e:
            logger.warning(
                f"Could not resolve DB person id for {tracker_person_id}: {e}")
            return None

    def _persist_to_disk(self, item):
        """Save a failed queue item to disk for later retry"""
        try:
            filename = f"{time.time_ns()}.pkl"
            filepath = self._pending_dir / filename
            with open(filepath, 'wb') as f:
                pickle.dump(item, f)
        except Exception as e:
            logger.error(f"Failed to persist queue item to disk: {e}")

    def _drain_disk_queue(self):
        """Try to send disk-persisted items. Returns True if all succeeded."""
        pending_files = sorted(self._pending_dir.glob("*.pkl"))
        if not pending_files:
            return True

        all_ok = True
        for filepath in pending_files:
            try:
                with open(filepath, 'rb') as f:
                    item = pickle.load(f)
                self._dispatch_item(item)
                filepath.unlink()
            except Exception:
                all_ok = False
                break  # DB still down, stop trying
        return all_ok

    def _dispatch_item(self, item):
        """Send a single queue item to DB. Raises on failure."""
        self._send_sighting_to_db(item)

    def _db_flush_worker(self):
        """Background worker that sends all queued items to DB.
        On failure, persists items to disk and retries periodically."""
        last_disk_drain = 0.0
        last_cleanup = 0.0

        while not self.stop_event.is_set():
            db_available = True

            # Drain in-memory queue
            while not self.db_queue.empty():
                try:
                    item = self.db_queue.get_nowait()
                except queue.Empty:
                    break

                if not db_available:
                    self._persist_to_disk(item)
                    continue

                try:
                    self._dispatch_item(item)
                except Exception as e:
                    logger.error(f"DB send failed, persisting to disk: {e}")
                    self._persist_to_disk(item)
                    db_available = False

            # Flush expired aggregation windows
            expired_sightings = self.sighting_aggregator.flush_expired_windows()
            for sighting in expired_sightings:
                if not db_available:
                    self._persist_to_disk(sighting)
                    continue
                try:
                    self._send_sighting_to_db(sighting)
                except Exception as e:
                    logger.error(f"DB send failed, persisting to disk: {e}")
                    self._persist_to_disk(sighting)
                    db_available = False

            # Periodically try to drain disk queue (every 10s)
            now = time.time()
            if db_available and now - last_disk_drain > 10.0:
                last_disk_drain = now
                pending = list(self._pending_dir.glob("*.pkl"))
                if pending:
                    logger.info(f"Retrying {len(pending)} pending DB items from disk...")
                    if self._drain_disk_queue():
                        logger.info("All pending DB items sent successfully")

            # Periodically clean up stale persons from memory (every 60s)
            if now - last_cleanup > 60.0:
                last_cleanup = now
                self.tracker.cleanup_stale_persons()

            time.sleep(0.5)

    def detect_cameras(self) -> List[int]:
        """Detect available cameras by checking /dev/video* devices"""
        print("Detecting available cameras...")
        available_cameras = []

        # Only probe indices that have a /dev/videoN device to avoid OpenCV error spam
        candidate_indices = []
        for i in range(10):
            if Path(f'/dev/video{i}').exists():
                candidate_indices.append(i)

        if not candidate_indices:
            # Fallback: try index 0 only
            candidate_indices = [0]

        for i in candidate_indices:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()
                print(f"  Found camera {i}")

        return available_cameras

    def run_cameras(self, camera_ids: List[int] = None):
        """Run system with local cameras"""
        if camera_ids is None:
            camera_ids = self.detect_cameras()

        if not camera_ids:
            print("No cameras detected!")
            return

        print(f"\nStarting system with {len(camera_ids)} camera(s)...")

        # Initialize cameras
        for cam_id in camera_ids:
            # Hardcode camera parameters for debug mode
            camera_params = None
            if self.debug:
                camera_params = {
                    'latitude': 55.759030,
                    'longitude': 38.087375,
                    'altitude': 0.0,
                    'pan': 180.0,  # Looking straight south
                    'tilt': 0.0,
                    'roll': 0.0,
                    'focal_length': self.focal_length,
                    'camera_height': 2.5
                }
                print(f"Debug mode: Camera {cam_id} at GPS: 55.759030, 38.087375, heading: South (180deg)")

            camera = CameraSource(cam_id, camera_params=camera_params)
            if camera.open():
                self.sources.append(camera)
                print(f"Opened {camera.name}")
            else:
                print(f"Failed to open camera {cam_id}")

        if not self.sources:
            print("No cameras could be opened!")
            return

        # Start processing threads (auto-restart on crash)
        for camera in self.sources:
            thread = threading.Thread(target=self._run_with_restart,
                                     args=(self.process_source, camera, camera.camera_id))
            thread.daemon = True
            thread.start()
            self.source_threads[camera.camera_id] = thread

        print("\nSystem running. Press 'q' in any window to quit.\n")

        # Monitor threads and restart dead ones
        try:
            while not self.stop_event.is_set():
                time.sleep(1.0)
                for cam_id, thread in list(self.source_threads.items()):
                    if not thread.is_alive() and not self.stop_event.is_set():
                        camera = next((c for c in self.sources
                                       if getattr(c, 'camera_id', None) == cam_id), None)
                        if camera and camera.is_active:
                            logger.warning(f"Thread for camera {cam_id} died, restarting...")
                            new_thread = threading.Thread(
                                target=self._run_with_restart,
                                args=(self.process_source, camera, cam_id))
                            new_thread.daemon = True
                            new_thread.start()
                            self.source_threads[cam_id] = new_thread
        except KeyboardInterrupt:
            print("\nShutting down...")

        self.shutdown()

        for camera in self.sources:
            camera.close()

        for thread in self.source_threads.values():
            thread.join(timeout=2.0)

        if self.debug:
            cv2.destroyAllWindows()

        # Save face database
        try:
            self.face_db.save()
            print("Face database saved successfully")
        except Exception as e:
            print(f"Warning: Failed to save face database: {e}")

        # Print statistics
        stats = self.tracker.get_person_stats()
        print("\n" + "=" * 50)
        print("Session Statistics:")
        print(f"  Total persons tracked: {stats['total']}")
        print(f"  Known persons: {stats['known']}")
        print(f"  Unknown persons: {stats['unknown']}")
        print("=" * 50)

    def add_stream_source(self, stream_source, stream_id: str):
        """Add a stream source (for stream server mode)"""
        self.sources.append(stream_source)

        thread = threading.Thread(target=self._run_with_restart,
                                 args=(self.process_source, stream_source, stream_id))
        thread.daemon = True
        thread.start()
        self.source_threads[stream_id] = thread

        logger.info(f"Added and started processing stream: {stream_source.name}")

    def remove_stream_source(self, stream_id: str):
        """Remove a stream source"""
        for i, source in enumerate(self.sources):
            if hasattr(source, 'stream_id') and source.stream_id == stream_id:
                source.close()
                del self.sources[i]
                if stream_id in self.source_threads:
                    del self.source_threads[stream_id]
                logger.info(f"Removed stream: {stream_id}")
                return True
        return False

    @staticmethod
    def _decode_trait_embedding(trait: dict) -> 'np.ndarray | None':
        """Decode embedding from a trait API response.
        Handles both MongoDB Buffer format and base64 string."""
        emb = trait.get('embedding')
        if not emb:
            return None
        try:
            if isinstance(emb, dict) and 'data' in emb:
                # MongoDB Buffer: { type: 'Buffer', data: [byte, byte, ...] }
                return np.array(emb['data'], dtype=np.uint8).view(np.float32).copy()
            elif isinstance(emb, str):
                return np.frombuffer(base64.b64decode(emb), dtype=np.float32).copy()
            else:
                return None
        except Exception:
            return None

    def run_batch_similarity_analysis(self):
        """Analyze all existing persons for similarities and merge duplicates.
        Progress tracked via reid-db-handler API. Triggered manually."""
        if not self.db_client:
            return {'error': 'No DB client configured'}

        logger.info("Starting batch similarity analysis...")

        try:
            result = self.db_client.list_persons(limit=10000)
            persons = result.get('persons', [])
        except Exception as e:
            logger.error(f"Failed to list persons: {e}")
            return {'error': str(e)}

        # Build map of ext_id -> db_id
        person_map = {}
        for p in persons:
            ext_id = p.get('external_id', '')
            if ext_id:
                person_map[ext_id] = p['id']

        if len(person_map) < 2:
            logger.info(f"Only {len(person_map)} person(s), nothing to compare")
            return {'merges': 0, 'processed': 0, 'total': len(person_map)}

        # Check progress from reid-db-handler to find already-processed persons
        already_complete = set()
        try:
            progress_data = self.db_client.get_similarity_progress()
            for entry in progress_data.get('progress', []):
                if entry.get('complete', False):
                    already_complete.add(entry['person_id'])
        except Exception:
            pass  # No progress data yet, start fresh

        logger.info(f"Analyzing {len(person_map)} persons "
                    f"({len(already_complete)} already complete)")

        merge_count = 0
        processed_count = 0
        merged_ids = set()
        ext_ids = list(person_map.keys())

        for i, ext_a in enumerate(ext_ids):
            if ext_a in merged_ids or ext_a in already_complete:
                processed_count += 1
                continue

            db_a = person_map[ext_a]

            # Get this person's face embedding
            try:
                traits_a = self.db_client.list_traits(
                    person_id=db_a, trait_type='face', limit=5)
                trait_list_a = traits_a.get('traits', [])
                if not trait_list_a:
                    processed_count += 1
                    continue

                best_a = max(trait_list_a, key=lambda t: t.get('quality_score', 0))
                emb_a = self._decode_trait_embedding(best_a)
                if emb_a is None:
                    processed_count += 1
                    continue

                emb_a_norm = emb_a / np.linalg.norm(emb_a)
            except Exception as e:
                logger.warning(f"Failed to get embedding for {ext_a}: {e}")
                processed_count += 1
                continue

            compared_count = 0
            person_merged_away = False

            for ext_b in ext_ids[i + 1:]:
                if ext_b in merged_ids:
                    continue

                db_b = person_map[ext_b]

                try:
                    traits_b = self.db_client.list_traits(
                        person_id=db_b, trait_type='face', limit=5)
                    trait_list_b = traits_b.get('traits', [])
                    if not trait_list_b:
                        compared_count += 1
                        continue

                    best_b = max(trait_list_b, key=lambda t: t.get('quality_score', 0))
                    emb_b = self._decode_trait_embedding(best_b)
                    if emb_b is None:
                        compared_count += 1
                        continue

                    emb_b_norm = emb_b / np.linalg.norm(emb_b)
                except Exception as e:
                    logger.warning(f"Failed to get embedding for {ext_b}: {e}")
                    compared_count += 1
                    continue

                sim = float(np.dot(emb_a_norm, emb_b_norm))
                compared_count += 1

                # Store notable similarities as PersonRelationship
                if sim >= 0.2:
                    try:
                        self.db_client.upsert_similarity_relationship(
                            db_a, db_b, sim)
                    except Exception as e:
                        logger.warning(f"Failed to store similarity relationship: {e}")

                if sim >= self.similarity_threshold:
                    try:
                        stats_a = self.db_client.get_person_statistics(db_a)
                        stats_b = self.db_client.get_person_statistics(db_b)
                        sightings_a = stats_a.get('total_sightings', 0)
                        sightings_b = stats_b.get('total_sightings', 0)

                        if sightings_a >= sightings_b:
                            source_db, source_ext = db_b, ext_b
                            target_db, target_ext = db_a, ext_a
                        else:
                            source_db, source_ext = db_a, ext_a
                            target_db, target_ext = db_b, ext_b

                        merge_result = self.db_client.merge_persons(source_db, target_db)
                        merged_ids.add(source_ext)
                        merge_count += 1

                        target_emb = emb_a if target_ext == ext_a else emb_b
                        try:
                            self.face_db.add_face(target_emb, target_ext)
                        except Exception:
                            pass

                        logger.info(
                            f"Merged {source_ext} into {target_ext} "
                            f"(sim={sim:.3f}, traits={merge_result.get('traits_moved', '?')}, "
                            f"sightings={merge_result.get('sightings_moved', '?')})")

                        if source_ext == ext_a:
                            person_merged_away = True
                            break
                    except Exception as e:
                        logger.error(f"Merge failed {ext_a}<->{ext_b}: {e}")

            # Report progress to reid-db-handler
            if not person_merged_away:
                try:
                    self.db_client.update_similarity_progress(ext_a, compared_count)
                except Exception:
                    pass
            else:
                try:
                    self.db_client.reset_similarity_progress(ext_a)
                except Exception:
                    pass

            processed_count += 1
            if processed_count % 10 == 0:
                logger.info(f"Similarity progress: {processed_count}/{len(ext_ids)} "
                            f"persons, {merge_count} merges so far")

        if merge_count > 0:
            try:
                self.face_db.save()
            except Exception:
                pass

        result = {'merges': merge_count, 'processed': processed_count,
                  'total': len(ext_ids)}
        logger.info(f"Batch similarity analysis complete: {result}")
        return result

    def shutdown(self):
        """Shutdown the system"""
        print("Shutting down system...")
        self.stop_event.set()

        # Wait for threads
        for thread in self.source_threads.values():
            thread.join(timeout=2.0)

        # Drain in-memory queue -- persist any that fail
        while not self.db_queue.empty():
            try:
                item = self.db_queue.get_nowait()
            except queue.Empty:
                break
            try:
                self._dispatch_item(item)
            except Exception:
                self._persist_to_disk(item)

        # Flush remaining aggregation windows
        remaining_sightings = self.sighting_aggregator.flush_expired_windows(
            current_time=datetime.now() + timedelta(days=1)
        )
        for sighting in remaining_sightings:
            if self.db_client:
                try:
                    self._send_sighting_to_db(sighting)
                except Exception:
                    self._persist_to_disk(sighting)

        # Final attempt to drain disk queue
        self._drain_disk_queue()
        pending = list(self._pending_dir.glob("*.pkl"))
        if pending:
            print(f"WARNING: {len(pending)} DB items still pending on disk (will retry next run)")

        # Save face database
        try:
            self.face_db.save()
            print("Face database saved")
        except Exception as e:
            print(f"Error saving face database: {e}")


# ============================================================================
# FLASK API (for stream server mode)
# ============================================================================

# Global variables for stream server mode
app = Flask(__name__)
CORS(app)
stream_manager = StreamManager()
reid_system = None


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat()})


@app.route('/api/streams', methods=['GET'])
def list_streams():
    """List all active streams"""
    stats = stream_manager.get_stats()
    return jsonify(stats)


@app.route('/api/streams/<stream_id>', methods=['POST'])
def create_stream(stream_id: str):
    """Create a new stream with camera parameters"""
    data = request.get_json() or {}
    name = data.get('name', None)
    camera_params = data.get('camera_params', {})

    stream = stream_manager.create_stream(stream_id, name, camera_params)

    if reid_system:
        reid_system.add_stream_source(stream, stream_id)

    return jsonify({
        'status': 'created',
        'stream_id': stream_id,
        'name': stream.name,
        'camera_params': stream.camera_params
    }), 201


@app.route('/api/streams/<stream_id>', methods=['DELETE'])
def delete_stream(stream_id: str):
    """Delete a stream"""
    if reid_system:
        reid_system.remove_stream_source(stream_id)

    success = stream_manager.remove_stream(stream_id)

    if success:
        return jsonify({'status': 'deleted', 'stream_id': stream_id})
    else:
        return jsonify({'error': 'Stream not found'}), 404


@app.route('/api/streams/<stream_id>/frame', methods=['POST'])
def push_frame(stream_id: str):
    """Push a new frame to a stream"""
    # Get or create stream
    stream = stream_manager.get_stream(stream_id)
    if not stream:
        camera_params = None
        if request.headers.get('X-Camera-Params'):
            import json
            try:
                camera_params = json.loads(request.headers.get('X-Camera-Params'))
            except:
                pass

        stream = stream_manager.create_stream(stream_id, camera_params=camera_params)
        if reid_system:
            reid_system.add_stream_source(stream, stream_id)

    # Check if IR frame
    is_ir = request.args.get('is_ir', '').lower() == 'true'

    # Get frame data
    frame_data = None
    ir_frame_data = None

    if request.content_type and 'multipart/form-data' in request.content_type:
        if 'frame' in request.files:
            frame_data = request.files['frame'].read()
        if 'ir_frame' in request.files:
            ir_frame_data = request.files['ir_frame'].read()
    elif request.content_type and 'application/json' in request.content_type:
        data = request.get_json()
        if data:
            if 'frame' in data:
                import base64
                frame_data = base64.b64decode(data['frame'])
            if 'ir_frame' in data:
                import base64
                ir_frame_data = base64.b64decode(data['ir_frame'])
            if 'camera_params' in data:
                stream.update_camera_params(data['camera_params'])
    else:
        frame_data = request.data

    if not frame_data and not ir_frame_data:
        return jsonify({'error': 'No frame data provided'}), 400

    # Push frames
    success = True
    if frame_data:
        success = stream.push_frame(frame_data, is_ir=is_ir)
    if ir_frame_data:
        success = success and stream.push_frame(ir_frame_data, is_ir=True)

    if success:
        return jsonify({
            'status': 'ok',
            'stream_id': stream_id,
            'frame_count': stream.frame_count
        })
    else:
        return jsonify({'error': 'Failed to decode frame'}), 400


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    stream_stats = stream_manager.get_stats()
    person_stats = {}
    if reid_system:
        person_stats = reid_system.tracker.get_person_stats()

    return jsonify({
        'streams': stream_stats,
        'persons': person_stats,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/analyze-similarities', methods=['POST'])
def analyze_similarities():
    """Trigger batch similarity analysis of all existing persons.
    Compares face embeddings pairwise, merges duplicates.
    Progress tracked in MongoDB so the process is resumable."""
    global reid_system
    if reid_system is None:
        return jsonify({'error': 'ReID system not initialized'}), 503

    # Run in background thread
    def run_analysis():
        try:
            reid_system.run_batch_similarity_analysis()
        except Exception as e:
            logger.error(f"Batch similarity analysis failed: {e}", exc_info=True)

    thread = threading.Thread(target=run_analysis, daemon=True)
    thread.start()

    return jsonify({
        'status': 'started',
        'message': 'Batch similarity analysis running in background. Check logs for progress.'
    })


@app.route('/api/identify', methods=['POST'])
def identify_person():
    """Identify a person from a single photo. Returns person data with OSINT info."""
    global reid_system
    if reid_system is None:
        return jsonify({'error': 'ReID system not initialized'}), 503

    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'Missing "image" field (base64)'}), 400

    # Decode image
    image_b64 = data['image']
    if ',' in image_b64:
        image_b64 = image_b64.split(',', 1)[1]
    image_bytes = base64.b64decode(image_b64)
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({'error': 'Failed to decode image'}), 400

    # Detect faces
    bboxes, kpss = reid_system.face_detector.detect(frame)
    if len(bboxes) == 0:
        return jsonify({'error': 'No face detected in image'}), 404

    # Select best face (largest bbox area)
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in bboxes]
    best_idx = int(np.argmax(areas))
    bbox = bboxes[best_idx]
    kps = kpss[best_idx]

    # Extract embedding
    embedding = reid_system.face_encoder.get_embedding(frame, kps)

    # Search FAISS
    matched_name, similarity = reid_system.face_db.search(embedding)
    is_new = not matched_name or matched_name == "Unknown" or similarity < reid_system.similarity_threshold

    # Crop face for snapshot (with padding)
    x1, y1, x2, y2, conf = bbox
    pad = 20
    fx1, fy1 = max(0, int(x1) - pad), max(0, int(y1) - pad)
    fx2, fy2 = min(frame.shape[1], int(x2) + pad), min(frame.shape[0], int(y2) + pad)
    face_img = frame[fy1:fy2, fx1:fx2]
    _, buf = cv2.imencode('.jpg', face_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    face_b64 = base64.b64encode(buf).decode('utf-8')
    snapshot_url = f"data:image/jpeg;base64,{face_b64}"

    db = reid_system.db_client
    if not db:
        return jsonify({'error': 'Database not configured'}), 503

    if is_new:
        external_id = str(uuid.uuid4())[:8]
        person_data = db.find_or_create_person(external_id=external_id)
        db_person_id = person_data['id']
        # Key FAISS by external_id, NOT the DB id. The stream path keys FAISS by
        # external_id too (PersonTracker returns matched_name = the FAISS name),
        # so both ingestion paths must agree. Keying by the DB id here made the
        # stream path resolve a DB id as if it were an external_id and double-
        # create a bogus person every frame, so 3D accumulation never stabilized.
        reid_system.face_db.add_face(embedding, person_data.get('external_id', external_id))
    else:
        person_data = db.find_or_create_person(external_id=matched_name)
        db_person_id = person_data['id']

    # Store trait + sighting
    trait = db.create_trait(
        person_id=db_person_id, trait_type='face', embedding=embedding,
        model_name='ArcFace', model_version='w600k_r50',
        quality_score=float(conf), source_camera_id='identify_api',
        source_timestamp=datetime.now()
    )
    db.create_sighting(
        person_id=db_person_id, camera_id='identify_api',
        detected_at=datetime.now(), confidence_score=float(conf),
        bounding_box={'x1': float(x1), 'y1': float(y1), 'x2': float(x2), 'y2': float(y2)},
        snapshot_path=snapshot_url, face_trait_id=trait['id']
    )

    # Use the person dict returned by find_or_create_person above rather than
    # re-fetching by id. A re-fetch races with the auto-merge that create_trait
    # queues: the just-created person can be merged into a lookalike and its id
    # deleted before we read it back, which 404s and 500s the whole identify.
    person = dict(person_data)
    person.pop('created', None)
    # Strip binary data from response
    person.pop('face_model_3d', None)
    return jsonify({
        'person': person,
        'match': {'is_new': is_new, 'similarity': float(similarity) if not is_new else 0.0},
        'face_bbox': {'x1': float(x1), 'y1': float(y1), 'x2': float(x2), 'y2': float(y2)}
    })


def cleanup_thread():
    """Background thread to cleanup inactive streams"""
    while True:
        time.sleep(30)
        stream_manager.cleanup_inactive_streams(timeout_seconds=30.0)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    global reid_system

    parser = argparse.ArgumentParser(
        description="Person ReID Multi-Camera and Stream Server System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  camera:  Process local cameras (legacy mode)
  server:  HTTP API for remote video streams (default)

Examples:
  # Stream server mode (default)
  python main.py

  # Stream server on custom port
  python main.py --port 8080

  # Camera mode with debug GUI
  python main.py --mode camera --debug

  # Camera mode with specific cameras
  python main.py --mode camera --cameras 0 1 --debug

Hotkeys (Debug Mode):
  q: Quit
  s: Save screenshot
  r: Reset statistics
  SPACE: Pause/Resume
  h: Toggle help overlay
        """
    )

    parser.add_argument('--mode', type=str, default='server',
                       choices=['camera', 'server'],
                       help='Operation mode')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port for stream server mode')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host for stream server mode')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with GUI')
    parser.add_argument('--cameras', type=int, nargs='+',
                       help='Specific camera IDs for camera mode')
    parser.add_argument('--config', type=str,
                       help='Path to config file')

    args = parser.parse_args()

    # Initialize ReID system
    print("Initializing ReID System...")
    reid_system = UnifiedReIDSystem(config_path=args.config, debug=args.debug)

    if args.mode == 'camera':
        # Camera mode
        print("\n=== CAMERA MODE ===")
        reid_system.run_cameras(camera_ids=args.cameras)

    else:
        # Stream server mode (default)
        print("\n=== STREAM SERVER MODE ===")

        # Start cleanup thread
        cleanup = threading.Thread(target=cleanup_thread, daemon=True)
        cleanup.start()

        print(f"\nStarting server on {args.host}:{args.port}")
        print(f"Stream endpoint: http://{args.host}:{args.port}/api/streams/<stream_id>/frame")
        print("\nPress Ctrl+C to stop\n")

        try:
            from waitress import serve
            serve(app, host=args.host, port=args.port)
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            if reid_system:
                reid_system.shutdown()
            stream_manager.close_all()


if __name__ == "__main__":
    main()
