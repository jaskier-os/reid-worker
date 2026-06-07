#!/usr/bin/env python3
"""
Depth Estimation Module
Supports RGB and IR feeds for distance calculation
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, List
import onnxruntime as ort
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DepthEstimator:
    """
    Depth estimation from RGB/IR images
    Supports multiple methods: monocular depth, stereo, and IR-based estimation
    """

    def __init__(
        self,
        method: str = "monocular",
        model_path: Optional[str] = None,
        use_gpu: bool = True
    ):
        """
        Initialize depth estimator

        Args:
            method: Estimation method ('monocular', 'stereo', 'ir', 'hybrid')
            model_path: Path to depth estimation model
            use_gpu: Use GPU acceleration if available
        """
        self.method = method
        self.model = None
        self.session = None

        # Initialize based on method
        if method == "monocular":
            self._init_monocular_model(model_path, use_gpu)
        elif method == "stereo":
            self._init_stereo_matcher()
        elif method == "ir":
            self._init_ir_processor()
        elif method == "hybrid":
            self._init_hybrid_model(model_path, use_gpu)

    def _init_monocular_model(self, model_path: Optional[str], use_gpu: bool):
        """Initialize monocular depth estimation model (MiDaS or similar)"""
        if model_path is None:
            # Use lightweight MiDaS variant
            model_path = str(Path(__file__).parent.parent / "models" / "midas_v21_small.onnx")

        if Path(model_path).exists():
            providers = ['CUDAExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
            try:
                self.session = ort.InferenceSession(model_path, providers=providers)
                logger.info(f"Loaded monocular depth model: {model_path}")
                print(f"[DEPTH] Monocular depth model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load depth model, using fallback: {e}")
                print(f"[DEPTH] WARNING: Failed to load depth model, using simple fallback")
                self.session = None
        else:
            logger.warning(f"Depth model not found at {model_path}, using fallback estimation")
            print(f"[DEPTH] WARNING: Depth model not found, using simple fallback")
            print(f"[DEPTH] Expected location: {model_path}")
            print(f"[DEPTH] Depth visualization ('d' hotkey) will show simplified estimation")
            self.session = None

    def _init_stereo_matcher(self):
        """Initialize stereo matching for dual camera setup"""
        # Stereo BM (Block Matching) for depth from stereo pairs
        self.stereo_matcher = cv2.StereoBM_create(numDisparities=96, blockSize=15)

        # Stereo SGBM for better quality (slower)
        self.stereo_sgbm = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=96,
            blockSize=11,
            P1=8 * 3 * 11**2,
            P2=32 * 3 * 11**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32
        )

    def _init_ir_processor(self):
        """Initialize IR image processor for depth from infrared"""
        # IR processing parameters
        self.ir_params = {
            'blur_kernel': 5,
            'threshold_low': 50,
            'threshold_high': 200,
            'distance_scale': 0.001  # Convert IR values to meters
        }

    def _init_hybrid_model(self, model_path: Optional[str], use_gpu: bool):
        """Initialize hybrid RGB+IR depth model"""
        # This would use a custom trained model that takes both RGB and IR as input
        self._init_monocular_model(model_path, use_gpu)
        self._init_ir_processor()

    def estimate_depth(
        self,
        rgb_image: np.ndarray,
        ir_image: Optional[np.ndarray] = None,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        focal_length: float = 1000.0,
        real_height: float = 1.75
    ) -> Dict[str, float]:
        """
        Estimate depth/distance for a person

        Args:
            rgb_image: RGB image
            ir_image: Optional IR image for enhanced depth
            bbox: Person bounding box (x1, y1, x2, y2)
            focal_length: Camera focal length in pixels

        Returns:
            Dictionary with depth information
        """
        results = {}

        # Get depth map based on available inputs and method
        if self.method == "monocular" or (self.method == "hybrid" and ir_image is None):
            depth_map = self._estimate_monocular_depth(rgb_image)
        elif self.method == "ir" and ir_image is not None:
            depth_map = self._estimate_ir_depth(ir_image)
        elif self.method == "hybrid" and ir_image is not None:
            depth_map = self._estimate_hybrid_depth(rgb_image, ir_image)
        else:
            # Fallback to simple estimation
            depth_map = self._estimate_fallback_depth(rgb_image, bbox)

        if depth_map is not None and bbox is not None:
            # Extract depth for the bounding box region
            x1, y1, x2, y2 = bbox
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(depth_map.shape[1], int(x2)), min(depth_map.shape[0], int(y2))

            if x2 > x1 and y2 > y1:
                roi_depth = depth_map[y1:y2, x1:x2]

                # Calculate statistics
                results['mean_depth'] = float(np.mean(roi_depth))
                results['median_depth'] = float(np.median(roi_depth))
                results['min_depth'] = float(np.min(roi_depth))
                results['max_depth'] = float(np.max(roi_depth))

                # Use median for person distance (more robust to outliers)
                results['person_distance'] = results['median_depth']

                # Calculate distance from bbox size as validation
                bbox_height = y2 - y1
                height_based_distance = self._distance_from_bbox_height(
                    bbox_height, focal_length, real_height
                )
                results['height_based_distance'] = height_based_distance

                # Combine estimates if both are available
                if depth_map is not None:
                    # Weight based on confidence
                    depth_weight = 0.7 if self.session is not None else 0.3
                    height_weight = 1.0 - depth_weight
                    results['combined_distance'] = (
                        depth_weight * results['person_distance'] +
                        height_weight * height_based_distance
                    )
                else:
                    results['combined_distance'] = height_based_distance

                # Estimate confidence
                results['confidence'] = self._calculate_confidence(results, bbox)

        elif bbox is not None:
            # Only bbox available, use height-based estimation
            _, y1, _, y2 = bbox
            bbox_height = y2 - y1
            distance = self._distance_from_bbox_height(bbox_height, focal_length, real_height)
            results['person_distance'] = distance
            results['combined_distance'] = distance
            results['height_based_distance'] = distance
            results['confidence'] = 0.5  # Lower confidence without depth map

        results['depth_map'] = depth_map
        return results

    def _estimate_monocular_depth(self, rgb_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate depth from single RGB image using neural network

        Args:
            rgb_image: Input RGB image

        Returns:
            Depth map or None if model not available
        """
        if self.session is None:
            return None

        try:
            # Prepare input
            input_height, input_width = self.session.get_inputs()[0].shape[2:4]
            img_resized = cv2.resize(rgb_image, (input_width, input_height))

            # Normalize and prepare batch
            img_normalized = (img_resized.astype(np.float32) - 127.5) / 127.5
            img_batch = np.transpose(img_normalized, (2, 0, 1))[np.newaxis, ...]

            # Run inference
            output = self.session.run(None, {self.session.get_inputs()[0].name: img_batch})[0]

            # Resize back to original resolution
            depth_map = cv2.resize(output[0, 0], (rgb_image.shape[1], rgb_image.shape[0]))

            # Convert relative depth to absolute (this requires calibration)
            # For now, use a simple scaling
            depth_map = depth_map * 10.0  # Scale to approximate meters

            return depth_map

        except Exception as e:
            logger.error(f"Depth estimation failed: {e}")
            return None

    def _estimate_ir_depth(self, ir_image: np.ndarray) -> np.ndarray:
        """
        Estimate depth from IR image

        Args:
            ir_image: Infrared image

        Returns:
            Depth map in meters
        """
        # IR intensity often correlates with distance
        # Closer objects reflect more IR light

        # Apply preprocessing
        ir_blur = cv2.GaussianBlur(ir_image, (5, 5), 0)

        # Normalize IR values
        ir_normalized = cv2.normalize(ir_blur, None, 0, 255, cv2.NORM_MINMAX)

        # Invert (closer = higher value)
        ir_inverted = 255 - ir_normalized

        # Convert to distance (requires calibration)
        # This is a simplified linear mapping
        max_distance = 20.0  # meters
        min_distance = 0.5   # meters

        depth_map = min_distance + (ir_inverted / 255.0) * (max_distance - min_distance)

        return depth_map

    def _estimate_hybrid_depth(
        self,
        rgb_image: np.ndarray,
        ir_image: np.ndarray
    ) -> np.ndarray:
        """
        Combine RGB and IR for better depth estimation

        Args:
            rgb_image: RGB image
            ir_image: IR image

        Returns:
            Combined depth map
        """
        # Get individual depth maps
        rgb_depth = self._estimate_monocular_depth(rgb_image)
        ir_depth = self._estimate_ir_depth(ir_image)

        if rgb_depth is None:
            return ir_depth

        # Weighted combination
        # IR is generally more reliable at close range
        # RGB provides better structure at distance

        # Calculate weights based on IR confidence
        ir_confidence = self._calculate_ir_confidence(ir_image)

        # Combine with adaptive weighting
        combined_depth = (
            ir_confidence * ir_depth +
            (1 - ir_confidence) * rgb_depth
        )

        return combined_depth

    def _estimate_fallback_depth(
        self,
        rgb_image: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]]
    ) -> Optional[np.ndarray]:
        """
        Fallback depth estimation using simple heuristics

        Args:
            rgb_image: RGB image
            bbox: Person bounding box

        Returns:
            Simple depth map or None
        """
        # Create a full-frame depth map based on image position
        # Assumption: lower in image = closer (perspective)
        height, width = rgb_image.shape[:2]
        depth_map = np.ones((height, width), dtype=np.float32)

        # Linear gradient from top to bottom (20m at top, 1m at bottom)
        y_coords = np.arange(height).reshape(-1, 1)
        depth_map = 20.0 * (1.0 - y_coords / height) + 1.0

        # Add some variance based on edges (objects with edges are "closer")
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges_float = edges.astype(np.float32) / 255.0

        # Reduce depth where edges are detected (simulate pop-out)
        depth_map = depth_map * (1.0 - 0.3 * edges_float)

        # Smooth the depth map
        depth_map = cv2.GaussianBlur(depth_map, (5, 5), 0)

        return depth_map

    def _distance_from_bbox_height(
        self,
        bbox_height: float,
        focal_length: float,
        real_height: float = 1.75
    ) -> float:
        """
        Estimate distance from bounding box height

        Args:
            bbox_height: Height of bounding box in pixels
            focal_length: Focal length in pixels
            real_height: Assumed real height of person in meters

        Returns:
            Estimated distance in meters
        """
        if bbox_height <= 0:
            return 10.0  # Default distance

        # Simple pinhole camera model
        distance = (real_height * focal_length) / bbox_height

        # Clamp to reasonable range
        distance = max(0.5, min(50.0, distance))

        return distance

    def _calculate_ir_confidence(self, ir_image: np.ndarray) -> float:
        """
        Calculate confidence score for IR depth

        Args:
            ir_image: IR image

        Returns:
            Confidence score (0-1)
        """
        # Check IR image quality metrics
        mean_intensity = np.mean(ir_image)
        std_intensity = np.std(ir_image)

        # Good IR images have moderate mean and high contrast
        if mean_intensity < 20 or mean_intensity > 235:
            confidence = 0.3
        elif std_intensity < 10:
            confidence = 0.4
        else:
            confidence = min(1.0, std_intensity / 50.0)

        return confidence

    def _calculate_confidence(
        self,
        depth_results: Dict[str, float],
        bbox: Tuple[int, int, int, int]
    ) -> float:
        """
        Calculate overall confidence score

        Args:
            depth_results: Depth estimation results
            bbox: Bounding box

        Returns:
            Confidence score (0-1)
        """
        confidence = 0.5

        # Check consistency between methods
        if 'person_distance' in depth_results and 'height_based_distance' in depth_results:
            diff_ratio = abs(
                depth_results['person_distance'] - depth_results['height_based_distance']
            ) / depth_results['height_based_distance']

            if diff_ratio < 0.2:
                confidence = 0.9
            elif diff_ratio < 0.5:
                confidence = 0.7
            else:
                confidence = 0.4

        # Adjust based on bbox size
        x1, y1, x2, y2 = bbox
        bbox_area = (x2 - x1) * (y2 - y1)
        if bbox_area > 10000:  # Large bbox
            confidence *= 1.1

        return min(1.0, max(0.0, confidence))


def create_depth_estimator(config: Dict) -> DepthEstimator:
    """
    Factory function to create depth estimator based on config

    Args:
        config: Configuration dictionary

    Returns:
        DepthEstimator instance
    """
    method = config.get('depth_method', 'monocular')
    model_path = config.get('depth_model_path', None)
    use_gpu = config.get('use_gpu', True)

    return DepthEstimator(method=method, model_path=model_path, use_gpu=use_gpu)