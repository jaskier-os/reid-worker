import cv2
import numpy as np
import onnxruntime

from typing import Tuple


class AntiSpoof:
    """
    Anti-spoofing model using MiniFASNetV2SE (Silent-Face-Anti-Spoofing).
    Detects whether a face is live or a spoof (photo/video/mask).

    Source: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
    """

    def __init__(self, model_path: str, threshold: float = 0.5) -> None:
        """
        Initialize anti-spoofing model.

        Args:
            model_path: Path to ONNX model file.
            threshold: Liveness threshold (0-1). Scores above this are live.
        """
        self.threshold = threshold
        self.input_size = (80, 80)

        self._initialize_model(model_path)

    def _initialize_model(self, model_path):
        """Initialize ONNX Runtime session with CUDA/CPU fallback."""
        available_providers = onnxruntime.get_available_providers()
        providers = []
        if 'CUDAExecutionProvider' in available_providers:
            providers.append('CUDAExecutionProvider')
        if 'CPUExecutionProvider' in available_providers:
            providers.append('CPUExecutionProvider')

        if not providers:
            providers = [available_providers[0]]

        self.session = onnxruntime.InferenceSession(
            model_path,
            providers=providers
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [x.name for x in self.session.get_outputs()]

    def _crop_face(self, frame, bbox):
        """
        Crop face region from frame with padding.

        Args:
            frame: Full frame image (BGR).
            bbox: Face bounding box [x1, y1, x2, y2, conf].

        Returns:
            Cropped face image.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        face_w = x2 - x1
        face_h = y2 - y1
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        scale = 2.7
        new_w = int(face_w * scale)
        new_h = int(face_h * scale)

        nx1 = max(0, cx - new_w // 2)
        ny1 = max(0, cy - new_h // 2)
        nx2 = min(w, cx + new_w // 2)
        ny2 = min(h, cy + new_h // 2)

        return frame[ny1:ny2, nx1:nx2]

    def _preprocess(self, face_crop):
        """
        Preprocess face crop for model input.

        Args:
            face_crop: Cropped face image (BGR).

        Returns:
            Preprocessed blob ready for inference.
        """
        resized = cv2.resize(face_crop, self.input_size)

        blob = cv2.dnn.blobFromImage(
            resized,
            scalefactor=1 / 255.0,
            size=self.input_size,
            mean=(0, 0, 0),
            swapRB=True
        )

        return blob.astype(np.float32)

    def check_liveness(self, frame, bbox, kps=None) -> Tuple[bool, float]:
        """
        Check if a detected face is live or spoofed.

        Args:
            frame: Full frame image (BGR).
            bbox: Face bounding box [x1, y1, x2, y2, conf].
            kps: Face keypoints (unused, kept for interface consistency).

        Returns:
            Tuple of (is_live: bool, liveness_score: float).
            Score is 0-1 where higher = more likely live.
        """
        face_crop = self._crop_face(frame, bbox)

        if face_crop.size == 0:
            return (False, 0.0)

        blob = self._preprocess(face_crop)

        outputs = self.session.run(self.output_names, {self.input_name: blob})

        prediction = outputs[0]

        if prediction.shape[-1] == 2:
            score = float(prediction[0][1])
        else:
            score = float(prediction[0][0])

        is_live = score >= self.threshold
        return (is_live, score)
