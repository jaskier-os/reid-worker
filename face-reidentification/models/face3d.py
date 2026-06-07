import cv2
import io
import numpy as np
import onnxruntime
from pathlib import Path
from typing import Dict, Optional
import logging

__all__ = ["Face3DReconstructor"]

logger = logging.getLogger(__name__)

_RESCALE_FACTOR = 102.0  # Deep3DFaceRecon default face scale in 224x224 crop


def _pos(xp: np.ndarray, x: np.ndarray):
    """
    Least-squares 3D-to-2D projection (from Deep3DFaceRecon_pytorch/util/preprocess.py).

    Solves for scale + translation that best maps 3D reference landmarks to
    detected 2D landmarks (Y-flipped coordinate system).

    Args:
        xp: (2, N) detected 2D landmarks (Y-flipped)
        x:  (3, N) 3D BFM reference landmarks
    Returns:
        t: (2, 1) translation
        s: float scale
    """
    npts = xp.shape[1]
    A = np.zeros([2 * npts, 8])
    A[0:2 * npts - 1:2, 0:3] = x.transpose()
    A[0:2 * npts - 1:2, 3] = 1
    A[1:2 * npts:2, 4:7] = x.transpose()
    A[1:2 * npts:2, 7] = 1
    b = np.reshape(xp.transpose(), [2 * npts, 1])
    k, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    R1 = k[0:3]
    R2 = k[4:7]
    sTx = k[3]
    sTy = k[7]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2
    t = np.stack([sTx, sTy], axis=0)
    return t, s


class Face3DReconstructor:
    """
    3D face reconstruction using Deep3DFaceRecon (Deng et al., 2019).

    Network: ResNet50 backbone + 7 parallel 1x1 conv heads.
    Input:   [1, 3, 224, 224] RGB [0,1]-normalised, aligned via POS projection.
    Output:  [1, 257] coefficients:
               [0:80]    identity shape
               [80:144]  expression
               [144:224] texture (used for albedo vertex coloring)
               [224:227] Euler rotation angles
               [227:254] SH lighting (unused for geometry)
               [254:257] translation

    BFM09 basis loaded from companion .npz (keys: u, w_shp, w_exp, meantex, texBase, tri, lm3d).
    Reconstruction: pts = u + w_shp @ id_coeff + w_exp @ exp_coeff
    Albedo:         tex = meantex + texBase @ tex_coeff
    """

    def __init__(self, model_path: str, device: str = 'cuda') -> None:
        self._load_model(model_path, device)

    def _load_model(self, model_path: str, device: str) -> None:
        available = onnxruntime.get_available_providers()
        providers = []
        if device != 'cpu' and 'CUDAExecutionProvider' in available:
            providers.append('CUDAExecutionProvider')
        if 'CPUExecutionProvider' in available:
            providers.append('CPUExecutionProvider')
        if not providers:
            providers = [available[0]]

        self.session     = onnxruntime.InferenceSession(model_path, providers=providers)
        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        actual = self.session.get_providers()[0]
        logger.info(f'Face3DReconstructor loaded: {model_path} on {actual}')

        # BFM09 basis -- same stem as ONNX, .npz extension
        npz_path = Path(model_path).with_suffix('.npz')
        if npz_path.exists():
            data = np.load(str(npz_path), allow_pickle=False)
            # Recenter mean shape (matches Deep3DFaceRecon ParametricFaceModel)
            raw_u = data['u'].astype(np.float32)
            u_verts = raw_u.reshape(-1, 3)
            u_verts -= u_verts.mean(axis=0, keepdims=True)
            self.bfm_u     = u_verts.flatten()
            self.bfm_w_shp = data['w_shp'].astype(np.float32)
            self.bfm_w_exp = data['w_exp'].astype(np.float32)
            self.tri       = data['tri'].astype(np.int32)
            # Texture basis for albedo vertex coloring
            if 'meantex' in data and 'texBase' in data:
                self.bfm_meantex = data['meantex'].astype(np.float32)
                self.bfm_texBase = data['texBase'].astype(np.float32)
            else:
                self.bfm_meantex = self.bfm_texBase = None
            # 3D reference landmarks for POS alignment
            if 'lm3d' in data:
                self.lm3d = data['lm3d'].astype(np.float32)  # (5, 3)
            else:
                self.lm3d = None
                logger.warning('lm3d not in NPZ; face alignment will use bbox crop fallback')
            logger.info(
                f'BFM basis loaded from {npz_path}: '
                f'u={self.bfm_u.shape}, w_shp={self.bfm_w_shp.shape}, '
                f'w_exp={self.bfm_w_exp.shape}, tri={self.tri.shape}, '
                f'texture={"yes" if self.bfm_meantex is not None else "no"}, '
                f'lm3d={"yes" if self.lm3d is not None else "no"}'
            )
        else:
            self.bfm_u = self.bfm_w_shp = self.bfm_w_exp = self.tri = None
            self.bfm_meantex = self.bfm_texBase = None
            self.lm3d = None
            logger.warning(f'BFM basis not found at {npz_path}; reconstruction will fail')

    # ------------------------------------------------------------------
    # Alignment helpers
    # ------------------------------------------------------------------

    def _align_face(self, frame: np.ndarray, kps: np.ndarray) -> Optional[np.ndarray]:
        """
        POS-based face alignment matching Deep3DFaceRecon_pytorch/util/preprocess.py.

        Uses 3D BFM reference landmarks and least-squares projection to compute
        scale + translation, then produces a 224x224 crop via affine warp.

        Args:
            frame: BGR uint8 image
            kps: (5, 2) SCRFD keypoints in image coordinates (Y-down)
        """
        h0, w0 = frame.shape[:2]
        pts = np.array(kps[:5], dtype=np.float64)

        # Y-flip landmarks to match Deep3DFaceRecon coordinate convention
        pts_flip = pts.copy()
        pts_flip[:, 1] = h0 - 1 - pts_flip[:, 1]

        # POS: find 3D-to-2D projection (scale + translation)
        t, s_pos = _pos(pts_flip.T, self.lm3d.T)
        if s_pos < 1e-6:
            return None
        s = _RESCALE_FACTOR / s_pos

        # Compute affine warp matrix matching resize_n_crop_img behavior.
        # After scaling by s, crop 224x224 centered on the face:
        #   left = t[0]*s - 112
        #   up   = h0*s - t[1]*s - 112
        # Forward mapping: dst(ox,oy) = src at (ox/s + t[0] - 112/s, ...)
        # For cv2.warpAffine, M maps src -> dst:
        #   ox = s * ix - s * t[0,0] + 112
        #   oy = s * iy - s * (h0 - t[1,0]) + 112
        t_flat = t.flatten()
        tx = float(t_flat[0])
        ty = float(t_flat[1])
        M = np.array([
            [s, 0, -s * tx + 112],
            [0, s, -s * (h0 - ty) + 112],
        ], dtype=np.float64)

        aligned = cv2.warpAffine(frame, M, (224, 224),
                                 flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REFLECT_101)
        return aligned

    def _crop_bbox(self, frame: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        """Fallback: square crop centred on bbox, resized to 224x224."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        half = int(max(x2 - x1, y2 - y1) * 0.6)
        nx1 = max(0, cx - half); ny1 = max(0, cy - half)
        nx2 = min(w, cx + half); ny2 = min(h, cy + half)
        crop = frame[ny1:ny2, nx1:nx2]
        if crop.size == 0:
            return None
        return cv2.resize(crop, (224, 224))

    def _preprocess(self, face_img: np.ndarray) -> np.ndarray:
        """BGR uint8 -> [0,1] NCHW float32 RGB (no ImageNet normalization)."""
        rgb = face_img[..., ::-1].astype(np.float32) / 255.0
        return rgb.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def infer_params(self, frame: np.ndarray, bbox: np.ndarray,
                     kps: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Run ONNX inference and return raw 257-dim coefficient vector.

        Layout: [0:80] identity, [80:144] expression, [144:224] texture,
                [224:227] rotation, [227:254] lighting, [254:257] translation.

        kps: optional (5, 2) SCRFD keypoints for accurate alignment.
             Falls back to bbox crop if None or fewer than 5 points.
        Returns None on failure.
        """
        try:
            if kps is not None and len(kps) >= 5 and self.lm3d is not None:
                face_img = self._align_face(frame, kps)
            else:
                face_img = self._crop_bbox(frame, bbox)

            if face_img is None or face_img.size == 0:
                return None

            blob = self._preprocess(face_img)
            out  = self.session.run([self.output_name], {self.input_name: blob})[0]
            return out.flatten().astype(np.float32)
        except Exception as e:
            logger.debug(f'3D param inference failed: {e}')
            return None

    def reconstruct_from_params_list(self, params_list: list) -> Optional[Dict]:
        """
        Build the final 3D mesh by averaging shape/expression parameters across
        all collected frames.

        Identity coefficients (id_coeff) are view-invariant and person-specific.
        Averaging them over multiple diverse-pose frames suppresses single-frame
        noise, yielding the best per-person shape estimate the network can produce.

        Canonical (identity) pose is used so the mesh is always reconstructed
        frontal and upright regardless of camera angle during capture.
        """
        if not params_list:
            return None
        if self.bfm_u is None:
            return None

        params_array = np.stack(params_list)           # (N, 257)
        avg_id  = np.mean(params_array[:,  0:80],  axis=0)  # (80,)
        avg_exp = np.mean(params_array[:, 80:144], axis=0)  # (64,)

        # Clamp to the number of basis vectors available
        n_shp = self.bfm_w_shp.shape[1]
        n_exp = self.bfm_w_exp.shape[1]
        avg_id  = avg_id[:n_shp]
        avg_exp = avg_exp[:n_exp]

        pts      = self.bfm_u + self.bfm_w_shp @ avg_id + self.bfm_w_exp @ avg_exp
        vertices = pts.reshape(-1, 3).astype(np.float32)

        result = {'vertices': vertices, 'faces': self.tri}

        # Compute per-vertex albedo color from texture coefficients [144:224]
        if self.bfm_meantex is not None and self.bfm_texBase is not None:
            avg_tex = np.mean(params_array[:, 144:224], axis=0)  # (80,)
            n_tex = self.bfm_texBase.shape[1]
            avg_tex = avg_tex[:n_tex]
            tex_flat = self.bfm_meantex + self.bfm_texBase @ avg_tex  # (N*3,) RGB 0-255
            tex_rgb = np.clip(tex_flat.reshape(-1, 3), 0, 255).astype(np.uint8)
            result['vertex_colors'] = tex_rgb

        return result

    def reconstruct(self, frame: np.ndarray, bbox: np.ndarray,
                    kps: Optional[np.ndarray] = None) -> Optional[Dict]:
        params = self.infer_params(frame, bbox, kps=kps)
        if params is None:
            return None
        return self.reconstruct_from_params_list([params])

    @staticmethod
    def to_glb_bytes(vertices: np.ndarray, faces: np.ndarray,
                     vertex_colors: Optional[np.ndarray] = None) -> bytes:
        import trimesh

        mesh = trimesh.Trimesh(
            vertices=vertices.astype(np.float32),
            faces=faces.astype(np.int32),
            process=False
        )

        if vertex_colors is not None:
            n = len(vertices)
            rgba = np.ones((n, 4), dtype=np.uint8) * 255
            rgba[:, :3] = vertex_colors[:n, :3]
            mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=rgba)

        mesh.fix_normals()

        buffer = io.BytesIO()
        mesh.export(buffer, file_type='glb')
        return buffer.getvalue()
