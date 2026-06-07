#!/usr/bin/env python3
"""
Export 3DDFA_V2 model to ONNX format and extract BFM basis matrices.

Clones the 3DDFA_V2 repo, loads the pretrained MobileNet checkpoint,
exports to ONNX, and saves BFM basis matrices + param normalization as .npz.

Output:
  face-reidentification/weights/3ddfa_v2.onnx  -- MobileNet [1,3,120,120] -> [1,62]
  face-reidentification/weights/3ddfa_v2.npz   -- BFM basis (u, w_shp, w_exp, tri) + param_mean/param_std

Requires: torch, onnxruntime (should already be in the venv)
"""

import sys
import subprocess
import tempfile
from pathlib import Path

WEIGHTS_DIR = Path(__file__).parent.parent / "face-reidentification" / "weights"
REPO_URL = "https://github.com/cleardusk/3DDFA_V2.git"
CHECKPOINT_URL = "https://github.com/cleardusk/3DDFA_V2/raw/master/weights/mb1_120x120.pth"


def download_file(url: str, dest: Path):
    """Download a file using curl"""
    print(f"Downloading {url} -> {dest}")
    result = subprocess.run(
        ["curl", "-L", "-o", str(dest), url],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Download failed: {result.stderr}")
    if not dest.exists() or dest.stat().st_size == 0:
        raise RuntimeError(f"Downloaded file is empty: {dest}")


def clone_repo(work_dir: Path) -> Path:
    """Clone 3DDFA_V2 repo"""
    repo_dir = work_dir / "3DDFA_V2"
    if not repo_dir.exists():
        print("Cloning 3DDFA_V2 repository...")
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, str(repo_dir)],
            check=True
        )
    return repo_dir


def export_onnx(repo_dir: Path, work_dir: Path):
    """Export MobileNet backbone to ONNX"""
    import torch
    import numpy as np

    # Download checkpoint
    checkpoint_path = work_dir / "mb1_120x120.pth"
    if not checkpoint_path.exists():
        download_file(CHECKPOINT_URL, checkpoint_path)

    # Add repo to path for imports
    sys.path.insert(0, str(repo_dir))

    # Use the repo's own model loading -- arch='mobilenet', accessed via getattr(models, 'mobilenet')
    import models as tddfa_models

    model = tddfa_models.mobilenet(num_classes=62, widen_factor=1.0)

    # Load weights using 3DDFA_V2's own loading logic
    checkpoint = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint)

    model_dict = model.state_dict()
    for k in state_dict.keys():
        kc = k.replace('module.', '')
        if kc in model_dict:
            model_dict[kc] = state_dict[k]
        # Handle fc_param -> fc rename
        if kc in ['fc_param.bias', 'fc_param.weight']:
            model_dict[kc.replace('_param', '')] = state_dict[k]

    model.load_state_dict(model_dict)
    model.eval()

    # Export to ONNX
    dummy_input = torch.randn(1, 3, 120, 120)
    onnx_path = WEIGHTS_DIR / "3ddfa_v2.onnx"

    print(f"Exporting to ONNX: {onnx_path}")
    torch.onnx.export(
        model,
        (dummy_input,),
        str(onnx_path),
        input_names=['input'],
        output_names=['output'],
        do_constant_folding=True,
        opset_version=11
    )

    # Verify
    import onnxruntime
    sess = onnxruntime.InferenceSession(str(onnx_path))
    result = sess.run(None, {'input': dummy_input.numpy()})
    print(f"ONNX output shape: {result[0].shape}")
    assert result[0].shape == (1, 62), f"Unexpected output shape: {result[0].shape}"
    print(f"ONNX export successful: {onnx_path}")


def extract_bfm_and_params(repo_dir: Path):
    """Extract BFM basis matrices and param normalization from 3DDFA_V2 repo"""
    import numpy as np
    import pickle

    npz_path = WEIGHTS_DIR / "3ddfa_v2.npz"
    save_data = {}

    # Load BFM data
    bfm_path = repo_dir / "configs" / "bfm_noneck_v3.pkl"
    if not bfm_path.exists():
        raise FileNotFoundError(f"BFM file not found: {bfm_path}")

    print(f"Loading BFM data from {bfm_path}")
    with open(bfm_path, 'rb') as f:
        bfm = pickle.load(f, encoding='latin1')

    u = np.array(bfm['u'], dtype=np.float32)
    save_data['u'] = u
    print(f"  u (mean shape): {u.shape}")

    w_shp = np.array(bfm['w_shp'], dtype=np.float32)[:, :40]
    save_data['w_shp'] = w_shp
    print(f"  w_shp (shape basis): {w_shp.shape}")

    w_exp = np.array(bfm['w_exp'], dtype=np.float32)[:, :10]
    save_data['w_exp'] = w_exp
    print(f"  w_exp (expression basis): {w_exp.shape}")

    # tri for bfm_noneck_v3 is stored separately in tri.pkl
    tri_path = repo_dir / "configs" / "tri.pkl"
    if tri_path.exists():
        with open(tri_path, 'rb') as f:
            tri = pickle.load(f, encoding='latin1')
        # tri comes as [3, T], need [T, 3] for our face3d.py
        tri = np.array(tri, dtype=np.int32).T
        save_data['tri'] = tri
        print(f"  tri (triangles): {tri.shape}")
    else:
        print(f"  WARNING: tri.pkl not found, falling back to bfm['tri']")
        if 'tri' in bfm:
            tri = np.array(bfm['tri'], dtype=np.int32).T
            save_data['tri'] = tri
            print(f"  tri (triangles): {tri.shape}")

    if 'keypoints' in bfm:
        kp = np.array(bfm['keypoints'], dtype=np.int64)
        save_data['keypoints'] = kp
        print(f"  keypoints: {kp.shape}")

    # Load param normalization (mean/std for denormalizing raw ONNX output)
    param_fp = repo_dir / "configs" / "param_mean_std_62d_120x120.pkl"
    if param_fp.exists():
        with open(param_fp, 'rb') as f:
            param_data = pickle.load(f, encoding='latin1')
        param_mean = np.array(param_data['mean'], dtype=np.float32)
        param_std = np.array(param_data['std'], dtype=np.float32)
        save_data['param_mean'] = param_mean
        save_data['param_std'] = param_std
        print(f"  param_mean: {param_mean.shape}")
        print(f"  param_std: {param_std.shape}")
    else:
        print(f"  WARNING: param normalization file not found: {param_fp}")

    np.savez(str(npz_path), **save_data)
    print(f"BFM data saved: {npz_path}")


def main():
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    onnx_path = WEIGHTS_DIR / "3ddfa_v2.onnx"
    npz_path = WEIGHTS_DIR / "3ddfa_v2.npz"

    if onnx_path.exists() and npz_path.exists():
        print(f"Both files already exist:")
        print(f"  {onnx_path} ({onnx_path.stat().st_size / 1024:.0f} KB)")
        print(f"  {npz_path} ({npz_path.stat().st_size / 1024:.0f} KB)")
        print("Delete them to re-export.")
        return

    with tempfile.TemporaryDirectory(prefix="3ddfa_export_") as tmpdir:
        work_dir = Path(tmpdir)
        repo_dir = clone_repo(work_dir)

        if not onnx_path.exists():
            export_onnx(repo_dir, work_dir)
        else:
            print(f"ONNX already exists: {onnx_path}")

        if not npz_path.exists():
            extract_bfm_and_params(repo_dir)
        else:
            print(f"NPZ already exists: {npz_path}")

    print("\nDone. Files:")
    for p in [onnx_path, npz_path]:
        if p.exists():
            print(f"  {p} ({p.stat().st_size / 1024:.0f} KB)")
        else:
            print(f"  {p} -- MISSING")


if __name__ == "__main__":
    main()
