#!/usr/bin/env python3
"""
One-time setup script for Deep3DFaceRecon.

Downloads pretrained weights and BFM09 data from HuggingFace (vinthony/SadTalker),
builds the ReconNet (ResNet50 + 7 parallel 1x1 conv coefficient heads), exports to
ONNX, and preprocesses BFM09 basis to npz format.

Run from the reid-worker directory:
    python scripts/setup_deep3d.py

Produces:
    face-reidentification/weights/deep3d_recon.onnx  (input [1,3,224,224] -> [1,257])
    face-reidentification/weights/deep3d_recon.npz   (u, w_shp, w_exp, tri)
"""

import sys
import struct
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

WEIGHTS_DIR = Path(__file__).parent.parent / 'face-reidentification' / 'weights'
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

BFM_DIR = WEIGHTS_DIR / 'BFM_Fitting'

ONNX_PATH = WEIGHTS_DIR / 'deep3d_recon.onnx'
NPZ_PATH  = WEIGHTS_DIR / 'deep3d_recon.npz'
PTH_PATH  = WEIGHTS_DIR / 'deep3d_recon_epoch20.pth'

# Source: vinthony/SadTalker on HuggingFace (bundles Deep3DFaceRecon checkpoint + BFM09)
HF_REPO  = 'vinthony/SadTalker'
HF_TOKEN = None  # set via --token flag or HF_TOKEN env var


def hf_download(filename: str, dest_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download
    print(f'  Downloading {filename} ...')
    path = Path(hf_hub_download(repo_id=HF_REPO, filename=filename,
                                local_dir=str(dest_dir), token=HF_TOKEN))
    print(f'    -> {path} ({path.stat().st_size / 1e6:.1f} MB)')
    return path


def download_weights() -> None:
    if PTH_PATH.exists():
        print(f'Checkpoint already present: {PTH_PATH}')
        return
    print('Downloading Deep3DFaceRecon checkpoint...')
    src = hf_download('epoch_20.pth', WEIGHTS_DIR)
    if src != PTH_PATH:
        import shutil
        shutil.move(str(src), str(PTH_PATH))
    print(f'  Saved as {PTH_PATH}')


def download_bfm() -> None:
    """Download raw BFM09 files from SadTalker repo."""
    needed = [
        'BFM_Fitting/01_MorphableModel.mat',
        'BFM_Fitting/Exp_Pca.bin',
        'BFM_Fitting/BFM_front_idx.mat',
        'BFM_Fitting/BFM_exp_idx.mat',
        'BFM_Fitting/facemodel_info.mat',
        'BFM_Fitting/std_exp.txt',
        'BFM_Fitting/similarity_Lm3D_all.mat',
    ]
    all_present = all((WEIGHTS_DIR / f).exists() for f in needed)
    if all_present:
        print('BFM files already present')
        return
    print('Downloading BFM09 basis files...')
    for f in needed:
        dest = WEIGHTS_DIR / f
        if not dest.exists():
            hf_download(f, WEIGHTS_DIR)


def build_bfm_model_front() -> dict:
    """
    Preprocess raw BFM09 + Facewarehouse expression basis into the
    BFM_model_front format used by Deep3DFaceRecon.

    Follows Deep3DFaceRecon_pytorch transferBFM09() scaling:
      idBase    = shapePC * shapeEV / 1e5
      exBase    = expPC   * expEV   / 1e5
      texBase   = texPC   * texEV           (no 1e5, stays 0-255)
      meanshape = shapeMU / 1e5
      meantex   = texMU                     (no 1e5)

    Vertex selection via BFM_front_idx (35709 front-face vertices):
      - Shape/texture indexed via trimIndex into BFM 53490 space
      - Expression indexed directly into exp 53215 space

    Returns dict with keys:
      meanshape, idBase, exBase, meantex, texBase, tri
    """
    from scipy.io import loadmat

    # --- Load raw BFM09 ---
    morph_path = BFM_DIR / '01_MorphableModel.mat'
    print(f'  Loading {morph_path}')
    raw = loadmat(str(morph_path))

    shape_mu = np.array(raw['shapeMU']).flatten().astype(np.float64)   # [53490*3]
    shape_pc = np.array(raw['shapePC']).astype(np.float64)             # [53490*3, 199]
    shape_ev = np.array(raw['shapeEV']).flatten().astype(np.float64)   # [199]
    tex_mu   = np.array(raw['texMU']).flatten().astype(np.float64)     # [53490*3]
    tex_pc   = np.array(raw['texPC']).astype(np.float64)               # [53490*3, 199]
    tex_ev   = np.array(raw['texEV']).flatten().astype(np.float64)     # [199]

    n_bfm = shape_mu.shape[0] // 3  # 53490
    print(f'  Full BFM: {n_bfm} vertices, {shape_pc.shape[1]} id components')

    # --- Load expression basis (53215 vertices) ---
    exp_path = BFM_DIR / 'Exp_Pca.bin'
    with open(exp_path, 'rb') as f:
        n_exp_comps = struct.unpack('<i', f.read(4))[0]  # 79
        data = np.frombuffer(f.read(), dtype=np.float32)
    n_exp_verts = len(data) // (3 * (1 + n_exp_comps))  # 53215
    mu_size = n_exp_verts * 3
    # Binary layout is component-major: [n_comps, n_verts*3], must transpose
    exp_pc = data[mu_size:].reshape(n_exp_comps, n_exp_verts * 3).T.astype(np.float64)
    print(f'  Exp_Pca.bin: {n_exp_comps} components, {n_exp_verts} vertices')

    # --- Load expression eigenvalues ---
    exp_ev_path = BFM_DIR / 'std_exp.txt'
    exp_ev = np.loadtxt(str(exp_ev_path), dtype=np.float64)  # [79]
    print(f'  Expression eigenvalues: {exp_ev.shape[0]} values')

    # --- Scale basis vectors (Deep3DFaceRecon_pytorch transferBFM09) ---
    id_base  = shape_pc[:, :80] * shape_ev[:80].reshape(1, -1) / 1e5
    ex_base  = exp_pc[:, :64]   * exp_ev[:64].reshape(1, -1)   / 1e5
    tex_base = tex_pc[:, :80]   * tex_ev[:80].reshape(1, -1)          # no 1e5
    mean_shape = shape_mu / 1e5
    mean_tex   = tex_mu                                                # no 1e5

    # --- Vertex selection ---
    # BFM_front_idx: 35709 indices into 53215 exp vertex space
    front_data = loadmat(str(BFM_DIR / 'BFM_front_idx.mat'))
    front_idx = np.array(front_data['idx']).flatten().astype(np.int64) - 1  # 0-based

    # trimIndex: maps 53215 exp vertices -> 53490 BFM vertices
    trim_data = loadmat(str(BFM_DIR / 'BFM_exp_idx.mat'))
    trim_idx = np.array(trim_data['trimIndex']).flatten().astype(np.int64) - 1  # 0-based

    # Map front vertices to BFM space for shape/texture
    shape_idx = trim_idx[front_idx]  # [35709] indices into 53490 BFM space
    n_front = len(front_idx)
    print(f'  Front vertices: {n_front}')

    # Coordinate indices for shape/texture (BFM 53490 space)
    sci = np.empty(n_front * 3, dtype=np.int64)
    sci[0::3] = shape_idx * 3
    sci[1::3] = shape_idx * 3 + 1
    sci[2::3] = shape_idx * 3 + 2

    # Coordinate indices for expression (exp 53215 space)
    eci = np.empty(n_front * 3, dtype=np.int64)
    eci[0::3] = front_idx * 3
    eci[1::3] = front_idx * 3 + 1
    eci[2::3] = front_idx * 3 + 2

    front_mean_shape = mean_shape[sci]          # [35709*3]
    front_id_base    = id_base[sci]             # [35709*3, 80]
    front_ex_base    = ex_base[eci]             # [35709*3, 64]
    front_mean_tex   = mean_tex[sci]            # [35709*3]
    front_tex_base   = tex_base[sci]            # [35709*3, 80]

    # --- Front triangles from facemodel_info.mat ---
    info = loadmat(str(BFM_DIR / 'facemodel_info.mat'))
    front_tri = np.array(info['tri']).astype(np.int32) - 1  # 0-based

    print(f'  Front mesh: {n_front} vertices, {front_tri.shape[0]} triangles')
    print(f'  idBase: {front_id_base.shape}, exBase: {front_ex_base.shape}, '
          f'texBase: {front_tex_base.shape}')

    # Sanity: mean shape bbox should be ~1-2 units after /1e5 scaling
    ms_verts = front_mean_shape.reshape(-1, 3)
    bbox_size = ms_verts.max(axis=0) - ms_verts.min(axis=0)
    print(f'  Mean shape bbox: {bbox_size}')

    # --- 3D reference landmarks for face alignment (from similarity_Lm3D_all.mat) ---
    lm3d_path = BFM_DIR / 'similarity_Lm3D_all.mat'
    lm3d_data = loadmat(str(lm3d_path))
    lm68 = lm3d_data['lm']  # (68, 3)
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    lm3d_5 = np.stack([
        lm68[lm_idx[0], :],
        np.mean(lm68[lm_idx[[1, 2]], :], 0),
        np.mean(lm68[lm_idx[[3, 4]], :], 0),
        lm68[lm_idx[5], :],
        lm68[lm_idx[6], :],
    ], axis=0)
    lm3d_5 = lm3d_5[[1, 2, 0, 3, 4], :]  # reorder: L_eye, R_eye, nose, L_mouth, R_mouth
    print(f'  3D reference landmarks: {lm3d_5.shape}')

    return {
        'meanshape': front_mean_shape.astype(np.float32),
        'idBase':    front_id_base.astype(np.float32),
        'exBase':    front_ex_base.astype(np.float32),
        'meantex':   front_mean_tex.astype(np.float32),
        'texBase':   front_tex_base.astype(np.float32),
        'tri':       front_tri,
        'lm3d':      lm3d_5.astype(np.float32),
    }


def build_recon_net():
    """
    ReconNet matching sicxu/Deep3DFaceRecon_pytorch (SadTalker's epoch_20.pth).

    Architecture (from checkpoint key inspection):
      - backbone: ResNet50 conv1..layer4 (no avgpool, no fc)
                  output: [B, 2048, 7, 7]
      - Global average pool applied in forward: [B, 2048, 1, 1]
      - final_layers: 7 parallel 1x1 Conv2d heads (all with bias), each reading
        from the same pooled features independently:
          [0] Conv2d(2048, 80)  -> identity shape
          [1] Conv2d(2048, 64)  -> expression
          [2] Conv2d(2048, 80)  -> texture
          [3] Conv2d(2048, 3)   -> Euler rotation angles
          [4] Conv2d(2048, 27)  -> SH lighting
          [5] Conv2d(2048, 2)   -> translation XY
          [6] Conv2d(2048, 1)   -> translation Z
      - Outputs concatenated -> [B, 257]

    Coefficient layout:
      [0:80]    identity shape
      [80:144]  expression
      [144:224] texture
      [224:227] Euler rotation
      [227:254] SH lighting
      [254:256] translation XY
      [256:257] translation Z
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.models as M

    class _Backbone(nn.Module):
        """ResNet50 feature extractor (output before avgpool)."""
        def __init__(self):
            super().__init__()
            r = M.resnet50(weights=None)
            self.conv1   = r.conv1
            self.bn1     = r.bn1
            self.relu    = r.relu
            self.maxpool = r.maxpool
            self.layer1  = r.layer1
            self.layer2  = r.layer2
            self.layer3  = r.layer3
            self.layer4  = r.layer4

        def forward(self, x):
            x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
            return self.layer4(self.layer3(self.layer2(self.layer1(x))))

    class ReconNet(nn.Module):
        fc_dims = [80, 64, 80, 3, 27, 2, 1]  # sum = 257

        def __init__(self):
            super().__init__()
            self.backbone = _Backbone()
            self.final_layers = nn.ModuleList([
                nn.Conv2d(2048, d, 1) for d in self.fc_dims
            ])

        def forward(self, x):
            x = self.backbone(x)
            x = F.adaptive_avg_pool2d(x, 1)  # [B, 2048, 1, 1]
            return torch.cat([layer(x).flatten(1) for layer in self.final_layers], dim=1)

    return ReconNet()


def load_checkpoint(model, pth_path: Path):
    import torch

    ckpt = torch.load(str(pth_path), map_location='cpu', weights_only=False)

    if isinstance(ckpt, dict):
        for key in ('net_recon', 'state_dict', 'model', 'params'):
            if key in ckpt:
                state = ckpt[key]
                print(f'Using checkpoint["{key}"] ({len(state)} keys)')
                break
        else:
            state = ckpt
    else:
        state = ckpt

    result = model.load_state_dict(state, strict=False)
    n_loaded = len(model.state_dict()) - len(result.missing_keys)
    n_total  = len(model.state_dict())
    print(f'Loaded {n_loaded}/{n_total} tensors')

    if result.missing_keys:
        print(f'  Missing: {result.missing_keys[:5]}')
    if result.unexpected_keys:
        print(f'  Unexpected: {result.unexpected_keys[:5]}')

    if n_loaded < n_total // 2:
        raise RuntimeError(
            f'Too few weights loaded ({n_loaded}/{n_total}). '
            f'Checkpoint architecture mismatch.'
        )
    return model


def export_onnx(model) -> None:
    import torch

    model.eval()
    dummy = torch.zeros(1, 3, 224, 224)

    with torch.no_grad():
        out = model(dummy)
    print(f'Forward pass output shape: {out.shape}')
    assert out.shape == (1, 257), f'Expected (1, 257), got {out.shape}'

    print(f'Exporting ONNX -> {ONNX_PATH}')
    torch.onnx.export(
        model, dummy, str(ONNX_PATH),
        input_names=['input'],
        output_names=['output'],
        opset_version=17,
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
    )
    print(f'  Exported: {ONNX_PATH.stat().st_size / 1e6:.1f} MB')


def extract_bfm_basis() -> None:
    """Build BFM front-face model from raw files and save as npz."""
    bfm = build_bfm_model_front()

    u       = bfm['meanshape']  # (N*3,)
    w_shp   = bfm['idBase']     # (N*3, 80)
    w_exp   = bfm['exBase']     # (N*3, 64)
    meantex = bfm['meantex']    # (N*3,)   mean texture (RGB 0-255)
    texBase = bfm['texBase']    # (N*3, 80) texture PCA basis
    tri     = bfm['tri']        # (T, 3), 0-based
    lm3d    = bfm['lm3d']       # (5, 3), 3D reference landmarks

    np.savez(str(NPZ_PATH), u=u, w_shp=w_shp, w_exp=w_exp,
             meantex=meantex, texBase=texBase, tri=tri, lm3d=lm3d)
    print(f'BFM basis saved: {NPZ_PATH}')
    print(f'  u={u.shape}, w_shp={w_shp.shape}, w_exp={w_exp.shape}')
    print(f'  meantex={meantex.shape}, texBase={texBase.shape}, tri={tri.shape}')
    print(f'  lm3d={lm3d.shape}')


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description='Setup Deep3DFaceRecon')
    parser.add_argument('--token', type=str, default=None,
                        help='HuggingFace token (or set HF_TOKEN env var)')
    args = parser.parse_args()

    global HF_TOKEN
    import os
    HF_TOKEN = args.token or os.environ.get('HF_TOKEN')

    print('=== Deep3DFaceRecon Setup ===')
    print(f'Output directory: {WEIGHTS_DIR}\n')

    download_weights()
    download_bfm()

    if not ONNX_PATH.exists():
        print('\nBuilding ReconNet...')
        model = build_recon_net()
        n_params = sum(p.numel() for p in model.parameters())
        print(f'  Parameters: {n_params:,}')

        print('Loading pretrained weights...')
        model = load_checkpoint(model, PTH_PATH)
        export_onnx(model)
    else:
        print(f'\nONNX already exists: {ONNX_PATH}')

    if not NPZ_PATH.exists():
        print('\nPreprocessing BFM09 basis...')
        extract_bfm_basis()
    else:
        print(f'NPZ already exists: {NPZ_PATH}')

    print('\n=== Setup complete ===')
    print(f'  ONNX:  {ONNX_PATH}')
    print(f'  NPZ:   {NPZ_PATH}')
    print('\nNext: python main.py --debug')


if __name__ == '__main__':
    main()
