"""Standalone tests for src/face3d_store.py (pack/unpack, eviction, build cadence)."""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.face3d_store import (
    Face3DAccumulator, FrameRecord, pack_frames, unpack_frames,
    EMBEDDING_DIM, PARAMS_DIM, SCHEMA_VERSION,
)


def _rec(yaw, q=1.0, cam="stream:cam1"):
    return FrameRecord(
        yaw=yaw, pitch=2.0, roll=-3.0, quality=q,
        timestamp=1700000000, camera_id=cam,
        embedding=np.random.randn(EMBEDDING_DIM).astype(np.float32),
        params=np.random.randn(PARAMS_DIM).astype(np.float32),
    )


def test_pack_unpack_roundtrip():
    recs = [_rec(float(i * 5 - 50), q=i / 10.0) for i in range(20)]
    # include a NaN-yaw record
    recs.append(_rec(float("nan"), q=0.5, cam="x"))
    blob = pack_frames(recs)
    ver, out = unpack_frames(blob)
    assert ver == SCHEMA_VERSION
    assert len(out) == len(recs)
    for a, b in zip(recs, out):
        assert b.camera_id == a.camera_id
        assert b.timestamp == a.timestamp
        if np.isnan(a.yaw):
            assert np.isnan(b.yaw)
        else:
            assert abs(a.yaw - b.yaw) < 0.1
        # float16 round-trip tolerance
        ea = np.asarray(a.embedding, dtype=np.float16).astype(np.float32)
        eb = np.asarray(b.embedding, dtype=np.float32)
        assert np.allclose(ea, eb, atol=1e-3)
        pa = np.asarray(a.params, dtype=np.float16).astype(np.float32)
        pb = np.asarray(b.params, dtype=np.float32)
        assert np.allclose(pa, pb, atol=1e-3)
    print("PASS pack/unpack round-trip (float16 tolerant, NaN yaw preserved)")


def test_bin_balanced_eviction():
    acc = Face3DAccumulator(min_samples=50, refine_interval=50, max_frames=300,
                            yaw_bin_width=30.0)
    pid = "p1"
    # Feed 500 frames: heavily skewed toward the frontal bin (yaw 0) plus some spread.
    for i in range(500):
        if i % 5 == 0:
            yaw = -45.0  # bin -2
        elif i % 5 == 1:
            yaw = 45.0   # bin 1
        else:
            yaw = 0.0    # bin 0 (dominant)
        acc.add(pid, np.random.randn(EMBEDDING_DIM), yaw, 1.0, 1.0,
                quality=np.random.rand(), camera_id="c", params=np.random.randn(PARAMS_DIM))
    st = acc._people[pid]
    assert len(st.records) <= 300, f"ring exceeded ceiling: {len(st.records)}"
    assert st.frame_count == 500, f"lifetime counter wrong: {st.frame_count}"
    bins = {}
    for r in st.records:
        b = int(np.floor(r.yaw / 30.0))
        bins[b] = bins.get(b, 0) + 1
    # All three originally-populated bins must survive (diversity preserved).
    assert -2 in bins and 0 in bins and 1 in bins, f"bins lost: {bins}"
    print(f"PASS bin-balanced eviction: ring={len(st.records)}<=300, "
          f"frame_count=500, bins preserved={sorted(bins)}")


class _FakeReconstructor:
    def __init__(self):
        self.calls = 0

    def reconstruct_from_params_list(self, params_list):
        self.calls += 1
        return {"vertices": np.zeros((3, 3)), "faces": np.zeros((1, 3), dtype=np.int32)}

    def to_glb_bytes(self, vertices, faces, vertex_colors=None):
        return b"GLB"


class _FakeDB:
    def __init__(self):
        self.uploads = []
        self.puts = 0

    def update_person_face3d(self, person_id, glb_bytes):
        self.uploads.append(person_id)

    def put_face3d_frames(self, person_id, fc, lbac, sv, b64):
        self.puts += 1

    def get_face3d_frames(self, person_id):
        return None


def test_build_cadence():
    acc = Face3DAccumulator(min_samples=50, refine_interval=50, max_frames=300,
                            yaw_bin_width=30.0)
    rec = _FakeReconstructor()
    db = _FakeDB()
    pid = "p2"
    build_at = []
    # Drive past the 300 ceiling: frame_count is a lifetime cadence counter, so
    # builds continue every 50 (350, 400) using the retained 300-frame ring.
    for n in range(1, 401):
        acc.add(pid, np.random.randn(EMBEDDING_DIM), 0.0, 0.0, 0.0,
                quality=1.0, camera_id="c", params=np.random.randn(PARAMS_DIM))
        built = acc.maybe_build(pid, rec, db)
        if built:
            build_at.append(n)
    assert build_at == [50, 100, 150, 200, 250, 300, 350, 400], f"unexpected: {build_at}"
    # No builds at any non-multiple-of-50 frame.
    assert all(n % 50 == 0 for n in build_at)
    assert rec.calls == 8
    assert db.uploads == [pid] * 8
    print(f"PASS build cadence: triggered exactly at {build_at} (every 50, "
          f"continues past 300 ceiling), none between")


def test_single_bin_builds():
    acc = Face3DAccumulator(min_samples=50, refine_interval=50, max_frames=300,
                            yaw_bin_width=30.0)
    rec = _FakeReconstructor()
    db = _FakeDB()
    pid = "p3"
    # All frames frontal -> a single yaw bin. Build must still proceed.
    for _ in range(50):
        acc.add(pid, np.random.randn(EMBEDDING_DIM), 0.0, 0.0, 0.0,
                quality=1.0, camera_id="c", params=np.random.randn(PARAMS_DIM))
    built = acc.maybe_build(pid, rec, db)
    assert built is True, "single-bin build did not proceed"
    assert acc._populated_bins(acc._people[pid]) == 1
    print("PASS best-effort build proceeds with a single populated bin")


def test_nan_pose_builds():
    acc = Face3DAccumulator(min_samples=50, refine_interval=50, max_frames=300,
                            yaw_bin_width=30.0)
    rec = _FakeReconstructor()
    db = _FakeDB()
    pid = "p4"
    # All frames have failed pose (yaw None -> NaN). Still counts + builds.
    for _ in range(50):
        acc.add(pid, np.random.randn(EMBEDDING_DIM), None, None, None,
                quality=1.0, camera_id="c", params=np.random.randn(PARAMS_DIM))
    built = acc.maybe_build(pid, rec, db)
    assert built is True
    print("PASS best-effort build proceeds with only NaN-pose frames")


def test_build_failure_leaves_counter():
    class FailRec(_FakeReconstructor):
        def reconstruct_from_params_list(self, params_list):
            raise RuntimeError("boom")

    acc = Face3DAccumulator(min_samples=50, refine_interval=50, max_frames=300)
    rec = FailRec()
    db = _FakeDB()
    pid = "p5"
    for _ in range(50):
        acc.add(pid, np.random.randn(EMBEDDING_DIM), 0.0, 0.0, 0.0,
                quality=1.0, camera_id="c", params=np.random.randn(PARAMS_DIM))
    built = acc.maybe_build(pid, rec, db)
    assert built is False
    assert acc._people[pid].last_built_at_count == 0, "counter advanced despite failure"
    print("PASS build failure leaves last_built_at_count unchanged (retries next step)")


if __name__ == "__main__":
    test_pack_unpack_roundtrip()
    test_bin_balanced_eviction()
    test_build_cadence()
    test_single_bin_builds()
    test_nan_pose_builds()
    test_build_failure_leaves_counter()
    print("\nALL FACE3D STORE TESTS PASSED")
