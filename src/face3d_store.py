"""
Persistent, traceable 3D face-frame accumulation.

Holds per-person in-RAM rings of per-frame records (pose + ArcFace embedding +
257-dim 3DDFA params), packs them into a compact zlib-compressed blob, and drives
best-effort 3D mesh builds on a fixed frame-count cadence.

Blob format (must match reid-db-handler exactly):
  zlib.compress( header + records )
    header  = magic b"F3" (2) + schema_version uint8 (1) + record_count uint16 LE (2)
    record  = yaw f16 (2) + pitch f16 (2) + roll f16 (2) + quality f16 (2)
              + timestamp uint32 LE (4)
              + camera_len uint8 (1) + camera utf8 (camera_len)
              + embedding 512 x f16 (1024)
              + params 257 x f16 (514)
  yaw = NaN (f16) when pose is None.

reid-worker NEVER touches the database directly; all persistence goes through the
reid-db-handler HTTP API via db_client.
"""

import base64
import struct
import zlib
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Callable

import numpy as np

from src.db_client import PersonNotFoundError

logger = logging.getLogger(__name__)

MAGIC = b"F3"
SCHEMA_VERSION = 1

EMBEDDING_DIM = 512
PARAMS_DIM = 257

# Fixed-width portion of a record (everything except the variable camera tag):
#   yaw f16 + pitch f16 + roll f16 + quality f16 + timestamp u32
_HEAD_FMT = "<eeeeI"
_HEAD_LEN = struct.calcsize(_HEAD_FMT)  # 12 bytes
_EMB_BYTES = EMBEDDING_DIM * 2          # float16
_PARAMS_BYTES = PARAMS_DIM * 2          # float16


@dataclass
class FrameRecord:
    """A single accumulated frame. yaw may be NaN when pose estimation failed."""
    yaw: float
    pitch: float
    roll: float
    quality: float
    timestamp: int            # epoch seconds
    camera_id: str
    embedding: np.ndarray     # (512,) float, stored as float16
    params: np.ndarray        # (257,) float, stored as float16


def pack_frames(records: List[FrameRecord]) -> bytes:
    """Pack records into the zlib-compressed blob described in the module docstring."""
    out = bytearray()
    out += MAGIC
    out += struct.pack("<B", SCHEMA_VERSION)
    out += struct.pack("<H", len(records))

    for r in records:
        cam = r.camera_id.encode("utf-8")
        if len(cam) > 255:
            cam = cam[:255]
        out += struct.pack(
            _HEAD_FMT,
            np.float16(r.yaw), np.float16(r.pitch),
            np.float16(r.roll), np.float16(r.quality),
            int(r.timestamp) & 0xFFFFFFFF,
        )
        out += struct.pack("<B", len(cam))
        out += cam
        emb = np.asarray(r.embedding, dtype=np.float16)
        if emb.shape[0] != EMBEDDING_DIM:
            raise ValueError(f"embedding dim {emb.shape[0]} != {EMBEDDING_DIM}")
        out += emb.tobytes()
        par = np.asarray(r.params, dtype=np.float16)
        if par.shape[0] != PARAMS_DIM:
            raise ValueError(f"params dim {par.shape[0]} != {PARAMS_DIM}")
        out += par.tobytes()

    return zlib.compress(bytes(out))


def unpack_frames(blob: bytes) -> Tuple[int, List[FrameRecord]]:
    """Inverse of pack_frames. Returns (schema_version, records)."""
    raw = zlib.decompress(blob)
    if raw[:2] != MAGIC:
        raise ValueError("bad magic in face3d frames blob")
    schema_version = raw[2]
    (count,) = struct.unpack_from("<H", raw, 3)

    records: List[FrameRecord] = []
    offset = 5
    for _ in range(count):
        yaw, pitch, roll, quality, ts = struct.unpack_from(_HEAD_FMT, raw, offset)
        offset += _HEAD_LEN
        (cam_len,) = struct.unpack_from("<B", raw, offset)
        offset += 1
        camera_id = raw[offset:offset + cam_len].decode("utf-8")
        offset += cam_len
        emb = np.frombuffer(raw[offset:offset + _EMB_BYTES], dtype=np.float16).copy()
        offset += _EMB_BYTES
        par = np.frombuffer(raw[offset:offset + _PARAMS_BYTES], dtype=np.float16).copy()
        offset += _PARAMS_BYTES
        records.append(FrameRecord(
            yaw=float(yaw), pitch=float(pitch), roll=float(roll),
            quality=float(quality), timestamp=int(ts), camera_id=camera_id,
            embedding=emb, params=par,
        ))

    return schema_version, records


def _yaw_bin(yaw: float, bin_width: float) -> int:
    """Bin index for a yaw angle. NaN-yaw records get their own sentinel bucket."""
    if yaw is None or np.isnan(yaw):
        return _NAN_BIN
    return int(np.floor(yaw / bin_width))


# Sentinel bucket for records without a valid pose. Kept distinct from any real
# yaw bin so it never participates in diversity selection and is evicted last.
_NAN_BIN = 1 << 30


class _PersonState:
    __slots__ = ("records", "frame_count", "last_built_at_count",
                 "db_person_id", "adds_since_persist", "rehydrated")

    def __init__(self):
        self.records: List[FrameRecord] = []
        self.frame_count: int = 0
        self.last_built_at_count: int = 0
        self.db_person_id: Optional[str] = None
        self.adds_since_persist: int = 0
        self.rehydrated: bool = False


class Face3DAccumulator:
    """
    Per-person 3D-frame accumulator with bin-balanced eviction, cadence-driven
    best-effort builds, debounced persistence, and lazy rehydration.

    Keyed entirely on the persistent DB person id (the same id update_person_face3d
    uses). Stateless across what id maps to which tracker session.
    """

    def __init__(self, min_samples: int = 50, refine_interval: int = 50,
                 max_frames: int = 300, yaw_bin_width: float = 30.0,
                 persist_every: int = 1):
        self.min_samples = int(min_samples)
        self.refine_interval = int(refine_interval)
        self.max_frames = int(max_frames)
        self.yaw_bin_width = float(yaw_bin_width)
        self.persist_every = int(persist_every)
        self._people: Dict[str, _PersonState] = {}

    def _state(self, person_db_id: str) -> _PersonState:
        st = self._people.get(person_db_id)
        if st is None:
            st = _PersonState()
            st.db_person_id = person_db_id
            self._people[person_db_id] = st
        return st

    def evict(self, person_db_id: str) -> None:
        """Drop a person's in-RAM ring state (after the DB id is gone)."""
        self._people.pop(person_db_id, None)

    def _handle_person_gone(self, person_db_id: str,
                            on_person_gone: Optional[Callable[[str], None]]) -> None:
        """A persistence call hit a 404: the DB id was merged away or deleted.

        Evict the in-RAM ring and notify the caller so the outer
        tracker-id -> db-id cache is cleared too; the next frame then re-resolves
        to the live merge-target id via FAISS and rehydrates fresh."""
        logger.info(
            f"face3d: person {person_db_id} gone (merged/deleted), evicting")
        if on_person_gone is not None:
            try:
                on_person_gone(person_db_id)
            except Exception as e:
                logger.warning(
                    f"face3d on_person_gone callback failed for {person_db_id}: {e}")
        self.evict(person_db_id)

    # ------------------------------------------------------------------
    # Rehydration
    # ------------------------------------------------------------------

    def rehydrate(self, person_db_id: str, db_client) -> None:
        """Load persisted frames into the RAM ring on first touch this session."""
        st = self._state(person_db_id)
        if st.rehydrated:
            return
        st.rehydrated = True  # set first so a failed GET does not retry every frame
        if db_client is None:
            return
        try:
            payload = db_client.get_face3d_frames(person_db_id)
        except Exception as e:
            logger.debug(f"face3d rehydrate GET failed for {person_db_id}: {e}")
            return
        if not payload:
            return
        try:
            schema = payload.get("schema_version")
            frames_b64 = payload.get("frames_b64")
            if not frames_b64:
                st.frame_count = int(payload.get("frame_count", 0))
                st.last_built_at_count = int(payload.get("last_built_at_count", 0))
                return
            blob = base64.b64decode(frames_b64)
            ver, records = unpack_frames(blob)
            if ver != SCHEMA_VERSION:
                logger.warning(
                    f"face3d blob schema {ver} != {SCHEMA_VERSION} for "
                    f"{person_db_id}; treating as empty")
                return
            st.records = records
            st.frame_count = int(payload.get("frame_count", len(records)))
            st.last_built_at_count = int(payload.get("last_built_at_count", 0))
            logger.info(
                f"Rehydrated 3D frames for person {person_db_id}: "
                f"{len(records)} retained, frame_count={st.frame_count}, "
                f"last_built_at_count={st.last_built_at_count}")
        except Exception as e:
            logger.warning(
                f"face3d rehydrate decode failed for {person_db_id}: {e}; "
                f"treating as empty")

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def _evict_one(self, st: _PersonState) -> None:
        """Evict a single record from the most over-represented yaw bin.

        Within the chosen bin, drop the LOWEST-quality record so the best frames
        survive. NaN-yaw records are treated as one bucket and evicted only when
        they are the most over-represented bin (effectively last among valid
        spread, but still subject to the same balancing if dominant)."""
        bins: Dict[int, List[int]] = {}
        for idx, r in enumerate(st.records):
            bins.setdefault(_yaw_bin(r.yaw, self.yaw_bin_width), []).append(idx)

        # Prefer evicting from the most populated VALID bin so diversity is kept.
        valid = {b: idxs for b, idxs in bins.items() if b != _NAN_BIN}
        pool = valid if valid else bins
        target_bin = max(pool, key=lambda b: len(pool[b]))
        idxs = pool[target_bin]
        worst = min(idxs, key=lambda i: st.records[i].quality)
        del st.records[worst]

    # ------------------------------------------------------------------
    # Add
    # ------------------------------------------------------------------

    def add(self, person_db_id: str, embedding: np.ndarray,
            yaw: Optional[float], pitch: Optional[float], roll: Optional[float],
            quality: float, camera_id: str, params: np.ndarray) -> None:
        """Append a record, increment the lifetime frame_count, enforce max_frames."""
        st = self._state(person_db_id)
        rec = FrameRecord(
            yaw=float(yaw) if yaw is not None else float("nan"),
            pitch=float(pitch) if pitch is not None else float("nan"),
            roll=float(roll) if roll is not None else float("nan"),
            quality=float(quality),
            timestamp=int(__import__("time").time()),
            camera_id=str(camera_id),
            embedding=np.asarray(embedding, dtype=np.float16),
            params=np.asarray(params, dtype=np.float16),
        )
        st.records.append(rec)
        st.frame_count += 1
        while len(st.records) > self.max_frames:
            self._evict_one(st)
        st.adds_since_persist += 1

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _select_params(self, st: _PersonState) -> List[np.ndarray]:
        """Choose up to max_frames params, round-robin across populated yaw bins,
        best-quality first within each bin. Single bin -> just best quality."""
        bins: Dict[int, List[FrameRecord]] = {}
        for r in st.records:
            bins.setdefault(_yaw_bin(r.yaw, self.yaw_bin_width), []).append(r)
        for b in bins:
            bins[b].sort(key=lambda r: r.quality, reverse=True)

        ordered_bins = sorted(bins.keys())  # deterministic; _NAN_BIN sorts last
        selected: List[np.ndarray] = []
        cursors = {b: 0 for b in ordered_bins}
        remaining = sum(len(v) for v in bins.values())
        while len(selected) < self.max_frames and remaining > 0:
            for b in ordered_bins:
                c = cursors[b]
                if c < len(bins[b]):
                    selected.append(np.asarray(bins[b][c].params, dtype=np.float32))
                    cursors[b] = c + 1
                    remaining -= 1
                    if len(selected) >= self.max_frames:
                        break
        return selected

    def _populated_bins(self, st: _PersonState) -> int:
        return len({_yaw_bin(r.yaw, self.yaw_bin_width)
                    for r in st.records if not np.isnan(r.yaw)})

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def maybe_build(self, person_db_id: str, reconstructor, db_client,
                    on_person_gone: Optional[Callable[[str], None]] = None) -> bool:
        """Best-effort build on the fixed cadence. Returns True on a successful build.

        Triggers when frame_count >= min_samples and
        frame_count - last_built_at_count >= refine_interval. Never aborts for lack
        of yaw-bin diversity. On reconstruct failure, leaves last_built_at_count
        unchanged so the next cadence step retries."""
        st = self._state(person_db_id)
        if st.frame_count < self.min_samples:
            return False
        if st.frame_count - st.last_built_at_count < self.refine_interval:
            return False
        if reconstructor is None or not st.records:
            return False

        selected = self._select_params(st)
        if not selected:
            return False

        try:
            result_3d = reconstructor.reconstruct_from_params_list(selected)
        except Exception as e:
            logger.error(f"3D reconstruction failed for person {person_db_id}: {e}")
            return False
        if not result_3d:
            logger.warning(
                f"3D reconstruction produced no result for person {person_db_id}")
            return False

        try:
            glb_bytes = reconstructor.to_glb_bytes(
                result_3d["vertices"], result_3d["faces"],
                vertex_colors=result_3d.get("vertex_colors"))
            db_client.update_person_face3d(person_db_id, glb_bytes)
        except PersonNotFoundError:
            self._handle_person_gone(person_db_id, on_person_gone)
            return False
        except Exception as e:
            logger.error(
                f"3D face upload failed for person {person_db_id}: {e}")
            return False

        bins = self._populated_bins(st)
        logger.info(
            f"3D face built/refined for person {person_db_id} "
            f"({len(selected)} frames, {bins} bins) from {st.frame_count} accumulated")
        st.last_built_at_count = st.frame_count
        self.persist(person_db_id, db_client, force=True, on_person_gone=on_person_gone)
        return True

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------

    def persist(self, person_db_id: str, db_client, force: bool = False,
                on_person_gone: Optional[Callable[[str], None]] = None) -> None:
        """PUT the packed blob, debounced to every persist_every adds (or forced)."""
        st = self._state(person_db_id)
        if db_client is None:
            return
        if not force and st.adds_since_persist < self.persist_every:
            return
        try:
            blob = pack_frames(st.records)
            frames_b64 = base64.b64encode(blob).decode("utf-8")
            db_client.put_face3d_frames(
                person_db_id, st.frame_count, st.last_built_at_count,
                SCHEMA_VERSION, frames_b64)
            st.adds_since_persist = 0
        except PersonNotFoundError:
            self._handle_person_gone(person_db_id, on_person_gone)
        except Exception as e:
            logger.warning(
                f"face3d persist PUT failed for {person_db_id}: {e}")

    # ------------------------------------------------------------------
    # Shared hook used by the ingestion paths
    # ------------------------------------------------------------------

    def accumulate(self, person_db_id: str, frame, bbox, kps,
                   embedding, pose, quality: float, camera_id: str,
                   reconstructor, db_client,
                   on_person_gone: Optional[Callable[[str], None]] = None) -> None:
        """One-call hook: infer params, add a record, maybe build, persist (debounced).

        Skips the frame entirely (no record, no count) when infer_params returns None.
        Never raises into the caller's frame loop."""
        try:
            if reconstructor is None:
                return
            params = reconstructor.infer_params(frame, bbox, kps=kps)
            if params is None:
                return
            self.rehydrate(person_db_id, db_client)
            yaw = pose.yaw if pose is not None else None
            pitch = pose.pitch if pose is not None else None
            roll = pose.roll if pose is not None else None
            self.add(person_db_id, embedding, yaw, pitch, roll,
                     float(quality), camera_id, params)
            built = self.maybe_build(person_db_id, reconstructor, db_client,
                                     on_person_gone=on_person_gone)
            if not built:
                self.persist(person_db_id, db_client, on_person_gone=on_person_gone)
        except Exception as e:
            logger.error(
                f"face3d accumulate failed for person {person_db_id}: {e}")
