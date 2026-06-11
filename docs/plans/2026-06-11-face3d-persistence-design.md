# Persistent, traceable 3D-frame accumulation + best-effort build

Date: 2026-06-11
Repos touched: reid-worker, reid-db-handler, repository-deployment-catalog

## Problem

The 3D face model never builds in production. Two compounding reasons:

1. **Wrong ingestion path.** All 3D accumulation (head-pose, `face3d_params`,
   the build gate) lives only inside `process_stream`'s frame loop
   (`main.py:1003-1060`), reachable only via `POST /api/streams/<id>/frame`.
   Production traffic arrives on `POST /api/identify`, which extracts a single
   embedding, writes one trait + sighting, and returns -- it never touches pose
   or `face3d_params` and never evaluates the build gate. So no model can ever
   build, regardless of frame count.
2. **Over-strict gate.** Even on the stream path the first build requires
   `emb_count >= 50` AND `angle_bins >= 3` (3 distinct 30-deg yaw bins). Frontal-
   only capture never satisfies the angle-bin condition.

Additionally, the accumulator is RAM-only (`tracker.persons[id]['face3d_params']`,
`main.py:79`), so any pod restart wipes progress and a person met weeks later
starts from zero.

## Goals

- Accumulate per-frame data on EVERY ingestion path (identify + stream).
- Persist per-frame data (embedding + pose + 257-d 3DDFA params) durably, keyed to
  the persistent person identity, so progress survives restarts and weeks-long gaps.
- Best-effort build: build at 50 frames, rebuild every +50 (50/100/150/200/250/300),
  hard ceiling 300. Prefer a yaw-bin-diverse, high-quality subset; if diversity is
  not available, build with whatever frames exist. NEVER block on angle bins.
- Everything traceable: each frame's embedding, pose, and params individually
  recoverable on demand.
- Respect the architecture rule: reid-worker NEVER touches Mongo directly; all
  persistence goes through the reid-db-handler HTTP API.

## Non-goals

- Fixing the `/api/identify` `db.get_person` 404 bug (`main.py:2000`) -- tracked
  separately.
- Fixing the missing CUDA (`libcudnn.so.9`) -- runs CPU, acceptable.
- Body/gait 3D -- face only (3DDFA/deep3d) exists.

## Storage (reid-db-handler)

New dedicated collection, ONE compressed document PER PERSON (not per frame):

```
Face3DFrames {
  _id,
  person_id: String (indexed, unique),   // = Person._id (the DB id, not external_id)
  frame_count: Number,                    // total frames ever accumulated (drives cadence)
  last_built_at_count: Number,            // frame_count at last successful build (0 = never)
  frames: Buffer,                         // zlib-compressed packed per-frame records
  schema_version: Number,                 // packing format version, start at 1
  updated_at: Date
}
```

### Per-frame record packing (inside the `frames` blob)

Packed little-endian, fixed-width, then zlib-compressed. Per record:

| field      | type     | bytes |
|------------|----------|-------|
| yaw        | float16  | 2 (NaN if pose failed) |
| pitch      | float16  | 2 |
| roll       | float16  | 2 |
| quality    | float16  | 2 |
| timestamp  | uint32   | 4 (epoch seconds) |
| camera_id  | uint8 len + utf8 | 1+N (short tag, e.g. "identify_api", "stream:cam1") |
| embedding  | float16 x 512 | 1024 |
| params     | float16 x 257 | 514 |

~1.5 KB raw per record (float16 halves vs float32). At the 300 ceiling ~450 KB raw,
zlib-compressed to roughly ~250-350 KB per person. float16 is loss-tolerant for both
ArcFace embeddings (cosine sim unaffected at 2-3 decimal places) and 3DDFA coeffs
(downstream averaging dominates the error budget). `schema_version` lets us change the
packing later.

Header inside the blob (before records): `magic(2) + schema_version(1) + record_count(2)`.

### Routes (reid-db-handler, src/routes/persons.js or new face3d.js)

- `GET  /api/persons/:id/face3d-frames`
  Returns `{ person_id, frame_count, last_built_at_count, schema_version, frames_b64 }`
  where `frames_b64` is base64 of the raw compressed Buffer. Worker decompresses.
  404 if none yet.
- `PUT  /api/persons/:id/face3d-frames`
  Body `{ frame_count, last_built_at_count, schema_version, frames_b64 }`. Upsert.
  Worker owns the blob; db-handler just stores it (no server-side packing logic).
- `GET  /api/persons/:id/face3d-frames/summary` (traceability/debug)
  Server-side decode of just the headers + per-bin yaw histogram + counts, WITHOUT
  shipping the heavy arrays: `{ frame_count, last_built_at_count, yaw_bins: {bin: n},
  null_pose_count, avg_quality }`. Cheap to call from analytics/CLI.

The heavy `frames` Buffer is never included in normal `GET /api/persons` list/detail
responses (like `face_model_3d` today), so it never bloats catalogue queries.

## Accumulator (reid-worker)

Extract one shared helper, e.g. `Face3DAccumulator`, that BOTH paths call:

```
accumulate_face3d(person_id, frame, bbox, kps) -> None
  - params = reconstructor.infer_params(frame, bbox, kps)   # 257-d, or None -> skip
  - pose   = estimate_head_pose(kps, w, h)                  # best-effort, None ok
  - append record {yaw,pitch,roll,quality,ts,camera,embedding,params} to in-RAM ring
  - lazily-rehydrated from db on first touch of this person in the session
  - persist: PUT the recompressed blob (debounced/batched; see below)
  - maybe_build(person_id)
```

Hook sites:
- `process_stream` loop: replace the inline block at `main.py:1003-1060` with a call.
- `identify_person` (`main.py:1986` area, after embedding+person resolution): add a
  call so identify-driven frames accumulate too.

### Rehydrate
On first encounter of a person in a session, `GET .../face3d-frames`, decompress into
the in-RAM ring, set `frame_count`/`last_built_at_count`. Identity itself already
persists via FAISS (`db.search`), so the same person resolves to the same `person_id`
across restarts and weeks; this step restores their frame history.

### Persist cadence
Persisting on literally every frame is a `PUT` per frame. Debounce: persist every N
frames (e.g. 10) and on a successful build and on stream teardown / person-disappear.
Worst case loss on hard crash = <10 frames, acceptable.

## Build logic (best-effort, every 50, ceiling 300)

`maybe_build(person_id)`:
- triggers when `frame_count >= 50` AND `frame_count - last_built_at_count >= 50`
  (i.e. crosses 50/100/150/.../300).
- selection: choose up to `max_frames` records, preferring high quality spread across
  yaw bins (round-robin across populated 30-deg bins, best-quality first within a bin).
  If only one bin is populated, just take the best-quality records. NEVER abort for
  lack of bins.
- reconstruct via existing `reconstruct_from_params_list(selected_params)`.
- on success: upload GLB via existing `update_person_face3d` ->
  `POST /api/persons/:id/face3d`; set `last_built_at_count = frame_count`; persist blob.
- on failure: log error, leave `last_built_at_count` unchanged (retry next +50).

### Ceiling 300 (bin-balanced eviction)
When `frame_count` would exceed 300 retained records, evict from the MOST over-
represented yaw bin first (keep highest-quality within each bin), so diversity is
preserved rather than blind FIFO. `frame_count` keeps incrementing as a lifetime
counter for cadence; the retained ring is capped at 300.

### Gate change
`min_angle_bins` is REMOVED as a hard gate. It survives only as the selection hint
(bin width). Delete the `angle_bins >= min` condition and the "samples but only X/3"
debug branch. Drop the in-RAM 500-cap (`main.py:1010`) -> use `max_frames` (300).

## Config (catalog configmap + config.ini.example)

`[face_3d]` section:
- `enabled = true`
- `min_samples = 50`        (first build threshold)
- `refine_interval = 50`    (rebuild step)
- `max_frames = 300`        (NEW: retained ring + ceiling)
- `yaw_bin_width = 30`       (selection hint only)
- REMOVE `min_angle_bins`    (no longer a gate)

Update `repository-deployment-catalog/apps/recon/reid-worker/configmap.yaml` to match,
and document the new keys in `config.ini.example`.

## Traceability summary

- Per-frame embedding + pose + params recoverable by `GET .../face3d-frames` +
  decompress (the "grab once needed" requirement).
- `GET .../face3d-frames/summary` gives counts + yaw-bin histogram cheaply.
- Build events logged at INFO: `"3D face built/refined for person {id}
  ({n} frames, {b} bins) from {total} accumulated"`.

## Error handling

- `infer_params` None -> skip that frame (no record), do not increment.
- pose None -> store record with NaN yaw; counts toward frame_count; excluded from
  bin diversity but usable for averaging.
- db-handler PUT/GET failure -> log, keep RAM state, retry on next debounce; never
  crash the frame loop.
- corrupt/old `schema_version` on GET -> log, treat as empty (rebuild fresh).

## Testing

- reid-db-handler: unit test the new routes (upsert + get + summary) with a small
  packed blob; assert round-trip.
- reid-worker: unit test pack/unpack (float16 round-trip within tolerance), bin-
  balanced eviction at 300, selection prefers diverse bins, build triggers exactly at
  50/100/.../300, best-effort build with a single bin.
- Integration (manual, on the pod after deploy): push >50 frames via identify, confirm
  `face_model_3d` becomes non-null and `face3d-frames/summary` shows the expected count.

## Rollout

Three commits, each via its repo's CI (no manual image builds, no kubectl):
1. reid-db-handler: model + routes.
2. reid-worker: accumulator + build changes + config.ini.example.
3. repository-deployment-catalog: configmap keys (after worker image ships).
Order: db-handler first (so the routes exist), then worker, then configmap.
