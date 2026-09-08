# Decord (modernized fork)

English | [简体中文](README.md)

![CI Build](https://github.com/chr431/decord/workflows/C/C++%20CI/badge.svg?branch=dev)

![symbol](docs/symbol.png)

`Decord` is a reverse procedure of `Record`. It provides convenient video slicing
methods based on a thin wrapper on top of hardware accelerated video decoders
(FFmpeg / NVIDIA Video Codec / NVDEC), treating video as a random-access sequence
of frames — built for deep learning data pipelines where video shuffling is slow
and awkward. It can also decode audio and slice it in sync with video.

This repository is a **modernized fork** of
[dmlc/decord](https://github.com/dmlc/decord). It stays fully API-compatible with
upstream 0.6.0 (verified file-by-file: zero missing public APIs; all differences
are backward-compatible extensions) while renovating the internals:

- **FFmpeg 9.0 only** (enforced at compile time, `avcodec 63`). FFmpeg 7.x/8.x
  support was removed: cross-version keyframe-index and seek-landing behaviour
  caused subtle correctness bugs.
- **CUDA 13 + NVDEC** build pipeline refreshed; covered on Windows / Linux CI.
- **Hybrid decoding (experimental)**: a single demux stream is split at keyframe
  boundaries between the CPU software decoder and NVDEC, decoded in parallel and
  merged in presentation order — see [Hybrid decoding](#hybrid-decoding-experimental).
- **Throughput & correctness work** (see [Major changes in this fork](#major-changes-in-this-fork)):
  consumer batch-assembly unlock, self-healing seek landing, hybrid scheduling
  pacing, NV12 direct output, on-disk keyframe index cache, and more.

> ⚠️ **`hybrid` / `hybrid_gpu` are EXPERIMENTAL**: scheduling, performance and
> memory bounds may change between releases without notice. Prefer `cpu()` /
> `gpu()` for production use.

## Table of contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Seek semantics: seek() vs seek_accurate()](#seek-semantics-seek-vs-seek_accurate)
- [Hybrid decoding (experimental)](#hybrid-decoding-experimental)
- [Performance reference](#performance-reference)
- [API overview](#api-overview)
- [Environment variables](#environment-variables)
- [Major changes in this fork](#major-changes-in-this-fork)
- [Testing](#testing)
- [Release process (maintainers)](#release-process-maintainers)
- [Acknowledgements & license](#acknowledgements--license)

## Installation

### Hard requirements

- **FFmpeg 9.0** (shared build, `avcodec-63`). The
  [BtbN n9.0 win64 gpl-shared](https://github.com/BtbN/FFmpeg-Builds/releases)
  builds are recommended; FFmpeg 7.x/8.x fail to compile by design.
- The GPU (NVDEC) build additionally needs the **CUDA Toolkit (13.x verified) +
  NVIDIA Video Codec SDK**, and a driver providing `nvcuvid`.

### Prebuilt package (Windows, recommended)

Download `decord-<version>-win64-gpu.zip` from
[GitHub Releases](https://github.com/chr431/decord/releases). Contents:

```
_decord_build/
├── decord.dll            # CPU+NVDEC build
├── avcodec-63.dll etc.   # FFmpeg 9.0 runtime
├── ffprobe.exe
└── python/decord/        # Python bindings
```

Usage: put the `python/decord` directory on `PYTHONPATH` (or copy it into your
project), and keep `decord.dll` plus the FFmpeg DLLs in the same directory or on
`PATH`. This fork does **not** depend on the PyPI `decord` package (that is
upstream 0.6.0, CPU-only).

### pip install from source

The build is driven by `pyproject.toml` (scikit-build-core): `pip install`
compiles the shared library with CMake and bundles it into the wheel.

```bash
git clone --recursive https://github.com/chr431/decord
cd decord

# CPU-only
pip install . --config-settings='cmake.args=-DUSE_CUDA=0'

# GPU (NVDEC): needs CUDA Toolkit + Video Codec SDK
pip install . --config-settings="cmake.args=-DUSE_CUDA=ON;-DFFMPEG_DIR=D:/path/to/ffmpeg-9.0"
```

On Linux, if the NVDEC build cannot find `libnvcuvid.so` (see upstream
[#102](https://github.com/dmlc/decord/issues/102)), locate it with
`ldconfig -p | grep libnvcuvid` and link it into `CUDA_TOOLKIT_ROOT_DIR/lib64`.

At runtime the FFmpeg shared libraries (`avcodec-63.dll` etc.) must be on
`PATH` or sit next to `decord.dll`.

### Development build (CMake + PYTHONPATH)

```bash
# Windows (CUDA enabled; drop -DUSE_CUDA or pass 0 for CPU-only)
cmake -S . -B build -DUSE_CUDA="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3" \
      -DFFMPEG_DIR="D:/path/to/ffmpeg-n9.0-win64-gpl-shared-9.0" -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

# Linux / macOS
cmake -S . -B build -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release -DFFMPEG_DIR=/path/to/ffmpeg
cmake --build build -j$(nproc)

# Run the Python bindings against the freshly built library (no pip install)
PYTHONPATH=python python -c "import decord; print(decord.__version__, decord.__ffmpeg_version__)"
```

Useful CMake options: `-DUSE_CUDA=ON|OFF|<cuda-root>` (NVDEC; the path must use
forward slashes on Windows), `-DFFMPEG_DIR=...`,
`-DDECORD_INSTALL_LIBDIR=...`. The Python bindings locate the library in
`build/` (`build/Release` on Windows) first, then `DECORD_LIBRARY_PATH`.

## Quick start

### VideoReader

```python
import decord
from decord import VideoReader, cpu, gpu

print(decord.__version__, decord.__ffmpeg_version__)   # 0.7.14 9.0.x

vr = VideoReader('examples/flipping_a_pancake.mkv', ctx=cpu(0))
# a file-like object works as well (in-memory decoding)
with open('examples/flipping_a_pancake.mkv', 'rb') as f:
    vr = VideoReader(f, ctx=cpu(0))

print('video frames:', len(vr))

# sequential frame access (seek/skip handled internally, efficiently)
for i in range(len(vr)):
    frame = vr[i]          # decord.ndarray.NDArray, HxWx3
    ...

# batch access: the efficient entry point for random / shuffled indices
frames = vr.get_batch([1, 3, 5, 7, 9])
print(frames.shape)        # (5, 240, 320, 3)
# duplicate indices are accepted and de-duplicated internally
frames2 = vr.get_batch([1, 2, 3, 2, 3, 4, 3, 4, 5]).asnumpy()

# arithmetic-progression sampling (frame-skip) takes a streaming fast path:
frames3 = vr.get_batch(range(0, 600, 3))

# streaming delivery: a background thread prefetches the next batch while
# you consume the current one (cross-batch order = completion order)
for start_pos, batch in vr.get_batch_stream(range(0, len(vr)), batch=250):
    ...  # within a batch frames follow `indices`; reorder across batches by
         # start_pos if your consumer is order-sensitive
```

### Output formats and ROI

```python
# output_format: 'rgb' (HxWx3) | 'gray' (HxW) | 'yuv420' (H*3/2 x W semi-planar NV12)
vr = VideoReader('video.mp4', ctx=cpu(0), output_format='yuv420')

# ROI-first: once fixed, the decoder only ever outputs that rectangle (the CPU
# crops before colour conversion; the GPU converts only the ROI window). Must
# be set before any frame is read.
vr = VideoReader('video.mp4', ctx=gpu(0), roi=(100, 60, 740, 660))
frame = vr.next()          # already the ROI frame
```

### Metadata probing (no decoder opened)

```python
from decord import probe

info = probe('video.mp4')
# {'duration_s':…, 'bit_rate':…, 'nb_frames':…, 'video_codec':'h264',
#  'width':1920, 'height':1080, 'pix_fmt':'yuv420p', 'avg_fps':…,
#  'audio_streams':1, 'subtitle_streams':0}
# A minimal ffprobe equivalent: milliseconds, no decoder, no frame buffers.
```

### Seek semantics: seek() vs seek_accurate()

```python
vr.seek(pos)            # fast seek: only guaranteed at KEYFRAME granularity
                        # (may land before or after pos) — documented,
                        # intentionally inaccurate
vr.seek_accurate(pos)   # accurate seek: self-healing landing verification;
                        # the next frame == sequential frame `pos`
vr.skip_frames(100)     # sequential skip
```

For frame-exact random access use `seek_accurate()` or `get_batch([pos])` (both
are frame-exact). `seek()` is only for "rewind / approximate position" cases.

### AudioReader / AVReader / VideoLoader

Audio, synced AV slicing and the training loader behave as upstream:

```python
from decord import AudioReader, AVReader, VideoLoader

ar = AudioReader('example.mp3', ctx=cpu(0), sample_rate=44100, mono=False)
print(ar[0:5])

av = AVReader('example.mov', ctx=cpu(0))
audio, video = av[0:20]      # each video frame maps to a chunk of samples

vl = VideoLoader(['1.mp4', '2.avi', '3.mpeg'], ctx=[cpu(0)],
                 shape=(2, 320, 240, 3), interval=1, skip=5, shuffle=1)
for batch in vl:
    ...
```

Shuffle modes are unchanged from upstream: `0` fully sequential; `1` shuffled
file order (sequential within each file, most efficient); `2` fully random;
`3` random frame access within each file.

### Deep learning framework bridges

```python
import decord
decord.bridge.set_bridge('torch')    # 'mxnet' | 'torch' | 'tensorflow' | 'native'
# from now on vr[0] / get_batch(...) return tensors of the target framework
```

## Hybrid decoding (experimental)

`decord.hybrid(dev_id)` (output in host memory) and `decord.hybrid_gpu(dev_id)`
(frames resident in VRAM) split **a single demux stream into chunks at keyframe
boundaries**, assign them to the CPU software decoder (FFmpeg, frame-parallel)
and NVDEC (CUVID) proportionally to their measured production rates
(water-filling), decode both sides concurrently and emit frames in presentation
order:

- The scheduler learns both production rates online (CPU side: post-filter
  effective output; GPU side: landing-rate EWMA) and allocates chunks
  proportionally; sticky block alternation avoids frequent decoder starvation.
- Memory/VRAM are governed by adaptive hard budgets (30% of free amounts by
  default, overridable via environment variables); all queue depths derive from
  the budgets to prevent OOM.
- Choosing hybrid *is* the explicit request for mixed CPU+GPU decoding: there is
  **no automatic slow-side fallback**, even when mixing is slower than pure
  NVDEC on some streams (user choice is respected; no silent substitution).

```python
import warnings
from decord import VideoReader, hybrid, hybrid_gpu

with warnings.catch_warnings():
    warnings.simplefilter('ignore')          # silence the experimental warning
    vr = VideoReader('video.mp4', ctx=hybrid(0), output_format='yuv420')

vr.seek_accurate(0)
batch = vr.get_batch(range(0, 1000))         # same API as any other reader
```

**When it helps**: streams where software decoding and NVDEC are close in speed
(typically 1080p H.264) benefit most; for streams where software decoding is far
slower than NVDEC (HEVC/AV1), mixing usually lands in between — faster than pure
CPU, slower than pure NVDEC. Measure on your own hardware (see
[Performance reference](#performance-reference)).

## Performance reference

Measured on one machine (AMD Ryzen 9 7945HX + RTX 4060 Laptop, Windows, FFmpeg
9.0 shared, 1080p, `output_format='yuv420'`, sequential `get_batch` of 250
frames, median of 5-10 runs, fps). Results vary greatly across machines/
streams/resolutions/**power plans** — measure your own workload; hybrid is
experimental, numbers are order-of-magnitude guidance.

| Decoder | H.264 | HEVC | AV1 |
|---|---|---|---|
| `cpu` (16 decode threads) | 1233 | 961 | 709 |
| `gpu` (NVDEC) | 958 | 1868 | 1285 |
| `hybrid` (experimental, → host memory) | **1756** | **1938** | 1724 |
| `hybrid_gpu` (experimental, → VRAM) | **1645** | **1979** | **1796** |

Notes:

- All six configurations (3 codecs × both modes) beat pure CPU AND pure NVDEC
  simultaneously (av1 CPU-out 1.34x, hevc 1.04x/1.06x, h264 1.42x/1.72x).
- **The Windows power plan strongly affects hybrid stability**: under mixed
  CPU+GPU load, boost-governor behaviour can produce ±15-25% run-to-run
  variance ("bimodal" throughput) on the same binary. For reproducible
  benchmarks, pin the power plan (high performance / ultimate performance)
  before measuring.
- `get_batch` throughput includes every pipeline (consumer batch assembly:
  batch buffer pool + copy/decode overlap); the pure-CPU path benefits
  equally.
- `probe()` inspects metadata in milliseconds without opening a decoder — handy
  for asset triage before scheduling.

## API overview

`VideoReader` (full docs in the `python/decord/video_reader.py` docstrings):

| Member | Description |
|---|---|
| `len(vr)` / `get_key_indices()` | frame count / keyframe index list |
| `vr[i]` / `vr.next()` | single-frame access (NDArray) |
| `vr.get_batch(indices, roi=None)` | batch access; shuffled, duplicate and arithmetic-progression indices supported |
| `vr.get_batch_stream(indices, batch=250)` | streaming delivery: background prefetch, yields `(start_pos, batch)` in completion order |
| `vr.seek(pos)` / `vr.seek_accurate(pos)` | fast (keyframe-level) / exact (frame-level) seek |
| `vr.skip_frames(n)` | sequential skip |
| `vr.next_roi(x1,y1,x2,y2)` / `get_batch(..., roi=…)` | ROI access (ROI-first decoders output only that rectangle) |
| `vr.get_frame_timestamp(idx)` / `get_avg_fps()` | timestamps / average fps |
| `vr.get_codec()` / `get_color_range()` | codec name / stream colour range |
| `probe(uri)` (module-level) | container/stream metadata without a decoder (dict) |
| `get_ffmpeg_version()` / `decord.__ffmpeg_version__` | FFmpeg version actually linked into the loaded native library |

Module-level: `cpu(id)`, `gpu(id)`, `hybrid(id)`, `hybrid_gpu(id)`, `probe`,
`bridge.set_bridge(...)`, `VideoLoader`, `AudioReader`, `AVReader`.

## Environment variables

Commonly used (all optional; adaptive defaults apply when unset):

| Variable | Default | Description |
|---|---|---|
| `DECORD_LIBRARY_PATH` | — | extra directory for the Python bindings to locate `decord.dll`/`libdecord.so` |
| `DECORD_FFMPEG_THREAD_COUNT` | physical cores (clamp 2–16) | CPU software decode threads (NVDEC unaffected) |
| `DECORD_FILTER_THREADS` | 1 | CPU-side filter threads |
| `DECORD_PREFETCH_DEPTH` | 8 | demux lead depth for pure CPU/GPU pipelines |
| `DECORD_CPU_FRAME_QUEUE_SIZE` | adaptive | CPU decode output queue depth (memory bound) |
| `DECORD_CONVERT_WORKERS` | auto (2–8 for RGB) | RGB conversion fan-out pool (`0/1` disables) |
| `DECORD_BATCH_BUF_POOL` / `DECORD_BATCH_BUF_POOL_MAX` | 1 / 2 | get_batch batch-buffer pool switch and retained blocks |
| `DECORD_BATCH_COPY_WORKERS` | auto 2–4 | batch copy worker threads (`0` restores serial copies) |
| `DECORD_DISABLE_INDEX_CACHE` | unset | set to disable the on-disk keyframe index cache (system cache dir by default) |
| `DECORD_HYBRID_VRAM_BUDGET_MB` / `DECORD_HYBRID_RAM_BUDGET_MB` | 30% of free | hybrid VRAM/RAM hard budgets (override auto) |
| `DECORD_PREFETCH_DEPTH_HYBRID` | 384 | hybrid demux lead baseline (effective value = max(this, decoder suggestion)) |
| `DECORD_EOF_RETRY_MAX` / `DECORD_REWIND_RETRY_MAX` | 10240 / 16 | EOF/rewind retry bounds for corrupted-stream tolerance |

Diagnostics (not needed in normal use):

| Variable | Description |
|---|---|
| `DECORD_HYBRID_DEBUG` | hybrid scheduling/budget/emission telemetry (stderr) |
| `DECORD_SEEK_DEBUG` | seek landing trace |
| `DECORD_CPU_RATE_DEBUG` | CPU production-rate segment prints |
| `DECORD_HYBRID_FORCE_SIDE=cpu\|gpu` | force all hybrid chunks onto one side (experiments) |
| `DECORD_HYBRID_PREFETCH` | direct override of the hybrid demux lead depth |

## Major changes in this fork

Main engineering items relative to upstream 0.6.0 (besides dependency
renovation), by theme:

**Throughput**

- get_batch consumer unlock: batch buffer pool (avoids the first-touch page
  fault tax of ~0.8 GB allocations per batch) + per-frame copies dispatched to
  worker threads overlapping decode; arithmetic-progression sampling takes a
  single-pass streaming fast path.
- CPU pipeline: NV12 direct output (filter `format=nv12` + two memcpys)
  replacing the scalar U/V interleave loop; decode threads scale with physical
  cores by default (clamp 2–16).
- NVDEC: per-frame CUDA event sync instead of full-stream sync; async D2H ring;
  batched H2D upload (one sync per batch).
- Hybrid: water-filling scheduling on measured rates; demux decision pacing
  decoupled from rate learning; GPU-lead hard cap preventing dav1d tail-frame
  deadlocks; batched H2D upload.

**Correctness**

- Seek landing verification rewritten as a bounded pts-anchored self-healing
  loop (deliver on hit / advance on undershoot / re-anchor to keyframe on
  overshoot).
- FFmpeg 9-only compile-time gate, eliminating an entire class of cross-version
  index/landing behaviour differences.
- Concurrency fixes around duplicate indices, batch buffer reuse and hybrid
  chunk boundaries (guarded by a byte-exact lockstep suite).

**Engineering**

- On-disk keyframe index cache (invalidated by path hash + size + mtime,
  stored in the system cache directory).
- `probe()` / `get_ffmpeg_version()` C APIs and Python bindings.
- ROI-first decoding pipeline (CPU/GPU uniformly output a fixed rectangle).
- Tests: `tests/run_fast.sh` runs four suites in parallel (md5 / streaming /
  stride / byte-exact lockstep).

See the commit log for the full history; both `dev` and `master` are watched by
CI.

## Testing

```bash
# fast regression (~3 min; point DECORD_LIBRARY_PATH at the build output)
DECORD_LIBRARY_PATH=<dll dir> bash tests/run_fast.sh [frame count]

# individual suites
python tests/test_hybrid.py --n 600          # md5 + seek
python tests/test_hybrid_lockstep.py 600     # byte-exact interleaved hybrid check
```

## Release process (maintainers)

The version source of truth is `__version__` in
`python/decord/_ffi/libinfo.py` (`pyproject.toml` kept in sync; update both via
`python tools/update_version.py`). Releases run from GitHub Actions →
**Release** → Run workflow: provide the version (e.g. `0.7.14`) and ref
(default `master`); the workflow bumps the version → tags `vX.Y.Z` → builds
with CUDA + FFmpeg 9.0 → packages `decord-<ver>-win64-gpu.zip` → creates the
Release. A failed build produces no commit/tag/release. Tag pushes do not
trigger PyPI publishing (removed).

## Acknowledgements & license

- Upstream [dmlc/decord](https://github.com/dmlc/decord) — all of the base
  architecture in this fork (FFI, readers/loaders, bridges) comes from upstream,
  under **Apache License 2.0** (see [LICENSE](LICENSE)).
- Video decoding depends on [FFmpeg](https://ffmpeg.org) (dynamically linked).
  This repository is Apache-2.0; FFmpeg's own license (LGPL-2.1+, or GPL when
  built with GPL components) applies to the binaries you choose to distribute —
  comply accordingly.
- NVIDIA Video Codec SDK / NVDEC are subject to NVIDIA's respective licenses.
