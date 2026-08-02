#!/usr/bin/env python
"""Comprehensive decord benchmark: CPU/GPU x sequential/random, with memory monitoring.

Each (context, mode, video) combination runs in its own subprocess so that
memory/GPU state from earlier combinations cannot leak into later ones
(benchmark-order artifacts).  Within a combination, memory is reported both
as the in-process peak and as the delta over the process baseline.

Usage:
    python bench_gpu.py <video_dir>                 # default subset + all contexts
    python bench_gpu.py <video_dir> --videos test.mp4,test3.mp4
    python bench_gpu.py <video_dir> --all-videos
    python bench_gpu.py --worker <spec-json>        # internal: run one combo
"""
import os, sys, time, argparse, random, json, subprocess

_FFMPEG_BIN = os.environ.get("DECORD_FFMPEG_BIN", "")
if _FFMPEG_BIN:
    os.add_dll_directory(_FFMPEG_BIN)

# Default video subset: one HEVC + two H.264 clips.  Override with --videos.
DEFAULT_SUBSET = ["test.mp4", "test3.mp4", "test5.mp4"]

from decord import VideoReader, cpu, gpu

# ---------------------------------------------------------------------------
# memory helpers
# ---------------------------------------------------------------------------
_psutil = None
_pynvml = None

def _get_psutil():
    global _psutil
    if _psutil is None:
        import psutil
        _psutil = psutil
    return _psutil

def _get_pynvml():
    global _pynvml
    if _pynvml is None:
        try:
            import pynvml
            pynvml.nvmlInit()
            _pynvml = pynvml
        except Exception:
            _pynvml = False
    return _pynvml

def get_ram_mb():
    """Resident set size in MB."""
    try:
        return _get_psutil().Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return -1.0

def get_vram_mb(device_id=0):
    """Used GPU memory in MB (nvidia-smi equivalent)."""
    try:
        nvml = _get_pynvml()
        if not nvml:
            return -1.0
        h = nvml.nvmlDeviceGetHandleByIndex(device_id)
        info = nvml.nvmlDeviceGetMemoryInfo(h)
        return info.used / (1024 * 1024)
    except Exception:
        return -1.0

# ---------------------------------------------------------------------------
# benchmark helpers
# ---------------------------------------------------------------------------
def bench_sequential(vr, warmup=10):
    """Read every frame in order, return (sec, n, fps)."""
    n_total = len(vr)
    for i in range(min(warmup, n_total)):
        _ = vr[i]
    t0 = time.perf_counter()
    for i in range(warmup, n_total):
        _ = vr[i]
    elapsed = time.perf_counter() - t0
    n = n_total - warmup
    return elapsed, n, n / elapsed if elapsed > 0 else 0.0

def bench_random(vr, indices):
    """Read frames at given indices via get_batch, return (sec, n, fps)."""
    t0 = time.perf_counter()
    for idx in indices:
        _ = vr[idx]
    elapsed = time.perf_counter() - t0
    n = len(indices)
    return elapsed, n, n / elapsed if elapsed > 0 else 0.0

def mem_peak_during(func):
    """Run func, sampling RAM every 50 ms; return (result, ram_samples, vram_samples)."""
    ram_samples = []
    vram_samples = []
    stop_flag = [False]
    result = [None]
    exc = [None]

    def _runner():
        try:
            result[0] = func()
        except Exception as e:
            exc[0] = e
        finally:
            stop_flag[0] = True

    import threading
    t = threading.Thread(target=_runner)
    t.start()
    while not stop_flag[0]:
        ram_samples.append(get_ram_mb())
        vram_samples.append(get_vram_mb())
        time.sleep(0.05)
    t.join()
    if exc[0]:
        raise exc[0]
    return result[0], ram_samples, vram_samples

# ---------------------------------------------------------------------------
# worker mode: run one (context, mode, video) combo in an isolated process
# ---------------------------------------------------------------------------
def worker(spec):
    ctx_name = spec["ctx"]
    mode_name = spec["mode"]
    path = spec["path"]
    warmup = spec["warmup"]
    random_samples = spec["random_samples"]
    rounds = spec["rounds"]

    ctx = cpu(0) if ctx_name == "CPU" else gpu(0)
    rng = random.Random(42)
    probe = VideoReader(path, ctx=ctx)
    nframes = len(probe)
    random_indices = sorted([rng.randint(0, nframes - 1) for _ in range(random_samples)])
    del probe

    ram_start = get_ram_mb()
    vram_start = get_vram_mb()

    best = None
    for r in range(rounds):
        vr = VideoReader(path, ctx=ctx)
        if mode_name == "random":
            def _run():
                return bench_random(vr, random_indices)
        else:
            def _run():
                return bench_sequential(vr, warmup)
        (elapsed, n, fps), rams, vrams = mem_peak_during(_run)
        del vr
        if best is None or elapsed < best[0]:
            best = (elapsed, n, fps, rams, vrams)

    elapsed, n, fps, rams, vrams = best
    ram_peak = max(rams) if rams else -1.0
    ram_delta = ram_peak - ram_start
    vram_peak = max(vrams) if vrams and vrams[0] > 0 else -1.0

    result = {
        "video": os.path.basename(path),
        "ctx": ctx_name,
        "mode": mode_name,
        "nframes": n,
        "elapsed_s": round(elapsed, 3),
        "fps": round(fps, 1),
        "ram_peak_mb": round(ram_peak, 1),
        "ram_delta_mb": round(ram_delta, 1),
        "vram_peak_mb": round(vram_peak, 1) if ctx_name == "GPU" else None,
    }
    print(json.dumps(result))
    return 0

# ---------------------------------------------------------------------------
# driver mode: spawn an isolated worker per combination
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", default=True)
    parser.add_argument("--gpu", action="store_true", default=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--random-samples", type=int, default=200,
                        help="How many random frames to access in random test")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--videos", type=str, default=None,
                        help="Comma-separated video names to benchmark (default: "
                             "HEVC + 2x H.264 subset)")
    parser.add_argument("--all-videos", action="store_true",
                        help="Benchmark every video in the directory")
    parser.add_argument("video_dir", nargs="?", default="test_video_long")
    parser.add_argument("--worker", type=str, default=None,
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        return worker(json.loads(args.worker))

    video_dir = args.video_dir
    available = sorted([f for f in os.listdir(video_dir)
                        if f.endswith((".mp4", ".mkv", ".mov", ".avi"))])
    if not available:
        print(f"No videos found in {video_dir}")
        sys.exit(1)

    if args.all_videos:
        videos = available
    elif args.videos:
        wanted = [v.strip() for v in args.videos.split(",")]
        missing = [v for v in wanted if v not in available]
        if missing:
            print(f"Unknown videos: {missing}")
            sys.exit(1)
        videos = wanted
    else:
        videos = [v for v in DEFAULT_SUBSET if v in available]
        if not videos:
            videos = available

    ctxs = []
    if args.cpu:
        ctxs.append("CPU")
    if args.gpu:
        # verify GPU availability once in this process
        try:
            gpu(0)
            ctxs.append("GPU")
        except Exception as e:
            print(f"GPU not available: {e}")
            if not ctxs:
                ctxs.append("CPU")
    if not ctxs:
        ctxs.append("CPU")

    modes = ["seq", "random"]

    results = []
    sep = "=" * 80
    print(sep)
    print(f"DECORD BENCHMARK (subprocess-isolated) | {len(videos)} video(s) | "
          f"{len(ctxs)} ctx | {len(modes)} mode(s) | {args.rounds} round(s)")
    print(sep)

    for ctx_name in ctxs:
        for mode_name in modes:
            for vname in videos:
                spec = {
                    "ctx": ctx_name,
                    "mode": mode_name,
                    "path": os.path.join(video_dir, vname),
                    "warmup": args.warmup,
                    "random_samples": args.random_samples,
                    "rounds": args.rounds,
                }
                try:
                    out = subprocess.run(
                        [sys.executable, os.path.abspath(__file__),
                         "--worker", json.dumps(spec)],
                        capture_output=True, text=True, timeout=1800)
                except subprocess.TimeoutExpired:
                    print(f"  {vname:30s} [{ctx_name}/{mode_name}] TIMEOUT")
                    continue
                if out.returncode != 0:
                    tail = out.stderr.strip().splitlines()[-1] if out.stderr.strip() else "?"
                    print(f"  {vname:30s} [{ctx_name}/{mode_name}] ERROR: {tail[:120]}")
                    continue
                r = json.loads(out.stdout.strip().splitlines()[-1])
                results.append(r)
                mem_str = f"RAM_peak={r['ram_peak_mb']:7.1f}MB (delta {r['ram_delta_mb']:+.1f})"
                if r.get("vram_peak_mb"):
                    mem_str += f"  VRAM_peak={r['vram_peak_mb']:7.1f}MB"
                print(f"  {r['video']:30s} [{r['ctx']}/{r['mode']}] "
                      f"{r['nframes']:5d}f  {r['elapsed_s']:7.3f}s  {r['fps']:8.1f} fps  {mem_str}")

    # summary
    print(f"\n{sep}")
    print("SUMMARY (fps)")
    print(f"{'Video':30s}", end="")
    for ctx_name in ctxs:
        for mode_name in modes:
            print(f"  {ctx_name}/{mode_name:6s}", end="")
    print()
    for vname in videos:
        print(f"{vname:30s}", end="")
        for ctx_name in ctxs:
            for mode_name in modes:
                fps_val = -1
                for r in results:
                    if r["video"] == vname and r["ctx"] == ctx_name and r["mode"] == mode_name:
                        fps_val = r["fps"]
                print(f"  {fps_val:8.1f}", end="")
        print()
    print(sep)
    print("Done.")


if __name__ == "__main__":
    sys.exit(main())
