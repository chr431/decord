#!/usr/bin/env python
"""Comprehensive decord benchmark: CPU/GPU x sequential/random, with memory monitoring."""
import os, sys, time, argparse, random, json
import numpy as np

_FFMPEG_BIN = os.environ.get("DECORD_FFMPEG_BIN", "")
if _FFMPEG_BIN:
    os.add_dll_directory(_FFMPEG_BIN)

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
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", default=True)
    parser.add_argument("--gpu", action="store_true", default=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--random-samples", type=int, default=200,
                        help="How many random frames to access in random test")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("video_dir", nargs="?", default="test_video_long")
    args = parser.parse_args()

    video_dir = args.video_dir
    videos = sorted([f for f in os.listdir(video_dir)
                     if f.endswith((".mp4", ".mkv", ".mov", ".avi"))])
    if not videos:
        print(f"No videos found in {video_dir}")
        sys.exit(1)

    # Build test matrix
    ctxs = []
    if args.cpu:
        ctxs.append(("CPU", cpu(0)))
    if args.gpu:
        try:
            ctxs.append(("GPU", gpu(0)))
        except Exception as e:
            print(f"GPU not available: {e}")
            if not ctxs:
                ctxs.append(("CPU", cpu(0)))

    modes = [
        ("seq",    bench_sequential),
        ("random", bench_random),
    ]

    results = []
    sep = "=" * 80

    print(sep)
    print(f"DECORD BENCHMARK  |  {len(videos)} videos  |  {len(ctxs)} mode(s)  |  "
          f"{len(modes)} access pattern(s)  |  {args.rounds} round(s)")
    print(sep)

    for ctx_name, ctx in ctxs:
        is_gpu = (ctx_name == "GPU")
        ctx_results = {"context": ctx_name, "videos": {}}

        for mode_name, mode_fn in modes:
            use_random = (mode_name == "random")

            for vname in videos:
                path = os.path.join(video_dir, vname)
                try:
                    probe = VideoReader(path, ctx=ctx)
                    nframes = len(probe)
                except Exception as e:
                    print(f"  SKIP  {vname}  [{ctx_name}/{mode_name}]: {e}")
                    continue

                # prepare random indices once
                rng = random.Random(42)
                random_indices = sorted(
                    [rng.randint(0, nframes - 1) for _ in range(args.random_samples)]
                )

                best_elapsed = float("inf")
                best_ram = best_vram = None
                best_n = best_fps = 0

                for r in range(args.rounds):
                    vr = VideoReader(path, ctx=ctx)

                    if use_random:
                        def _run():
                            return bench_random(vr, random_indices)
                    else:
                        def _run():
                            return bench_sequential(vr, args.warmup)

                    (elapsed, n, fps), rams, vrams = mem_peak_during(_run)

                    if elapsed < best_elapsed:
                        best_elapsed = elapsed
                        best_n = n
                        best_fps = fps
                        best_ram = rams
                        best_vram = vrams

                # stats
                ram_peak = max(best_ram) if best_ram else -1
                ram_mean = sum(best_ram) / len(best_ram) if best_ram else -1
                vram_peak = max(best_vram) if best_vram else -1
                vram_mean = sum(best_vram) / len(best_vram) if best_vram and best_vram[0] > 0 else -1

                label = f"{vname:30s} [{ctx_name}/{mode_name}]"
                mem_str = f"RAM_peak={ram_peak:7.1f}MB"
                if is_gpu and vram_peak > 0:
                    mem_str += f"  VRAM_peak={vram_peak:7.1f}MB"

                print(f"  {label}  {best_n:5d}f  "
                      f"{best_elapsed:7.3f}s  {best_fps:8.1f} fps  {mem_str}")

                ctx_results["videos"].setdefault(vname, {})[mode_name] = {
                    "nframes": nframes,
                    "elapsed_s": round(best_elapsed, 3),
                    "fps": round(best_fps, 1),
                    "ram_peak_mb": round(ram_peak, 1),
                    "vram_peak_mb": round(vram_peak, 1) if is_gpu else None,
                }

        results.append(ctx_results)

    # summary
    print(f"\n{sep}")
    print("SUMMARY (fps)")
    print(f"{'Video':30s}", end="")
    for ctx_name, _ in ctxs:
        for mode_name, _ in modes:
            print(f"  {ctx_name}/{mode_name:6s}", end="")
    print()

    for vname in videos:
        print(f"{vname:30s}", end="")
        for ctx_name, _ in ctxs:
            for mode_name, _ in modes:
                fps_val = -1
                for r in results:
                    if r["context"] == ctx_name:
                        v = r["videos"].get(vname, {}).get(mode_name)
                        if v:
                            fps_val = v["fps"]
                if fps_val > 0:
                    print(f"  {fps_val:8.1f}", end="")
                else:
                    print(f"       N/A", end="")
        print()

    print(sep)
    print("Done.")

    return results


if __name__ == "__main__":
    main()
