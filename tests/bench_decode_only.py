"""纯解码基准（decode-only）：三编码 × 三后端，全帧顺序读。

**口径声明（重要）**：本脚本只测 **decord 解码本身**——不含引擎侧的
任何耦合（无 OCR、无分段、无消费线程）。README 的「性能参考」表必须用
本口径的数字；引擎侧 e2e 数字（含推理/装配）在 video_ocr_engine 仓，
两者不可混用。

协议（与 README 表一致）：
  · 1080p、`get_batch` 每批 250 帧顺序读、全帧输出（不设 ROI）；
  · 每臂 5 轮，取中位；每轮独立 VideoReader（冷实例）；
  · hybrid 的 CPU 臂线程数按 codec 分档（与引擎 `decode_caliber` 同源）。

用法：
  python tests/bench_decode_only.py                 # 三码 × 三后端
  python tests/bench_decode_only.py --rounds 7
  python tests/bench_decode_only.py --codec h264 --backend hybrid
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 默认视频（env 覆盖，便于换机复现）
VIDS = {
    "h264": ("test6_h264.mp4", 23970),
    "hevc": ("test6_hevc.mp4", 23970),
    "av1": ("test6.mp4", 23970),
}
# 引擎同源线程档（h264/hevc=32、av1=24；见 video_ocr_engine/config/decode_caliber.py）
THREADS = {"h264": 32, "hevc": 32, "av1": 24}


def _vdir() -> Path:
    return Path(os.environ.get("RACELOG_VIDEO_DIR", r"D:\Videos\racelog_test"))


def _ctx(backend: str):
    import decord
    if backend == "cpu":
        return decord.cpu(0)
    if backend == "gpu":
        return decord.gpu(0)
    if backend == "hybrid":
        return decord.hybrid_gpu(0)   # 与引擎 TRT 路径同上下文
    raise ValueError(backend)


def run_arm(codec: str, backend: str, batch: int, rounds: int,
            n_frames: int | None, roi=None) -> dict:
    """单臂 N 轮取中位；返回 {fps_med, fps_all, ...}。"""
    import decord
    name, full = VIDS[codec]
    path = str(_vdir() / name)
    n = min(n_frames or full, full)
    kw = {} if backend == "gpu" else {"num_threads": THREADS[codec]}
    if roi:
        kw["roi"] = roi
    fps_all = []
    for _ in range(rounds):
        vr = decord.VideoReader(path, ctx=_ctx(backend),
                                output_format="yuv420", **kw)
        vr.seek(0)
        got, t0 = 0, time.perf_counter()
        while got < n:
            e = min(got + batch, n)
            b = vr.get_batch(list(range(got, e)))
            got += b.shape[0]
            del b
        dt = time.perf_counter() - t0
        vr.close()
        fps_all.append(got / dt)
    return {"codec": codec, "backend": backend, "n": n,
            "fps_med": round(statistics.median(fps_all), 1),
            "fps_all": [round(x, 1) for x in fps_all]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--frames", type=int, default=0, help="0=全片")
    ap.add_argument("--codec", default="", help="限定单码（默认三码）")
    ap.add_argument("--backend", default="", help="限定单后端（默认三后端）")
    ap.add_argument("--json", default="", help="落盘路径")
    args = ap.parse_args()

    codecs = [args.codec] if args.codec else ["h264", "hevc", "av1"]
    backends = [args.backend] if args.backend else ["cpu", "gpu", "hybrid"]
    n_frames = args.frames or None

    print("纯解码基准（decode-only，全帧 %s，批 %d，%d 轮取中位）"
          % ("全片" if not n_frames else "%d 帧" % n_frames, args.batch,
             args.rounds))
    print("机器: %s" % os.environ.get("PROCESSOR_IDENTIFIER", "?"))
    rows = []
    print("\n%-6s %-8s %10s   %s" % ("编码", "后端", "fps(中位)", "各轮"))
    for codec in codecs:
        for backend in backends:
            try:
                r = run_arm(codec, backend, args.batch, args.rounds, n_frames)
            except Exception as e:  # noqa: BLE001 — 单臂失败不阻断矩阵
                print("%-6s %-8s %10s   %r" % (codec, backend, "FAIL", e))
                continue
            rows.append(r)
            print("%-6s %-8s %10.1f   %s"
                  % (codec, backend, r["fps_med"],
                     " ".join("%.0f" % x for x in r["fps_all"])))

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        print("\n→ %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
