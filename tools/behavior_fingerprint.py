"""decord 行为指纹：对固定视频集合用固定参数解码,输出每路结果的 sha256。

用途:打包/构建方式变更(editable -> wheel 等)前后各跑一次,对比哈希
完全一致即"行为无变化"。只依赖 numpy + decord 本身,不写任何状态。

用法: python tools/behavior_fingerprint.py [视频路径 ...]
不传视频则用内置默认集合(bench_videos + racelog_test 各取前几个)。
"""
import hashlib
import sys
from pathlib import Path

import numpy as np

import decord
from decord import VideoReader, cpu, gpu

REPO = Path(__file__).resolve().parent.parent
DEFAULT_VIDEOS = [
    REPO / "bench_videos" / "synthetic_1080p60s.mp4",
    Path(r"D:\Videos\racelog_test\test.mp4"),
    Path(r"D:\Videos\racelog_test\test6.mp4"),
]
ROIS = [None, (100, 80, 640, 400)]  # 全帧 + 固定 ROI(半开区间,>0.7.5)


def digest(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def fingerprint_one(path: str, ctx, tag: str) -> None:
    for fmt in ("rgb", "yuv420", "gray"):
        for roi in ROIS:
            kw = {}
            if roi is not None:
                try:
                    kw["roi"] = roi
                except Exception:
                    continue  # 旧 DLL 无 ROI API
            try:
                vr = VideoReader(str(path), ctx=ctx, output_format=fmt, num_threads=4, **kw)
            except Exception as e:
                print(f"{tag}|{Path(path).name}|{fmt}|roi={roi}|OPEN-FAIL {type(e).__name__}: {e}")
                continue
            n = min(len(vr), 12)
            batch = vr.get_batch(list(range(n))).asnumpy()
            h1 = digest(batch)
            # 顺序 next 3 帧
            seq = [vr.next().asnumpy() for _ in range(3)]
            h2 = digest(np.stack(seq))
            # 精确 seek 到第 5 帧再取一批
            vr.seek_accurate(5)
            b2 = vr.get_batch([5, 6, 7]).asnumpy()
            h3 = digest(b2)
            meta = ""
            try:
                meta = f"codec={vr.get_codec()} range={vr.get_color_range()}"
            except Exception:
                pass
            print(f"{tag}|{Path(path).name}|{fmt}|roi={roi}|batch={h1} seq={h2} seek={h3} shape={batch.shape} {meta}", flush=True)


def main() -> None:
    videos = [Path(p) for p in sys.argv[1:]] or [p for p in DEFAULT_VIDEOS if p.exists()]
    print(f"decord {decord.__version__} from {Path(decord.__file__).parent}")
    for v in videos:
        for name, mk in (
            ("cpu", lambda: cpu(0)),
            ("gpu", lambda: gpu(0)),
            ("hybrid", lambda: _ctx("hybrid")),
            ("hybrid_gpu", lambda: _ctx("hybrid_gpu")),
        ):
            try:
                ctx = mk()
            except Exception as e:
                print(f"{name}|{v.name}|CTX-UNAVAILABLE {type(e).__name__}: {e}")
                continue
            try:
                fingerprint_one(str(v), ctx, name)
            except Exception as e:
                print(f"{name}|{v.name}|RUN-FAIL {type(e).__name__}: {e}")


def _ctx(name: str):
    import decord as d
    return getattr(d, name)(0)


if __name__ == "__main__":
    main()
