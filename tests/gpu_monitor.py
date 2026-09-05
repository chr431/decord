# -*- coding: utf-8 -*-
"""GPU 监控: 每秒采样利用率/显存, 可独立运行或作为模块嵌入.

独立: python tests/gpu_monitor.py --seconds 60
嵌入: from gpu_monitor import GpuMonitor; m=GpuMonitor(); m.start(); ... m.stop()
"""
import argparse
import subprocess
import threading
import time


class GpuMonitor:
    def __init__(self, interval=1.0):
        self.interval = interval
        self.samples = []          # (t, util%, mem MiB)
        self._stop = threading.Event()
        self._thread = None

    def _query(self):
        try:
            out = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=2).stdout.strip()
            util, mem = out.split('\n')[0].split(',')
            return int(util.strip()), int(mem.strip())
        except Exception:
            return None, None

    def _loop(self, t0):
        while not self._stop.is_set():
            u, m = self._query()
            if u is not None:
                self.samples.append((time.perf_counter() - t0, u, m))
            self._stop.wait(self.interval)

    def start(self):
        self._thread = threading.Thread(target=self._loop, args=(time.perf_counter(),), daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def report(self, label=''):
        if not self.samples:
            return 'no samples'
        utils = [s[1] for s in self.samples]
        mems = [s[2] for s in self.samples]
        r = (f"{label} [{len(self.samples)}s] GPU util "
             f"min/avg/max = {min(utils)}/{sum(utils)//len(utils)}/{max(utils)}%, "
             f"VRAM min/avg/max = {min(mems)}/{sum(mems)//len(mems)}/{max(mems)} MiB")
        print(r)
        return r


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seconds', type=int, default=60)
    a = ap.parse_args()
    m = GpuMonitor().start()
    t0 = time.time()
    try:
        while time.time() - t0 < a.seconds:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    m.stop()
    m.report('monitor')
    # 逐秒明细
    for t, u, mem in m.samples:
        print(f"  t={t:6.1f}s util={u:3d}% vram={mem} MiB")
