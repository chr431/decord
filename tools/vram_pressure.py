"""占住指定 MiB 的显存后运行给定命令,命令结束后释放并透传退出码。

用法: python tools/vram_pressure.py <MiB> <cmd> [args...]
"""
import subprocess
import sys

from cuda.bindings import driver as cu

mib = int(sys.argv[1])
cmd = sys.argv[2:]

cu.cuInit(0)
err, dev = cu.cuDeviceGet(0)
err, ctx = cu.cuDevicePrimaryCtxRetain(dev)
cu.cuCtxSetCurrent(ctx)

chunk = 256 * 1024 * 1024
allocs = []
while len(allocs) * 256 < mib:
    err, ptr = cu.cuMemAlloc(chunk)
    if err != cu.CUresult.CUDA_SUCCESS:
        print(f"[vram_pressure] alloc stopped at {len(allocs)*256} MiB: {err}", flush=True)
        break
    allocs.append(ptr)
print(f"[vram_pressure] holding {len(allocs)*256} MiB", flush=True)

try:
    rc = subprocess.run(cmd).returncode
finally:
    for p in allocs:
        cu.cuMemFree(p)
    cu.cuDevicePrimaryCtxRelease(dev)
print(f"[vram_pressure] child exit={rc}", flush=True)
sys.exit(rc)
