# -*- coding: utf-8 -*-
"""py-spy 裸地址符号化器（2026-09-28 停滞取证）。

py-spy dump 的 decord.dll 帧显示为 `0x... (?)`（py-spy 自身不加载
第三方 PDB）。本脚本用 dbghelp（SymFromAddr + decord.pdb）离线解析：
  1. 读 py-spy dump 文本，抽全部 0x 十六进制地址
  2. 取目标进程的 decord.dll 基址（EnumProcessModulesEx；pid=0 时用
     本进程，或 --base 显式给）
  3. RVA = addr - base → SymFromAddr(base_of_pdb + RVA)

用法：
    python tools/resolve_addr.py <dump.txt> [--pid N | --base 0x..] [--out ..]
"""
import ctypes
import ctypes.wintypes as wt
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DLL = ROOT / 'build-081fix' / 'decord.dll'
PDB_DIR = str(ROOT / 'build-081fix')

dbghelp = ctypes.WinDLL('dbghelp.dll')
kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_VM_READ = 0x0010
LIST_MODULES_ALL = 0x03


def target_module_base(pid, name=b'decord.dll'):
    k32 = ctypes.WinDLL('kernel32', use_last_error=True)
    h = k32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid)
    if not h:
        raise OSError('OpenProcess failed')
    arr = (ctypes.c_void_p * 2048)()
    need = wt.DWORD()
    # EnumProcessModulesEx 在 psapi 兼容层
    psapi = ctypes.WinDLL('psapi.dll')
    ok = psapi.EnumProcessModulesEx(h, arr, ctypes.sizeof(arr),
                                    ctypes.byref(need), LIST_MODULES_ALL)
    if not ok:
        raise OSError('EnumProcessModulesEx failed')
    n = need.value // ctypes.sizeof(ctypes.c_void_p)
    base = None
    for i in range(n):
        path = ctypes.create_string_buffer(1024)
        psapi.GetModuleFileNameExA(h, arr[i], path, 1024)
        if path.value.lower().endswith(name):
            base = arr[i]
            break
    k32.CloseHandle(h)
    return base


class SYMBOL_INFO(ctypes.Structure):
    _fields_ = [('SizeOfStruct', wt.ULONG), ('TypeIndex', wt.ULONG),
                ('Reserved', ctypes.c_uint64 * 2), ('Index', wt.ULONG),
                ('Size', wt.ULONG), ('ModBase', ctypes.c_uint64),
                ('Flags', wt.ULONG), ('Value', ctypes.c_uint64),
                ('Address', ctypes.c_uint64), ('Register', wt.ULONG),
                ('Scope', wt.ULONG), ('Tag', wt.ULONG),
                ('NameLen', wt.ULONG), ('MaxNameLen', wt.ULONG),
                ('Name', ctypes.c_char * 256)]


def sym_init():
    dbghelp.SymSetOptions(0x00000010 | 0x00000002)   # LOAD_LINES|UNDNAME
    if not dbghelp.SymInitializeW(None, PDB_DIR, False):
        raise OSError('SymInitialize failed')
    base = dbghelp.SymLoadModuleExW(None, 0, str(DLL), None, 0x10000000, 0,
                                    None, 0)
    if not base:
        raise OSError('SymLoadModuleEx failed (PDB 缺失?)')
    return 0x10000000


def resolve(dll_base_in_target, sym_base, addr):
    if addr < dll_base_in_target or addr >= dll_base_in_target + 0x2000000:
        return None
    si = SYMBOL_INFO()
    si.SizeOfStruct = 88
    si.MaxNameLen = 256
    disp = ctypes.c_uint64()
    ok = dbghelp.SymFromAddr(None, sym_base + (addr - dll_base_in_target),
                             ctypes.byref(disp), ctypes.byref(si))
    if not ok:
        return f'decord+0x{addr - dll_base_in_target:x}(?)'
    name = si.Name[:si.NameLen].decode('utf-8', 'replace')
    return f'{name}+0x{disp.value:x}'


def main():
    args = sys.argv[1:]
    dump = Path(args[0])
    pid = None
    base = None
    if '--pid' in args:
        pid = int(args[args.index('--pid') + 1])
    if '--base' in args:
        base = int(args[args.index('--base') + 1], 0)
    if pid is not None:
        base = target_module_base(pid)
    if base is None:
        print('需要 --pid（目标进程存活时）或 --base', file=sys.stderr)
        sys.exit(2)
    sym_base = sym_init()
    txt = dump.read_text(encoding='utf-8', errors='replace')
    out_lines = []
    cache = {}

    def repl(m):
        a = int(m.group(0), 16)
        if a in cache:
            return cache[a]
        r = resolve(base, sym_base, a)
        cache[a] = f'{m.group(0)}{" = " + r if r else ""}' if r else m.group(0)
        return cache[a]

    for ln in txt.splitlines():
        out_lines.append(re.sub(r'\b0x[0-9a-f]{9,16}\b', repl, ln))
    outp = dump.with_name(dump.stem + '.sym.txt')
    outp.write_text('\n'.join(out_lines), encoding='utf-8')
    print(f'目标基址 0x{base:x}；符号化输出 {outp}')


if __name__ == '__main__':
    main()
