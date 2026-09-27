# -*- coding: utf-8 -*-
"""DECORD_USE_CUDA ifdef 剥离器（R3-2 GPU-only 解包，2026-09-28）。

GPU-only 化（USE_CUDA 强制 ON，OFF 即 FATAL_ERROR）后，纯
DECORD_USE_CUDA 条件编译成为死代码面。本脚本对 4 个文件
（hybrid_threaded_decoder.cc/.h、video_reader.cc、improc.h）把
「只引用 DECORD_USE_CUDA」的条件指令静态求值并展开：
    #ifdef DECORD_USE_CUDA  A #else B #endif   ->  A
    #ifndef DECORD_USE_CUDA A #else B #endif   ->  B
    #if defined(DECORD_USE_CUDA) ...           ->  同 ifdef
    #if !defined(DECORD_USE_CUDA) ...          ->  同 ifndef
混合条件（表达式还引用其他宏）**原样保留**——GPU-only 后
DECORD_USE_CUDA 恒定义，混合条件语义不变，留给编译器求值。
嵌套条件用栈正确配对（丢弃分支内的 #if/#endif 也计深度）。
行尾保持原样（CRLF 文件按 CRLF 写回）。

用法：python tools/_purge_cuda_ifdefs.py [--dry] [文件...]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FILES = [
    'src/video/hybrid_threaded_decoder.cc',
    'src/video/hybrid_threaded_decoder.h',
    'src/video/video_reader.cc',
    'src/improc/improc.h',
]

DIR_RE = re.compile(r'^[ \t]*#[ \t]*(ifdef|ifndef|if|elif|else|endif)\b(.*)$')
TOKEN_RE = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')


def strip_comment(expr: str) -> str:
    expr = re.sub(r'//.*$', '', expr)
    expr = re.sub(r'/\*.*?\*/', '', expr)
    return expr.strip()


def eval_pure(expr: str):
    """表达式只引用 DECORD_USE_CUDA 时返回 bool；否则 None（不动）。"""
    names = [t for t in TOKEN_RE.findall(expr) if t != 'defined']
    if any(n != 'DECORD_USE_CUDA' for n in names):
        return None
    py = expr.replace('defined(DECORD_USE_CUDA)', '1')
    py = py.replace('defined DECORD_USE_CUDA', '1')
    py = re.sub(r'\bDECORD_USE_CUDA\b', '1', py)
    py = py.replace('&&', ' and ').replace('||', ' or ').replace('!', ' not ')
    try:
        return bool(eval(py, {'__builtins__': {}}, {}))  # noqa: S307 - 纯 1/布尔运算
    except Exception:
        return None


def classify(kind: str, raw_expr: str):
    """返回 (ours, val)：ours=是否纯 DECORD_USE_CUDA 条件。"""
    expr = strip_comment(raw_expr)
    if kind == 'ifdef':
        return (expr == 'DECORD_USE_CUDA'), True
    if kind == 'ifndef':
        return (expr == 'DECORD_USE_CUDA'), False
    # #if：defined()/!/&&/|| 组合，纯 DECORD 才处理
    val = eval_pure(expr)
    return (val is not None), val


def purge(text: str):
    """返回 (新文本, 删除行数, 保留的混合条件指令数)。"""
    lines = text.split('\n')
    out = []
    stack = []   # {'ours': bool, 'alive': bool, 'seen_true': bool}
    dropped = 0
    mixed_dirs = 0

    def ctx_below(top_n: int) -> bool:
        return all(f['alive'] for f in stack[:top_n])

    for ln in lines:
        m = DIR_RE.match(ln)
        if not m:
            if all(f['alive'] for f in stack):
                out.append(ln)
            else:
                dropped += 1
            continue
        kind, raw = m.group(1), m.group(2)
        if kind in ('ifdef', 'ifndef', 'if'):
            outer = ctx_below(len(stack))
            ours, val = classify(kind, raw)
            if ours:
                stack.append({'ours': True, 'alive': bool(val) and outer,
                              'seen_true': bool(val)})
                dropped += 1          # 指令行本身删除
            else:
                stack.append({'ours': False, 'alive': True,
                              'seen_true': False})
                mixed_dirs += 1
                if outer:
                    out.append(ln)    # 混合条件原样保留（含其指令行）
                else:
                    dropped += 1
            continue
        if not stack:
            # 畸形输入（不配对的 #else/#endif）：原样保留防误删
            out.append(ln)
            continue
        top = stack[-1]
        outer = ctx_below(len(stack) - 1)
        if kind == 'elif':
            if top['ours']:
                val = eval_pure(strip_comment(raw))
                if val is None:
                    print(f'WARN: DECORD 链上混合 #elif: {raw!r}，按 False',
                          file=sys.stderr)
                    val = False
                top['alive'] = (not top['seen_true']) and bool(val) and outer
                if val:
                    top['seen_true'] = True
                dropped += 1
            else:
                if outer:
                    out.append(ln)
                else:
                    dropped += 1
            continue
        if kind == 'else':
            if top['ours']:
                top['alive'] = (not top['seen_true']) and outer
                dropped += 1
            else:
                if outer:
                    out.append(ln)
                else:
                    dropped += 1
            continue
        # endif
        stack.pop()
        if top['ours']:
            dropped += 1
        else:
            if outer:
                out.append(ln)
            else:
                dropped += 1
    return '\n'.join(out), dropped, mixed_dirs


def main():
    args = sys.argv[1:]
    dry = '--dry' in args
    files = [a for a in args if not a.startswith('--')] or DEFAULT_FILES
    for rel in files:
        p = ROOT / rel
        raw = p.read_bytes()
        crlf = b'\r\n' in raw
        text = raw.decode('utf-8')
        if crlf:
            text = text.replace('\r\n', '\n')
        new, dropped, mixed = purge(text)
        stat = f'{rel}: 删除 {dropped} 行（含指令行），混合条件保留 {mixed}'
        if dry:
            print(stat)
            continue
        data = new.encode('utf-8')
        if crlf:
            data = data.replace(b'\n', b'\r\n')
        p.write_bytes(data)
        print(stat + '  [written]')


if __name__ == '__main__':
    main()
