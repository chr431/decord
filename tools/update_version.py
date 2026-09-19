"""
This is the global script that set the version information of DECORD.
This script runs and update all the locations that related to versions
List of affected files:
- decord-root/python/decord/_ffi/libinfo.py
- decord-root/pyproject.toml
- decord-root/include/decord/runtime/c_runtime_api.h

用法：python tools/update_version.py [X.Y.Z]
- 带参数：三处版本全部设为 X.Y.Z（本地发版准备用）。
- 无参数：以 libinfo.py 的当前版本为事实源，把三处拉齐（幂等）。

本文件**不内嵌版本号**：release.yml 的 bump 清单不含本文件，此前内嵌
第 4 份版本串随每次发版漂移（0.8.1→0.8.3、0.8.3→0.8.4 各漂一次），
谁跑一次无参版本就把三源降级——2026-09-20 审计复触了一次，故根治。
"""
import os
import re
import sys


def _current_version(libinfo_path):
    m = re.search(r'(?<=__version__ = ")[.0-9a-z]+',
                  open(libinfo_path, encoding="utf-8").read())
    if not m:
        raise RuntimeError("Cannot find version in %s" % libinfo_path)
    return m.group(0)

# Implementations
def update(file_name, pattern, repl):
    update = []
    hit_counter = 0
    need_update = False
    for l in open(file_name):
        result = re.findall(pattern, l)
        if result:
            assert len(result) == 1
            hit_counter += 1
            if result[0] != repl:
                l = re.sub(pattern, repl, l)
                need_update = True
                print("%s: %s->%s" % (file_name, result[0], repl))
            else:
                print("%s: version is already %s" % (file_name, repl))

        update.append(l)
    if hit_counter != 1:
        raise RuntimeError("Cannot find version in %s" % file_name)

    if need_update:
        with open(file_name, "w") as output_file:
            for l in update:
                output_file.write(l)


def main():
    curr_dir = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    proj_root = os.path.abspath(os.path.join(curr_dir, ".."))
    libinfo = os.path.join(proj_root, "python", "decord", "_ffi", "libinfo.py")
    if len(sys.argv) > 1:
        if not re.fullmatch(r"[.0-9a-z]+", sys.argv[1]):
            raise SystemExit("版本号非法: %r" % sys.argv[1])
        ver = sys.argv[1]
    else:
        ver = _current_version(libinfo)
        print("无参数：以 libinfo.py 为事实源，把三处拉齐到 %s" % ver)
    # python path（版本事实源；pyproject.toml 与 python/decord/__init__.py 从中派生）
    update(libinfo, r"(?<=__version__ = \")[.0-9a-z]+", ver)
    # wheel 元数据（scikit-build-core 读 [project].version）
    update(os.path.join(proj_root, "pyproject.toml"),
           r"(?<=^version = \")[.0-9a-z]+", ver)
    # C++ header
    update(os.path.join(proj_root, "include", "decord", "runtime", "c_runtime_api.h"),
           "(?<=DECORD_VERSION \")[.0-9a-z]+", ver)

if __name__ == "__main__":
    main()
