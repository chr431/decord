# Decord（现代化 fork）

[English](README_EN.md) | 简体中文

![CI Build](https://github.com/chr431/decord/workflows/C/C++%20CI/badge.svg?branch=dev)

![symbol](docs/symbol.png)

`Decord` 是 `Record` 的逆序词。它在硬件加速视频解码器（FFmpeg / NVIDIA Video Codec
/NVDEC）之上提供一层薄封装，把视频当作可随机访问的帧序列来用——为深度学习的数据
管线解决"视频随机访问又慢又麻烦"的问题，也能解码音频并与视频同步切片。

本仓库是 [dmlc/decord](https://github.com/dmlc/decord) 的**现代化 fork**，在保持上游
0.6.0 API 完全兼容（已逐文件比对，公开 API 零缺失，差异全部是向后兼容扩展）的前提下
做了大范围翻新：

- **只支持 FFmpeg 9.0**（编译期强制，`avcodec 63`）。7.x/8.x 兼容已移除：跨版本的
  关键帧索引与 seek 落点行为差异曾引入难以排查的正确性问题。
- **CUDA 13 + NVDEC** 构建链路翻新，Windows / Linux CI 均覆盖。
- **hybrid 混合解码（实验性）**：单路 demux 在关键帧边界拆分给 CPU 软解与 NVDEC
  并行解码、按呈现序合并——见 [hybrid 混合解码](#hybrid-混合解码实验性)。
- **吞吐与正确性专项**（详见[本 fork 的主要改动](#本-fork-的主要改动)）：消费端批组装
  解锁、seek 着陆自愈、混合调度节拍、NV12 直出、关键帧索引磁盘缓存等。

> ⚠️ **hybrid / hybrid_gpu 是实验性接口**：调度策略、性能与显存/内存上界可能随版本
> 变化且不另行通知。生产环境请使用 `cpu()` / `gpu()`。

## 目录

- [安装](#安装)
- [快速上手](#快速上手)
- [seek 语义：seek() vs seek_accurate()](#seek-语义seek-vs-seek_accurate)
- [hybrid 混合解码（实验性）](#hybrid-混合解码实验性)
- [性能参考](#性能参考)
- [API 速览](#api-速览)
- [环境变量参考](#环境变量参考)
- [本 fork 的主要改动](#本-fork-的主要改动)
- [测试](#测试)
- [发布流程（维护者）](#发布流程维护者)
- [致谢与许可](#致谢与许可)

## 安装

### 硬性依赖

- **FFmpeg 9.0**（shared 构建，`avcodec-63`）。推荐
  [BtbN n9.0 win64 gpl-shared](https://github.com/BtbN/FFmpeg-Builds/releases)。
- GPU（NVDEC）构建只需 NVIDIA 驱动（**0.8.1 起不需要 CUDA Toolkit 与
  Video Codec SDK**：驱动 API 运行时动态加载，improc kernel 以内嵌 PTX 提供）。

### 预编译 wheel（Windows，推荐）

从 [GitHub Releases](https://github.com/chr431/decord/releases) 下载
`decord-<版本>-cp3xx-cp3xx-win_amd64.whl`，直接安装：

```bash
pip install decord-<版本>-cp313-cp313-win_amd64.whl
```

wheel 自包含运行时：`decord.dll` + FFmpeg 导入闭包（`avcodec/avformat/avutil/
avfilter/swresample/swscale` 共 6 个 DLL，约 163 MB 原始 / 64 MB 压缩）会装进
`site-packages/decord/`，`import decord` 即用，不依赖源码树、构建目录或 PATH
上有没有 FFmpeg。未打包 `avdevice` 与 `ffprobe.exe`（decord.dll 不导入）。
不依赖 PyPI 上的 decord（那是上游 0.6.0 CPU 版）。

### 便携 zip（Windows）

Releases 同时提供 `decord-<版本>-win64-gpu.zip`（布局 = RaceVideoToLog 的
`_decord_build/`，解压即用）：

```
_decord_build/
├── decord.dll            # CPU+NVDEC 构建
├── avcodec-63.dll 等     # FFmpeg 9.0 运行库
├── ffprobe.exe
└── python/decord/        # Python 绑定
```

使用方式：把 `python/decord` 所在目录加入 `PYTHONPATH`（或直接拷到你的项目里），
并保证 `decord.dll` 与 FFmpeg DLL 同目录或在 `PATH` 上。

### 从源码 pip 构建

构建由 `pyproject.toml`（scikit-build-core）驱动：`pip install` 会用 CMake 编译共享
库与 FFmpeg 运行闭包一起打进 wheel。Windows 下 pip 构建**默认启用 CUDA**（GPU 变体，
与本 fork 发布形态一致），需要 `FFMPEG_DIR` 指向 FFmpeg SDK（include/ + lib/）：

```bash
git clone --recursive https://github.com/chr431/decord
cd decord

# Windows：GPU（默认）——只要求 NVIDIA 驱动，无需 Toolkit
export FFMPEG_DIR=D:/path/to/ffmpeg-n9.0-latest-win64-gpl-shared-9.0
pip install .

# Windows：纯 CPU 覆盖
pip install . --config-settings="cmake.define.USE_CUDA=OFF"

# 仓库内一键脚本（vcvars + FFMPEG_DIR 默认值已配好）
make_wheel.bat
```

Linux 下 NVDEC 构建若报 `libnvcuvid.so` 找不到，可参考上游
[#102](https://github.com/dmlc/decord/issues/102)：用 `ldconfig -p | grep libnvcuvid`
找到该库并链接到 `CUDA_TOOLKIT_ROOT_DIR/lib64`。

### 开发构建（CMake + PYTHONPATH）

```bash
# Windows（启用 CUDA）
cmake -S . -B build -DUSE_CUDA=ON \
      -DFFMPEG_DIR="D:/path/to/ffmpeg-n9.0-latest-win64-gpl-shared-9.0" -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

# Linux / macOS
cmake -S . -B build -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release -DFFMPEG_DIR=/path/to/ffmpeg
cmake --build build -j$(nproc)

# 直接用新构建的库跑 Python 绑定（无需 pip install）
PYTHONPATH=python python -c "import decord; print(decord.__version__)"
```

常用 CMake 选项：`-DUSE_CUDA=ON|OFF`（NVDEC；0.8.1 起无需 CUDA Toolkit）、
`-DFFMPEG_DIR=...`、`-DDECORD_INSTALL_LIBDIR=...`。Python 绑定按
`build/`（Windows 为 `build/Release`）→ `DECORD_LIBRARY_PATH` 的顺序找库。

## 快速上手

### VideoReader

```python
import decord
from decord import VideoReader, cpu, gpu

print(decord.__version__, decord.__ffmpeg_version__)   # 0.8.1 9.0.x

vr = VideoReader('examples/flipping_a_pancake.mkv', ctx=cpu(0))
# 文件对象也可以（内存内解码）
with open('examples/flipping_a_pancake.mkv', 'rb') as f:
    vr = VideoReader(f, ctx=cpu(0))

print('video frames:', len(vr))

# 逐帧顺序读取（内部自动做最高效的 seek/skip）
for i in range(len(vr)):
    frame = vr[i]          # decord.ndarray.NDArray, HxWx3
    ...

# 批量读取：随机/乱序索引最高效的入口
frames = vr.get_batch([1, 3, 5, 7, 9])
print(frames.shape)        # (5, 240, 320, 3)
# 重复索引合法，内部去重，不重复解码
frames2 = vr.get_batch([1, 2, 3, 2, 3, 4, 3, 4, 5]).asnumpy()

# 等差步长采样（分频/frame-skip）走流式快速路径：一次顺序解码完成整段
frames3 = vr.get_batch(range(0, 600, 3))

# 流式发射：后台线程预取下一批，解码完即交付（批间顺序=完成序）
for start_pos, batch in vr.get_batch_stream(range(0, len(vr)), batch=250):
    ...  # 批内按 indices 序；跨批按 start_pos 重排或与顺序无关
```

### 输出格式与 ROI

```python
# output_format: 'rgb' (HxWx3) | 'gray' (HxW) | 'yuv420' (H*3/2 x W 半平面 NV12)
vr = VideoReader('video.mp4', ctx=cpu(0), output_format='yuv420')

# ROI-first：固化后解码器只输出该矩形（CPU 在颜色转换前裁剪，
# GPU 只转换 ROI 窗口），必须在读任何帧之前设置
vr = VideoReader('video.mp4', ctx=gpu(0), roi=(100, 60, 740, 660))
frame = vr.next()          # 直接得到 ROI 帧
```

### 元数据探测（不开解码器）

```python
from decord import probe

info = probe('video.mp4')
# {'duration_s':…, 'bit_rate':…, 'nb_frames':…, 'video_codec':'h264',
#  'width':1920, 'height':1080, 'pix_fmt':'yuv420p', 'avg_fps':…,
#  'audio_streams':1, 'subtitle_streams':0}
# 等价于一次最小 ffprobe，毫秒级，不创建解码器、不分配帧缓冲
```

### seek 语义：seek() vs seek_accurate()

```python
vr.seek(pos)            # 快速 seek：落点只在"关键帧级别"保证（可能落在
                        # pos 之前或之后的关键帧），文档化的非精确行为
vr.seek_accurate(pos)   # 精确 seek：着陆校验自愈，保证下一帧 == 顺序读的第 pos 帧
vr.skip_frames(100)     # 顺序跳帧
```

需要帧级精度的随机访问请用 `seek_accurate()` 或 `get_batch([pos])`（两者都逐帧精确）。
`seek()` 只适合"回到开头/大致位置"的场景。

### AudioReader / AVReader / VideoLoader

音频与音视频同步切片、以及训练用批量装载器与上游用法一致：

```python
from decord import AudioReader, AVReader, VideoLoader

ar = AudioReader('example.mp3', ctx=cpu(0), sample_rate=44100, mono=False)
print(ar[0:5])

av = AVReader('example.mov', ctx=cpu(0))
audio, video = av[0:20]      # 每帧视频对应一段音频样本

vl = VideoLoader(['1.mp4', '2.avi', '3.mpeg'], ctx=[cpu(0)],
                 shape=(2, 320, 240, 3), interval=1, skip=5, shuffle=1)
for batch in vl:
    ...
```

shuffle 模式与上游一致：`0` 全顺序；`1` 文件名乱序（每文件内顺序，最高效）；
`2` 全随机；`3` 文件内随机帧访问。

### 深度学习框架桥接

```python
import decord
decord.bridge.set_bridge('torch')    # 'mxnet' | 'torch' | 'tensorflow' | 'native'
# 此后 vr[0] / get_batch(...) 直接返回目标框架的张量
```

## hybrid 混合解码（实验性）

`decord.hybrid(dev_id)`（输出落主机内存）与 `decord.hybrid_gpu(dev_id)`（帧驻留显存）
把**一路 demux 码流在关键帧边界切成块**，按"最少剩余工作量"（实测产率 + 实时在途）分给 CPU 软解
（FFmpeg，多帧并行）与 NVDEC（CUVID），两侧并行解码、按呈现序合并发射：

- 调度器在线学习两侧产率（CPU 侧为 filter 后有效产出，GPU 侧为落地速率 EWMA），
  按比例分配块；粘性块交替避免频繁断流。
- 内存/显存有自适应硬预算（默认取空闲量的 30%，可用环境变量覆盖），所有队列深度
  由预算推导，防 OOM。
- 选择 hybrid 即是显式要求 CPU+GPU 混跑：**内部不做慢侧自动回退**，即使某些码流
  上混跑慢于纯 NVDEC（尊重用户选择；实验语义不作静默替换）。

```python
import warnings
from decord import VideoReader, hybrid, hybrid_gpu

with warnings.catch_warnings():
    warnings.simplefilter('ignore')          # 实验性 UserWarning 可按需忽略
    vr = VideoReader('video.mp4', ctx=hybrid(0), output_format='yuv420')

vr.seek_accurate(0)
batch = vr.get_batch(range(0, 1000))         # 与普通 reader 完全同 API
```

**何时有用**：软解与 NVDEC 速率接近的码流（典型如 1080p H.264）收益最大；软解明显
慢于 NVDEC 的码流（hevc/av1）混跑通常介于两者之间——快于纯 CPU、慢于纯 NVDEC。
请以自己机器上的实测为准（见[性能参考](#性能参考)）。

## 性能参考

以下为**本机实测口径**（AMD Ryzen 9 7945HX + RTX 4060 Laptop，Windows，FFmpeg 9.0
shared，1080p，`output_format='yuv420'`，`get_batch` 每批 250 帧顺序读，5-10 轮取
中位，单位 fps）。不同机器/码流/分辨率/**电源计划**差异很大，请自行实测；hybrid
属实验性，数字仅供量级参考。

| 解码方式 | H.264 | HEVC | AV1 |
|---|---|---|---|
| `cpu`（16 解码线程） | 1233 | 961 | 709 |
| `gpu`（NVDEC） | 958 | 1868 | 1285 |
| `hybrid`（实验性，→主机内存） | **1570** | 1680 | **1340** |
| `hybrid_gpu`（实验性，→显存） | ~1520 | 1848 | ~1250 |

要点：

- 三码流 hybrid 均显著快于纯 CPU；h264/av1 同时快于纯 NVDEC；hevc 引擎口径
  （hyb_gpu）与纯 NVDEC 相当（~99%）。
- **电源计划对混跑稳定性影响巨大**：CPU/GPU 混合负载下，Windows 电源策略的
  睿频行为会造成同二进制 ±15-25% 的运行间方差（表现为"双峰"）。若需要可复现
  的基准数字，固定电源计划（高性能/卓越性能）后测量。
- `get_batch` 的吞吐包含全部管线（消费端批组装：批缓冲池 + 拷贝/解码重叠），纯
  CPU 路径同样受益。
- `probe()` 元数据探测毫秒级、不建解码器，适合做调度前的资产检查。

## API 速览

`VideoReader`（完整文档见 `python/decord/video_reader.py` docstring）：

| 成员 | 说明 |
|---|---|
| `len(vr)` / `get_key_indices()` | 帧数 / 关键帧索引列表 |
| `vr[i]` / `vr.next()` | 单帧读取（NDArray） |
| `vr.get_batch(indices, roi=None)` | 批量读取；支持乱序、重复索引、等差步长快速路径 |
| `vr.get_batch_stream(indices, batch=250)` | 流式发射：后台预取，批间按完成序交付 `(start_pos, batch)` |
| `vr.seek(pos)` / `vr.seek_accurate(pos)` | 快速（关键帧级）/ 精确（帧级）seek |
| `vr.skip_frames(n)` | 顺序跳帧 |
| `vr.next_roi(x1,y1,x2,y2)` / `get_batch(..., roi=…)` | ROI 读取（ROI-first 时解码器只输出该矩形） |
| `vr.get_frame_timestamp(idx)` / `get_avg_fps()` | 时间戳 / 平均帧率 |
| `vr.get_codec()` / `get_color_range()` | 编码名 / 流 color range |
| `probe(uri)`（模块级） | 不开解码器的容器/流元数据（dict） |
| `get_ffmpeg_version()` / `decord.__ffmpeg_version__` | 加载中的原生库实际链接的 FFmpeg 版本 |

模块级：`cpu(id)`、`gpu(id)`、`hybrid(id)`、`hybrid_gpu(id)`、`probe`、
`bridge.set_bridge(...)`、`VideoLoader`、`AudioReader`、`AVReader`。

## 环境变量参考

常用（全部可选，不设即用自适应默认值）：

| 变量 | 默认 | 说明 |
|---|---|---|
| `DECORD_LIBRARY_PATH` | — | Python 绑定搜索 `decord.dll`/`libdecord.so` 的额外目录 |
| `DECORD_FFMPEG_THREAD_COUNT` | 物理核数（clamp 2–16） | CPU 软解线程数（NVDEC 不受影响） |
| `DECORD_FILTER_THREADS` | 1 | CPU 侧 filter 线程数 |
| `DECORD_PREFETCH_DEPTH` | 8 | 纯 CPU/GPU 管线的 demux 领先深度 |
| `DECORD_CPU_FRAME_QUEUE_SIZE` | 自适应 | CPU 解码输出队列深度（内存上界） |
| `DECORD_CONVERT_WORKERS` | 自动（RGB 输出 2–8） | RGB 转换扇出线程池（`0/1` 关闭） |
| `DECORD_BATCH_BUF_POOL` / `DECORD_BATCH_BUF_POOL_MAX` | 1 / 2 | get_batch 批缓冲池开关与保留块数 |
| `DECORD_BATCH_COPY_WORKERS` | 2–4 自动 | 批拷贝工作线程（`0` 恢复串行拷贝） |
| `DECORD_DISABLE_INDEX_CACHE` | 未设 | 设后禁用关键帧索引磁盘缓存（默认写系统缓存目录） |
| `DECORD_HYBRID_VRAM_BUDGET_MB` / `DECORD_HYBRID_RAM_BUDGET_MB` | 空闲量 ×0.65 / ×0.45 | hybrid 显存/内存硬预算（覆盖自动值） |
| `DECORD_HYBRID_PINNED_POOL` | 1 | CPU-out 的 pinned 主机帧池（D2H 直达；`0` 回退暂存路径） |
| `DECORD_PREFETCH_DEPTH_HYBRID` | 384 | hybrid demux 领先基线（实际取 max(此值, 解码器建议)） |
| `DECORD_EOF_RETRY_MAX` / `DECORD_REWIND_RETRY_MAX` | 10240 / 16 | 损坏流容错的 EOF/回退重试上界 |

诊断用（日常无需）：

| 变量 | 说明 |
|---|---|
| `DECORD_HYBRID_DEBUG` | hybrid 调度/预算/发射遥测（stderr） |
| `DECORD_SEEK_DEBUG` | seek 落点跟踪 |
| `DECORD_CPU_RATE_DEBUG` | CPU 产率段折叠打印 |
| `DECORD_HYBRID_FORCE_SIDE=cpu\|gpu` | 强制混合解码全走单侧（实验对照） |
| `DECORD_HYBRID_PREFETCH` | 直控 hybrid demux 领先深度（绕过公式） |

## 本 fork 的主要改动

相对上游 0.6.0（除依赖翻新外）的主要工程项，按主题：

**吞吐**

- get_batch 消费端解锁：批缓冲池（免每批 ~0.8GB 大块分配的首触缺页税）+ 逐帧拷贝
  派发后台线程与解码重叠；等差步长（分频采样）走单遍流式快速路径。
- CPU 管线：NV12 直出（filter `format=nv12` + 两次 memcpy 打包）替代标量 U/V 交错
  循环；解码线程默认随物理核扩展（clamp 2–16）。
- NVDEC：每帧 CUDA event 同步替代全流同步；D2H 异步环；上载批量化（批末单 sync）。
- hybrid：least-remaining-work 调度 + 库存帽按产率比分账 + pinned 帧池 D2H 直达；
  demux 决策节拍与速率学习解耦；GPU 领先硬顶防 dav1d 尾帧死锁。

**正确性**

- seek 着陆校验改为 pts 锚定的有界自愈循环（命中交付 / 欠冲前进 / 过冲回退关键帧）。
- FFmpeg 9-only 编译期门禁，消除跨版本索引/落点行为差异这一整类问题。
- 重复索引、批缓冲池复用、混合块边界的并发修正（逐字节 lockstep 套件守护）。

**工程**

- 关键帧索引磁盘缓存（按 路径哈希 + 大小 + mtime 失效，写系统缓存目录）。
- `probe()` / `get_ffmpeg_version()` C API 与 Python 绑定。
- ROI-first 解码管线（CPU/GPU 统一只输出固定矩形）。
- 测试：`tests/run_fast.sh` 并行四套件（md5 / 流式 / 步长 / lockstep 字节级对照）。

完整历史见 commit log；`dev` 与 `master` 均受 CI 监听。

## 测试

```bash
# 快速回归（约 3 分钟；DECORD_LIBRARY_PATH 指向构建产物目录）
DECORD_LIBRARY_PATH=<dll目录> bash tests/run_fast.sh [帧数]

# 单套件
python tests/test_hybrid.py --n 600          # md5 + seek
python tests/test_hybrid_lockstep.py 600     # 混跑字节级交错对照
```

## 发布流程（维护者）

版本事实源为 `python/decord/_ffi/libinfo.py` 的 `__version__`（`pyproject.toml` 同步，
用 `python tools/update_version.py` 统一更新）。发布走 GitHub Actions → **Release** →
Run workflow：填版本号（如 `0.8.1`）与 ref（默认 `master`），workflow 自动 bump 版本 →
tag `vX.Y.Z` → 构建 pip wheel（`decord-<ver>-cp3xx-win_amd64.whl`）→ 打包
`decord-<ver>-win64-gpu.zip` → 创建 Release 并上传两个产物。构建失败不产生任何
commit/tag/release。tag push 不触发 PyPI 发布（已移除）。

## 致谢与许可

- 上游项目 [dmlc/decord](https://github.com/dmlc/decord)——本 fork 的全部基础架构
  （FFI、reader/loader、bridge）来自上游，遵循 **Apache License 2.0**（见
  [LICENSE](LICENSE)）。
- 视频解码依赖 [FFmpeg](https://ffmpeg.org)（动态链接）。本仓库以 Apache-2.0 发布；
  FFmpeg 自身的许可（LGPL-2.1+，启用 GPL 构建组件时为 GPL）随所选二进制适用，
  分发时请一并遵守。
- NVIDIA Video Codec SDK / NVDEC 遵循 NVIDIA 相应许可。
