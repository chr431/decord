"""Regression tests for fixes ported from upstream dmlc/decord issues/PRs.

Covers:
  - VideoReader.close() / context manager (upstream #222)
  - import with PATH unset and the RTLD_GLOBAL escape hatch (upstream #357/#361/#362)
  - AVReader sample_rate default matching its docstring (upstream PR #237)
"""
import inspect
import os
import subprocess
import sys

import numpy as np
import pytest
from decord import VideoReader, AVReader, cpu
from decord._ffi.base import _LIB


def _video_path():
    return os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', '..',
        'examples', 'flipping_a_pancake.mkv'))


def _run_python(code, **extra_env):
    env = os.environ.copy()
    env.update(extra_env)
    env['PYTHONPATH'] = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', '..', 'python'))
    # point the child at the same native library the test process loaded,
    # regardless of how it was installed (site-packages vs local build tree)
    env['DECORD_LIBRARY_PATH'] = os.path.dirname(_LIB._name)
    return subprocess.run(
        [sys.executable, '-c', code],
        cwd=os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', '..')),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_close_is_idempotent_and_disables_reader():
    vr = VideoReader(_video_path(), ctx=cpu(0))
    assert vr._handle is not None
    vr.close()
    assert vr._handle is None
    vr.close()  # second close must be a no-op, not a double free


def test_reader_supports_context_manager():
    with VideoReader(_video_path(), ctx=cpu(0)) as vr:
        assert vr[0].asnumpy().shape == (240, 426, 3)
    assert vr._handle is None


def test_import_with_path_unset():
    """base.py must not KeyError when PATH is missing (upstream #357)."""
    env = os.environ.copy()
    env.pop('PATH', None)
    env['PYTHONPATH'] = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', '..', 'python'))
    env['DECORD_LIBRARY_PATH'] = os.path.dirname(_LIB._name)
    proc = subprocess.run(
        [sys.executable, '-c', 'import decord; assert decord.__version__'],
        cwd=os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', '..')),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr


def test_rtld_global_escape_hatch_still_loads():
    """DECORD_RTLD_GLOBAL=1 keeps the historical load mode working."""
    proc = _run_python('import decord; assert decord.__version__',
                       DECORD_RTLD_GLOBAL='1')
    assert proc.returncode == 0, proc.stderr


def test_av_reader_sample_rate_default_matches_docstring():
    """Upstream PR #237: docstring says -1 (keep original rate), signature said 44100."""
    sig = inspect.signature(AVReader.__init__)
    assert sig.parameters['sample_rate'].default == -1


def test_large_pts_get_batch_does_not_hang_or_truncate():
    """Upstream #269/PR #341: PTS > INT32_MAX used to truncate in NDArray::pts,
    breaking frame counting and hanging get_batch/seek on such clips."""
    path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', 'test_data', 'large_pts.mp4'))
    vr = VideoReader(path, ctx=cpu(0))
    assert len(vr) == 30
    # random access exercises SeekAccurate -> SkipFramesImpl on a
    # single-keyframe H.264 clip (previously infinite-looped at EOF)
    for i in range(0, len(vr), 7):
        assert vr[i].asnumpy().shape == (240, 320, 3)
    batch = vr.get_batch([0, 7, 14, 21, 28]).asnumpy()
    assert batch.shape == (5, 240, 320, 3)
    assert np.isfinite(batch).all()
