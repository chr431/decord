import os
import numpy as np
import pytest
from decord import AVReader, cpu
from decord.base import DECORDError

CTX = cpu(0)
_EXAMPLES = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'examples')
_TEST_DATA = os.path.join(os.path.dirname(__file__), '..', '..', 'test_data')

def get_normal_av_reader():
    return AVReader(os.path.join(_EXAMPLES, 'count.mov'), CTX)

def test_normal_av_reader():
    av = get_normal_av_reader()
    assert len(av) == 143

def test_bytes_io():
    fn = os.path.join(_EXAMPLES, 'count.mov')
    with open(fn, 'rb') as f:
        av = AVReader(f)
        assert len(av) == 143
        av2 = get_normal_av_reader()
        audio, video = av[10]
        audio2, video2 = av2[10]
        assert np.allclose(audio.asnumpy(), audio2.asnumpy())
        assert np.allclose(video.asnumpy(), video2.asnumpy())

def test_no_audio_stream():
    with pytest.raises(DECORDError):
        AVReader(os.path.join(_TEST_DATA, 'video_0.mov'), CTX)

def test_index():
    av = get_normal_av_reader()
    audio, video = av[0]

def test_indices():
    av = get_normal_av_reader()
    audio, video = av[:]

def test_get_batch():
    av = get_normal_av_reader()
    av.get_batch([-1, 0, 1, 2, 3])

if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__]))
