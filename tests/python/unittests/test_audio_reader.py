import os
import numpy as np
import pytest
from decord import AudioReader, cpu
from decord.base import DECORDError

CTX = cpu(0)
_EXAMPLES = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'examples')
_TEST_DATA = os.path.join(os.path.dirname(__file__), '..', '..', 'test_data')

def get_single_channel_reader():
    return AudioReader(os.path.join(_EXAMPLES, 'count.mov'), CTX)

def get_double_channels_reader():
    return AudioReader(os.path.join(_EXAMPLES, 'example.mp3'), CTX, mono=False)

def get_resampled_reader():
    return AudioReader(os.path.join(_EXAMPLES, 'count.mov'), CTX, 4410)

def test_single_channel_audio_reader():
    ar = get_single_channel_reader()
    assert ar.shape == (1, 394176)

def test_double_channels_audio_reader():
    ar = get_double_channels_reader()
    # example.mp3 carries a stereo track
    assert ar.shape[0] == 2
    assert ar.shape[1] > 0

def test_no_audio_stream():
    with pytest.raises(DECORDError):
        AudioReader(os.path.join(_TEST_DATA, 'video_0.mov'), CTX)

def test_bytes_io():
    fn = os.path.join(_EXAMPLES, 'count.mov')
    with open(fn, 'rb') as f:
        ar = AudioReader(f)
        assert ar.shape == (1, 394176)
        ar2 = get_single_channel_reader()
        assert np.allclose(ar[10].asnumpy(), ar2[10].asnumpy())

def test_resample():
    ar = get_resampled_reader()
    original = get_single_channel_reader()
    # resampling 44100 -> 4410 shrinks the sample count by ~10x; the exact
    # count depends slightly on the libswresample version, so use a ratio
    assert ar.shape[1] == pytest.approx(original.shape[1] * 0.1, rel=0.01)

def test_index():
    ar = get_double_channels_reader()
    ar[0]
    ar[-1]

def test_indices():
    ar = get_double_channels_reader()
    ar[:]
    ar[-20:-10]

def test_get_batch():
    ar = get_double_channels_reader()
    ar.get_batch([-1, 0, 1, 2, 3])

def test_get_info():
    ar = get_double_channels_reader()
    ar.get_info()

def test_add_padding():
    ar = get_single_channel_reader()
    num_channels = ar.shape[0]
    duration_before = ar.duration()
    # capture the rate first: add_padding() mutates _duration, which the
    # _effective_sample_rate property derives from
    rate = ar._effective_sample_rate
    num_padding = ar.add_padding()
    assert num_padding > 0
    assert np.array_equal(ar[:num_padding].asnumpy(), np.zeros((num_channels, num_padding)))
    # duration must grow by the padded samples expressed in seconds, not by
    # samples * sample_rate
    assert ar.duration() == pytest.approx(duration_before + num_padding / rate)

def test_free():
    ar = get_single_channel_reader()
    del ar

if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__]))
