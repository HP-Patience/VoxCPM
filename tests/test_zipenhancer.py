"""WAV normalization must not depend on TorchCodec / FFmpeg DLLs."""
import importlib.util
from pathlib import Path
from unittest.mock import patch
import sys
import types

import numpy as np
import unittest
import tempfile
import soundfile as sf
import torchaudio


def make_enhancer():
    pipelines = types.ModuleType("modelscope.pipelines")
    pipelines.pipeline = None
    constant = types.ModuleType("modelscope.utils.constant")
    constant.Tasks = types.SimpleNamespace(acoustic_noise_suppression="test")
    spec = importlib.util.spec_from_file_location(
        "zipenhancer_under_test",
        Path(__file__).resolve().parents[1] / "src/voxcpm/zipenhancer.py",
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"modelscope.pipelines": pipelines,
                                 "modelscope.utils.constant": constant}):
        spec.loader.exec_module(module)
    return module.ZipEnhancer.__new__(module.ZipEnhancer)


class TestNormalization(unittest.TestCase):
    def test_normalization_without_torchcodec(self):
        import torch
        enhancer = make_enhancer()
        for channels in (1, 2):
            for sr in (16000, 48000):
                with self.subTest(channels=channels, sr=sr), tempfile.TemporaryDirectory() as tmp:
                    t = np.arange(sr, dtype=np.float32) / sr
                    audio = np.stack([0.1 * np.sin(2 * np.pi * (440 + 220 * c) * t)
                                      for c in range(channels)], axis=1)
                    wav = Path(tmp) / "reference.wav"
                    sf.write(wav, audio, sr, subtype="FLOAT")
                    with patch.object(torchaudio, "load", side_effect=AssertionError("TorchCodec load")), \
                         patch.object(torchaudio, "save", side_effect=AssertionError("TorchCodec save")):
                        enhancer._normalize_loudness(str(wav))
                    actual, actual_sr = sf.read(wav, dtype="float32", always_2d=True)
                    self.assertEqual(actual_sr, sr)
                    self.assertEqual(actual.shape, audio.shape)
                    self.assertTrue(np.isfinite(actual).all())
                    loudness = torchaudio.functional.loudness(torch.from_numpy(actual.T.copy()), sr)
                    self.assertAlmostEqual(float(loudness), -20, delta=0.15)


if __name__ == "__main__":
    unittest.main()
