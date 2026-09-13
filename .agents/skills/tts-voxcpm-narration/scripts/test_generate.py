"""No GPU, model download, service shutdown or private audio required."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
import numpy as np
import soundfile as sf
import generate as g


class NarrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        model = root / 'model'
        model.mkdir()
        (model / 'config.json').write_text('{"architecture":"voxcpm2"}', encoding='utf-8')
        (model / 'model.safetensors').write_bytes(b'test only; never loaded')
        ref = root / 'ref.wav'
        sf.write(ref, 0.1*np.sin(np.arange(1600)*0.1), 16000)
        text = root / 'text.txt'
        text.write_text('第一段，测试。\n\n第二段，结束。', encoding='utf-8')
        self.args = g.parser().parse_args(['--model-path', str(model), '--reference', str(ref),
                                         '--text-file', str(text), '--output-dir', str(root/'out')])
        self.args.output_dir.mkdir()
        self.manifest = g.inspect_inputs(self.args)
        self.calls = []

    def factory(self, args):
        def generate(**kwargs):
            self.calls.append(kwargs)
            return 0.1*np.sin(np.arange(800)*0.1).astype(np.float32)
        return types.SimpleNamespace(tts_model=types.SimpleNamespace(sample_rate=16000), generate=generate)

    def run_job(self, factory=None):
        with contextlib.redirect_stdout(io.StringIO()):
            g.synthesize(self.args, self.manifest, factory or self.factory)

    def test_segmentation_preserves_text_and_bounds(self):
        for text in ('你好。\n\n世界！', 'A'*400+'，再说'+ 'B'*200+'。', '第一行\n第二行。\n\nZ与U独立。', '！？！？', 'a.b!? c。'):
            parts = g.split_text(text, 30)
            self.assertTrue(all(0 < len(p) <= 30 for p in parts))
            self.assertEqual(''.join(parts).replace(' ', ''), text.replace('\n','').replace(' ',''))
        with self.assertRaises(ValueError):
            g.split_text('  ', 30)

    def test_preview_resume_merge_and_no_reload(self):
        self.args.limit = 1
        self.run_job()
        self.assertEqual(len(self.calls), 1)
        self.assertFalse((self.args.output_dir/'narration.wav').exists())
        self.args.limit = 0
        self.run_job()
        self.assertEqual(len(self.calls), 2)
        info = sf.info(self.args.output_dir/'narration.wav')
        self.assertEqual(info.frames, 800*2+4800)
        self.assertEqual([c['seed'] for c in self.calls], [42, 43])
        self.assertFalse(self.calls[0]['denoise'])
        self.assertNotIn('prompt_text', self.calls[0])
        self.run_job(lambda _: self.fail('Completed cache must not load model'))
        status = json.loads((self.args.output_dir/'status.json').read_text())
        self.assertEqual(status['state'], 'complete')

    def test_changed_input_refused(self):
        self.run_job()
        self.manifest['params']['cfg'] = 3
        with self.assertRaisesRegex(ValueError, 'NEW output'):
            self.run_job()

    def test_changed_cache_refused(self):
        self.run_job()
        sf.write(self.args.output_dir/'001.wav', np.zeros(100), 16000)
        with self.assertRaisesRegex(ValueError, 'Cached audio changed'):
            self.run_job()

    def test_invalid_output_never_committed(self):
        def factory(_):
            return types.SimpleNamespace(tts_model=types.SimpleNamespace(sample_rate=16000),
                                         generate=lambda **k: np.array([float('nan')]))
        with self.assertRaisesRegex(ValueError, 'Invalid output'):
            self.run_job(factory)
        self.assertFalse((self.args.output_dir/'001.wav').exists())
        self.assertEqual(json.loads((self.args.output_dir/'status.json').read_text())['state'], 'failed')

    def test_prompt_mode_passes_real_transcript(self):
        self.manifest['prompt_text'] = '实际录音内容。'
        self.run_job()
        self.assertEqual(self.calls[0]['prompt_text'], '实际录音内容。')
        self.assertEqual(self.calls[0]['prompt_wav_path'], str(self.args.reference.resolve()))

    def test_silent_reference_rejected(self):
        sf.write(self.args.reference, np.zeros(1600), 16000)
        with self.assertRaisesRegex(ValueError, 'silent'):
            g.inspect_inputs(self.args)

    def test_nonempty_unowned_output_preserved(self):
        target = self.args.output_dir/'keep.txt'
        target.write_text('keep')
        with self.assertRaisesRegex(ValueError, 'not empty'):
            self.run_job()
        self.assertEqual(target.read_text(), 'keep')


if __name__ == '__main__':
    unittest.main()
