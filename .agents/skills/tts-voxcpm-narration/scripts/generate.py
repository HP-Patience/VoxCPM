"""Local, resumable VoxCPM2 narration. Run --help; no service/system mutations."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback

import numpy as np
import soundfile as sf


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def split_text(text, max_chars):
    if max_chars < 10:
        raise ValueError('max-chars must be at least 10')
    paragraphs = [re.sub(r'\s*\n\s*', ' ', p.strip())
                  for p in re.split(r'\n\s*\n', text.strip()) if p.strip()]
    segments = []
    for para in paragraphs:
        units = []
        for sentence in re.findall(r'[^。！？!?]+[。！？!?]*|[。！？!?]+', para):
            if len(sentence) <= max_chars:
                units.append(sentence)
            else:
                clauses = re.findall(r'[^，,；;：:]+[，,；;：:]*|[，,；;：:]+', sentence)
                for clause in clauses:
                    units.extend(clause[i:i+max_chars] for i in range(0, len(clause), max_chars))
        part = ''
        for unit in units:
            if part and len(part) + len(unit) > max_chars:
                segments.append(part)
                part = ''
            part += unit
        if part:
            segments.append(part)
    if not segments or ''.join(segments) != ''.join(paragraphs):
        raise ValueError('Empty input or segmentation changed text')
    return segments


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ('model-path', 'reference', 'text-file', 'output-dir'):
        p.add_argument('--' + flag, type=Path, required=True)
    p.add_argument('--prompt-file', type=Path, help='Accurate reference transcript; enables prompt mode')
    p.add_argument('--device', default='cuda')
    p.add_argument('--steps', type=int, default=10)
    p.add_argument('--cfg', type=float, default=2.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-chars', type=int, default=110)
    p.add_argument('--gap', type=float, default=0.3)
    p.add_argument('--denoise', action='store_true')
    p.add_argument('--normalize', action='store_true')
    p.add_argument('--dry-run', action='store_true', help='Inspect inputs without loading model or writing output')
    p.add_argument('--limit', type=int, default=0, help='Process only first N segments; 0 = all')
    return p


def inspect_inputs(args):
    if not 1 <= args.steps <= 100 or not 0.1 <= args.cfg <= 10:
        raise ValueError('steps must be 1..100 and cfg 0.1..10')
    if not 0 <= args.gap <= 10 or args.limit < 0 or args.seed < 0:
        raise ValueError('gap must be 0..10; limit and seed must be nonnegative')
    text = args.text_file.read_text(encoding='utf-8-sig')
    segments = split_text(text, args.max_chars)
    config_path = args.model_path / 'config.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if config.get('architecture', '').lower() != 'voxcpm2':
        raise ValueError('This helper requires a VoxCPM2 model')
    weights = [x for x in args.model_path.iterdir() if x.suffix in ('.safetensors', '.pth', '.bin')]
    if not weights:
        raise ValueError('Local model weights not found')
    reference, sr = sf.read(str(args.reference), dtype='float32', always_2d=True)
    if not len(reference) or not np.isfinite(reference).all() or np.max(np.abs(reference)) < 1e-6:
        raise ValueError('Reference audio is empty, silent or invalid')
    prompt = args.prompt_file.read_text(encoding='utf-8-sig').strip() if args.prompt_file else None
    if prompt == '':
        raise ValueError('Reference transcript is empty')
    manifest = {
        'version': 1, 'helper_sha256': digest(__file__),
        'text_sha256': digest(args.text_file),
        'reference': str(args.reference.resolve()), 'reference_sha256': digest(args.reference),
        'reference_info': {'duration': len(reference)/sr, 'sample_rate': sr, 'channels': reference.shape[1]},
        'prompt_text': prompt, 'model_path': str(args.model_path.resolve()),
        'config_sha256': digest(config_path),
        'weights_identity': {p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in sorted(weights)},
        'params': {'device': args.device, 'steps': args.steps, 'cfg': args.cfg, 'seed': args.seed,
                   'max_chars': args.max_chars, 'gap': args.gap, 'denoise': args.denoise,
                   'normalize': args.normalize, 'optimize': False},
        'segments': segments,
    }
    return manifest


def check_audio(path, sr=None):
    data, rate = sf.read(str(path), dtype='float32', always_2d=True)
    if data.shape[0] == 0 or data.shape[1] != 1 or not np.isfinite(data).all():
        raise ValueError(f'Invalid generated audio: {path}')
    if sr is not None and rate != sr:
        raise ValueError(f'Sample-rate mismatch: {path}')
    return rate, len(data)


def load_model(args):
    from voxcpm import VoxCPM
    return VoxCPM(voxcpm_model_path=str(args.model_path.resolve()),
                  device=args.device, optimize=False, enable_denoiser=args.denoise)


def prepare_output(args, manifest):
    out = args.output_dir
    identity = out / 'manifest.json'
    if identity.exists():
        if json.loads(identity.read_text(encoding='utf-8')) != manifest:
            raise ValueError('Inputs/parameters/model/helper changed. Use a NEW output directory.')
    else:
        if any(out.iterdir()):
            # The lock is created by main before entering this function.
            if any(p.name != '.generation.lock' for p in out.iterdir()):
                raise ValueError('Output directory is not empty and has no manifest; use a new directory')
        save_json(identity, manifest)
        (out / 'input.txt').write_bytes(args.text_file.read_bytes())


def synthesize(args, manifest, model_factory=load_model):
    prepare_output(args, manifest)
    out = args.output_dir
    previous_path = out / 'segments.json'
    previous = json.loads(previous_path.read_text(encoding='utf-8')) if previous_path.exists() else []
    hashes = {row['file']: row['sha256'] for row in previous}
    model, sr = None, None
    rows = []
    segments = manifest['segments']
    limit = min(args.limit or len(segments), len(segments))
    try:
        for index, text in enumerate(segments[:limit], 1):
            path = out / f'{index:03}.wav'
            started = time.monotonic()
            save_json(out / 'status.json', {'state': 'generating', 'current': index, 'total': len(segments)})
            if path.exists():
                if path.name in hashes and digest(path) != hashes[path.name]:
                    raise ValueError(f'Cached audio changed: {path}; preserve files and inspect')
                rate, _ = check_audio(path, sr)
                sr = rate
            else:
                if model is None:
                    save_json(out / 'status.json', {'state': 'loading', 'current': index, 'total': len(segments)})
                    model = model_factory(args)
                    model_sr = model.tts_model.sample_rate
                    if sr is not None and sr != model_sr:
                        raise ValueError('Model sample rate differs from cached audio')
                    sr = model_sr
                save_json(out / 'status.json', {'state': 'generating', 'current': index, 'total': len(segments)})
                kwargs = dict(text=text, reference_wav_path=str(args.reference.resolve()),
                              inference_timesteps=args.steps, cfg_value=args.cfg,
                              denoise=args.denoise, normalize=args.normalize, seed=args.seed+index-1)
                if manifest['prompt_text'] is not None:
                    kwargs.update(prompt_wav_path=str(args.reference.resolve()), prompt_text=manifest['prompt_text'])
                wav = np.asarray(model.generate(**kwargs))
                if wav.ndim != 1 or not wav.size or not np.isfinite(wav).all():
                    raise ValueError(f'Invalid output for segment {index}')
                tmp = out / f'{index:03}.partial.wav'
                sf.write(str(tmp), wav, sr, subtype='PCM_16')
                check_audio(tmp, sr)
                tmp.replace(path)
            _, frames = check_audio(path, sr)
            offset = sum(r['frames'] for r in rows) + len(rows)*int(sr*args.gap)
            rows.append({'index': index, 'file': path.name, 'text': text, 'frames': frames,
                         'duration': frames/sr, 'sample_rate': sr, 'sha256': digest(path),
                         'start_seconds': offset/sr, 'seed': args.seed+index-1})
            # Preserve later receipts when reviewing just the first N cached segments.
            save_json(previous_path, rows + [r for r in previous if r['index'] > index])
            print(f'DONE {index}/{len(segments)} audio={frames/sr:.2f}s elapsed={time.monotonic()-started:.1f}s', flush=True)
        if limit < len(segments):
            save_json(out / 'status.json', {'state': 'partial', 'completed': limit, 'total': len(segments)})
            return
        gap_frames = int(sr*args.gap)
        tmp = out / 'narration.partial.wav'
        final = out / 'narration.wav'
        with sf.SoundFile(str(tmp), 'w', samplerate=sr, channels=1, subtype='PCM_16') as dest:
            for index, row in enumerate(rows):
                if index:
                    dest.write(np.zeros(gap_frames, dtype=np.float32))
                audio, _ = sf.read(str(out / row['file']), dtype='float32')
                dest.write(audio)
        _, frames = check_audio(tmp, sr)
        if frames != sum(r['frames'] for r in rows) + (len(rows)-1)*gap_frames:
            raise ValueError('Merged frame count is incorrect')
        tmp.replace(final)
        save_json(out / 'status.json', {'state': 'complete', 'total': len(rows),
                                       'duration': frames/sr, 'output': str(final.resolve())})
        print(f'COMPLETE {final.resolve()} duration={frames/sr:.2f}s', flush=True)
    except BaseException as exc:
        save_json(out / 'status.json', {'state': 'failed', 'completed': len(rows), 'total': len(segments), 'error': str(exc)})
        raise


def main():
    args = parser().parse_args()
    manifest = inspect_inputs(args)
    print(json.dumps({'reference': manifest['reference_info'], 'segments': len(manifest['segments']),
                      'params': manifest['params']}, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        print(json.dumps(manifest['segments'], ensure_ascii=False, indent=2))
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = args.output_dir / '.generation.lock'
    # Never remove a pre-existing lock automatically, even if it appears stale.
    with lock.open('x', encoding='utf-8') as f:
        json.dump({'pid': os.getpid(), 'created': time.time()}, f)
    try:
        prepare_output(args, manifest)
        with (args.output_dir / 'generation-stderr.log').open('a', encoding='utf-8', buffering=1) as log:
            with contextlib.redirect_stderr(log):
                try:
                    synthesize(args, manifest)
                except BaseException:
                    traceback.print_exc()
                    raise
    finally:
        lock.unlink()


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'FAILED: {exc}', file=sys.stderr, flush=True)
        sys.exit(1)
