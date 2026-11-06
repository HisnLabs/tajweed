"""Tajweed v2: hear a recited phrase and write it in the Quran phonetic script, then (optionally) get per-word
tajweed remarks from the bundled rule engine (imam.py).

  pip install onnxruntime numpy faster-whisper quran-transcript levenshtein
  python inference.py recitation.wav 112 1            # Hafs, surah 112 ayah 1
  python inference.py recitation.wav 112 1 --warsh "قُلْ هُوَ اَ۬للَّهُ أَحَدٌ"   # Warsh: give the Warsh mushaf text

The audio: 16 kHz mono 16-bit WAV (resample first: ffmpeg -i in.mp3 -ar 16000 -ac 1 out.wav), one phrase or one ayah,
at most 30 s.
"""
import argparse, json, os, sys, wave
import numpy as np, onnxruntime as ort
from faster_whisper.feature_extractor import FeatureExtractor  # Whisper's 80-bin log-mel, no 30 s padding

HERE = os.path.dirname(os.path.abspath(__file__))
LEAD = int(16000 * 0.4)  # the training clips start and end with silence


def hear(audio, riwaya='hafs', model='onnx/model-int8.onnx'):
    meta = json.load(open(os.path.join(HERE, 'config.json')))
    sess = ort.InferenceSession(os.path.join(HERE, model), providers=['CPUExecutionProvider'])
    audio = np.concatenate([np.zeros(LEAD, np.float32), audio.astype(np.float32), np.zeros(LEAD, np.float32)])
    mel = FeatureExtractor(feature_size=80)(audio, padding=0)
    mel = mel[:, : min(3000, mel.shape[1] // 2 * 2)]
    lp = sess.run(None, {'mel': mel[None].astype(np.float32), 'riwaya': np.array([meta['riwayat'].index(riwaya)], np.int64)})[0][0]
    ids, step = lp.argmax(-1), len(audio) / 16000 / lp.shape[0]
    out, probs, times, prev = [], [], [], 0
    for t, i in enumerate(ids):  # CTC best path: blank = 0, repeats collapsed
        if i != prev and i != 0:
            out.append(meta['phonemes'][i - 1]); probs.append(float(np.exp(lp[t, i]))); times.append((t + 0.5) * step - LEAD / 16000)
        prev = i
    last = [t for t, i in enumerate(ids) if i != 0]
    times.append(((last[-1] + 1) * step - LEAD / 16000) if last else 0.0)
    return ''.join(out), probs, times


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('wav'); ap.add_argument('surah', type=int); ap.add_argument('ayah', type=int)
    ap.add_argument('--warsh', help='the ayah in the Warsh mushaf (King Fahd) spelling: judge with Warsh rules')
    ap.add_argument('--lang', default='en', choices=['ar', 'en', 'fr'])
    a = ap.parse_args()
    with wave.open(a.wav) as f:
        if f.getframerate() != 16000 or f.getsampwidth() != 2:
            sys.exit('give a 16 kHz 16-bit WAV (ffmpeg -i in.mp3 -ar 16000 -ac 1 out.wav)')
        x = np.frombuffer(f.readframes(f.getnframes()), '<i2').astype(np.float32).reshape(-1, f.getnchannels()).mean(1) / 32768
    riwaya = 'warsh' if a.warsh else 'hafs'
    heard, probs, times = hear(x, riwaya)
    print('heard:', heard)
    sys.path.insert(0, HERE)
    import imam
    if a.warsh:
        import warsh
        words = warsh.phrase(a.warsh.split())
    else:
        from quran_transcript import Aya
        words = Aya(a.surah, a.ayah).get().uthmani.split(' ')
    ref = imam.recited(words, heard, riwaya=riwaya)
    for w in imam.listen(ref, heard, heard_probs=probs, min_prob=0.3, min_prob_jali=0.7, langs=(a.lang,), heard_times=times):
        madd = ' '.join(f"[{m['name'][a.lang]}: {m['heard']}]" for m in w['madd'])
        notes = ' | '.join(n['msg'][a.lang] for n in w['notes'])
        print(f"{w['word']:>14}  {'ok' if w['ok'] else '✗'}  {madd}  {notes}")
