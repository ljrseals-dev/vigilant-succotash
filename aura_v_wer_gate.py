"""AURA-V Tier-2 WER gate: dry vs Chain A (no AA) vs Chain B (AA) [+ optional 16-bit ProDither].
Run locally:  pip install numpy scipy jiwer openai-whisper
  python aura_v_wer_gate.py --wav dry_vocal.wav --ref reference.txt --model small.en
Self-test of the DSP only (no Whisper needed):  python aura_v_wer_gate.py --selftest"""
import argparse, re, sys, numpy as np
from scipy import signal
from scipy.io import wavfile

COEFFS = np.array([2.033, -2.165, 1.959, -1.590, 0.6149])
rng = np.random.default_rng(1234)

def curve(x, a=2.0, bias=0.3): return np.tanh(a*x + bias) - np.tanh(bias)
def q(x, bits=8): s = 2**(bits-1); return np.round(x*s)/s
def zoh(x, D): return np.repeat(x[::D], D)[:len(x)]
def aa(x, sr, D, fc_frac=0.45):
    return signal.fftconvolve(x, signal.firwin(511, fc_frac*sr/D, fs=sr), mode="same")

def chain(x, sr, D=4, use_aa=False, a=2.0, bias=0.3, bits=8, jowiphon_dbfs=None):
    y = curve(x, a, bias)
    if use_aa: y = aa(y, sr, D)
    if jowiphon_dbfs is not None:                       # noise BEFORE the 8-bit quantizer (dither role)
        n = rng.standard_normal(len(y)); n *= 10**(jowiphon_dbfs/20)/n.std(); y = y + n
    return zoh(q(y, bits), D)

def prodither16(x):                                     # corrected loop: dither inside, feedback subtracted, newest-first
    s = 2**15; d = (rng.random(len(x)) - rng.random(len(x)))/s
    y = np.empty_like(x); h = np.zeros(5)
    for n in range(len(x)):
        t = x[n] - np.dot(COEFFS, h) + d[n]; y[n] = np.round(t*s)/s
        h = np.roll(h, 1); h[0] = y[n] - t + d[n]
    return y

def norm_text(t): return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]", " ", t.lower())).strip()

def load(path, sr_out=44100):
    sr, x = wavfile.read(path); x = x.astype(np.float64)
    if x.ndim > 1: x = x.mean(1)
    x /= max(abs(x).max(), 1e-9)
    if sr != sr_out: x = signal.resample_poly(x, sr_out, sr)
    return 0.5*x/abs(x).max(), sr_out                  # same 0.5 peak drive as the measured tests

def save(path, x, sr): wavfile.write(path, sr, (np.clip(x, -1, 1)*32767).astype(np.int16))

def wer_run(wavs, ref, model):
    import whisper, jiwer
    m = whisper.load_model(model); r = norm_text(ref)
    for name, p in wavs.items():
        h = norm_text(m.transcribe(p, temperature=0.0, language="en")["text"])
        print(f"{name:22s} WER = {100*jiwer.wer(r, h):5.1f}%   | {h[:90]}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--wav"); ap.add_argument("--ref"); ap.add_argument("--model", default="small.en")
    ap.add_argument("--decim", type=int, default=4); ap.add_argument("--jowiphon", type=float, default=None, help="dBFS RMS before 8-bit quantizer, e.g. -48")
    ap.add_argument("--selftest", action="store_true"); A = ap.parse_args()
    if A.selftest:
        sr = 44100; t = np.arange(sr*3)/sr
        f0 = 120 + 20*np.sin(2*np.pi*3*t); src = signal.sawtooth(2*np.pi*np.cumsum(f0)/sr)
        v = sum(signal.lfilter(*signal.iirpeak(fc/(sr/2), 8), src) for fc in (700, 1200, 2600))
        fric = signal.lfilter(*signal.butter(4, [3000/(sr/2), 8000/(sr/2)], "band"), rng.standard_normal(len(t)))
        env = (np.sin(2*np.pi*1.5*t) > 0)*1.0
        x = v/abs(v).max()*env + 0.5*fric/abs(fric).max()*(1-env); x = 0.5*x/abs(x).max()
        ya, yb = chain(x, sr), chain(x, sr, use_aa=True)
        Y = lambda y: np.abs(np.fft.rfft(y*np.hanning(len(y))))**2; f = np.fft.rfftfreq(len(x), 1/sr)
        base = (f > 20) & (f < sr/8)
        for nm, y in (("A no-AA", ya), ("B AA", yb)):
            e = Y(y - x); print(f"{nm}: error energy 0-5.5 kHz relative to input = {10*np.log10(e[base].sum()/Y(x)[base].sum()):6.1f} dB")
        print("prodither16 finite:", np.all(np.isfinite(prodither16(0.001*np.sin(2*np.pi*1000*t[:44100]))))); sys.exit()
    x, sr = load(A.wav); ref = open(A.ref).read()
    out = {"dry": x, "A_noAA": chain(x, sr, A.decim, False, jowiphon_dbfs=A.jowiphon),
           "B_AA": chain(x, sr, A.decim, True, jowiphon_dbfs=A.jowiphon)}
    out["B_AA+ProDither16"] = prodither16(out["B_AA"])
    wavs = {}
    for k, y in out.items(): save(f"{k}.wav", y, sr); wavs[k] = f"{k}.wav"
    wer_run(wavs, ref, A.model)
