#!/usr/bin/env python3
"""
AURA-V offline integration test.  Needs numpy + scipy only (numba optional).
Keep next to jowiphon_noise.py and prodither_fixed.py.

Chain under test (44.1 kHz):
  asym tanh -> [AA FIR] -> Jowiphon thermal dither -> 8-bit quantize -> /4 ZOH
  -> Jowiphon BS.1770 gain -> true-peak trim -> ProDither 16-bit (Lipshitz, fixed)

Gates:  LOUDNESS  TRUE-PEAK  SPUR (8-bit)  NOISE-FLOOR (16-bit)  DETERMINISM
Chain A = no AA filter, Chain B = AA filter.  WER is NOT measured here.
"""
import os, sys, importlib.util
import numpy as np
from scipy import signal

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m
JN = _load("jowiphon_noise", "jowiphon_noise.py")
PD = _load("prodither_fixed", "prodither_fixed.py")

SR, D, AA_TAPS, AA_FC = 44100, 4, 511, 0.45          # AA cutoff = 0.45 * (SR / D)
LATENCY_SAMPLES = (AA_TAPS - 1) // 2                  # linear-phase FIR group delay

def curve(x, a=2.0, bias=0.3): return np.tanh(a * x + bias) - np.tanh(bias)
def aa(x): return signal.fftconvolve(x, signal.firwin(AA_TAPS, AA_FC * SR / D, fs=SR), mode="same")

def front_end(x, use_aa, dither_dbfs, seed, bits=8, quantize=True):
    y = curve(x)
    if use_aa: y = aa(y)
    if dither_dbfs is not None:
        z = JN.thermal_noise(len(y), SR, np.random.default_rng(seed), 100.0)
        y = y + z / np.sqrt(np.mean(z ** 2)) * 10 ** (dither_dbfs / 20)
    if quantize:
        q = 2 ** (bits - 1); y = np.round(y * q) / q
    return np.repeat(y[::D], D)[: len(x)]

def back_end(y8, target, tp_ceiling, seed, shaping="lipshitz"):
    m = JN.integrated_lufs(y8, SR); g = target - m
    y = JN.apply_gain(y8, g, SR)
    y, trimmed = JN.apply_ceiling(y, 10 ** (tp_ceiling / 20), SR)
    cfg = PD.ProDitherConfig(bit_depth=16, sample_rate_hz=SR, mode="clean", shaping=shaping, dither_seed=seed)
    return y, PD.ProDither(cfg).process(y), m, g, trimmed

def speech_like(seconds=6, seed=3):
    rng = np.random.default_rng(seed); t = np.arange(int(SR * seconds)) / SR
    f0 = 120 + 20 * np.sin(2 * np.pi * 3 * t); v = signal.sawtooth(2 * np.pi * np.cumsum(f0) / SR)
    v = sum(signal.lfilter(*signal.iirpeak(c / (SR / 2), 8), v) for c in (700, 1200, 2600)); v /= abs(v).max()
    fr = signal.lfilter(*signal.butter(4, [3000 / (SR / 2), 8000 / (SR / 2)], "band"), rng.standard_normal(len(t)))
    fr /= abs(fr).max(); env = np.clip(np.sin(2 * np.pi * 1.2 * t), 0, 1) ** 2
    x = v * env + 0.4 * fr * (1 - env) * (np.sin(2 * np.pi * 2.4 * t) > 0)
    return 0.5 * x / abs(x).max()

def spur_count(err, thr_db=10.0):
    """Narrowband lines in the error spectrum that stand >thr_db above the local floor."""
    f, P = signal.welch(err, SR, nperseg=2 ** 15, window="blackmanharris"); db = 10 * np.log10(P + 1e-30)
    prom = np.zeros_like(db)
    for i in range(80, len(db) - 80):
        loc = np.r_[db[i - 80:i - 3], db[i + 4:i + 81]]; prom[i] = db[i] - np.median(loc)
    pk, _ = signal.find_peaks(prom, height=thr_db, distance=6)
    return len(pk), (prom[pk].max() if len(pk) else 0.0)

def band_db(err, a, b):
    f, P = signal.welch(err, SR, nperseg=2 ** 13); return 10 * np.log10(np.mean(P[(f >= a) & (f <= b)]) + 1e-30)

def gate(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}"); return ok

def main():
    results = []
    print(f"AA FIR latency to declare in the graph: {LATENCY_SAMPLES} samples = {1000*LATENCY_SAMPLES/SR:.2f} ms @ {SR} Hz (Chain B only)")

    # ---- SPUR GATE (8-bit stage isolated: error vs the same chain with no quantizer) ----
    print("\n== SPUR GATE: 1 kHz tone, 6 s, error = chain - unquantized chain ==")
    t = np.arange(SR * 6) / SR; tone = 0.03 * np.sin(2 * np.pi * 1000 * t)     # ~ -30 dBFS, small enough to see 8-bit spurs
    for label, use_aa in (("Chain A (no AA)", False), ("Chain B (AA)", True)):
        ref = front_end(tone, use_aa, None, 0, quantize=False)
        n0, p0 = spur_count(front_end(tone, use_aa, None, 0) - ref)
        row = {}
        for dbfs in (-54.0, -48.0):
            cnt = [spur_count(front_end(tone, use_aa, dbfs, s) - ref)[0] for s in range(5)]; row[dbfs] = cnt
        print(f" {label}: undithered lines={n0} (max {p0:.1f} dB over floor) | -54 dBFS: {row[-54.0]} | -48 dBFS: {row[-48.0]}")
        results.append(gate(f"{label} spurs buried at -48 dBFS (sensitivity: undithered must show lines)",
                            n0 > 0 and sum(row[-48.0]) == 0, f"undithered={n0}, dithered counts per seed={row[-48.0]}"))

    # ---- LOUDNESS / TRUE PEAK / NOISE FLOOR on a speech-like signal ----
    x = speech_like(); print("\n== LOUDNESS + TRUE PEAK + 16-bit floor: speech-like signal, 6 s ==")
    for target, tp in ((-23.0, -1.0), (-14.0, -1.0)):
        for label, use_aa in (("A", False), ("B", True)):
            y8 = front_end(x, use_aa, -48.0, 0)
            y, out, m, g, trimmed = back_end(y8, target, tp, 0)
            L = JN.integrated_lufs(out, SR); TP = JN.true_peak_dbtp(out, SR)
            err = L - target
            ok_l = abs(err) <= 0.5 or (trimmed and TP <= tp + 0.1)
            print(f" Chain {label} target {target:.0f} LUFS / {tp:.0f} dBTP: pre={m:.2f} gain={g:+.2f} -> out={L:.2f} LUFS (err {err:+.2f}), TP={TP:.2f}, trimmed={trimmed}")
            results.append(gate(f"Chain {label} {target:.0f} LUFS loudness", ok_l, "hit within 0.5 LU" if abs(err) <= 0.5 else ("TP-limited (trim binding)" if trimmed else "MISSED")))
            results.append(gate(f"Chain {label} {target:.0f} LUFS true peak <= ceiling+0.1", TP <= tp + 0.1, f"{TP:.2f} dBTP vs {tp:.1f}"))

    # 16-bit noise-floor shaping: error of the final stage, fixed Lipshitz vs flat TPDF
    print("\n== 16-bit FINAL STAGE: shaped vs flat TPDF (error = ProDither out - its input) ==")
    y8 = front_end(x, False, -48.0, 0); y, out_l, *_ = back_end(y8, -23.0, -1.0, 0)
    cfg = PD.ProDitherConfig(bit_depth=16, sample_rate_hz=SR, shaping="none", dither_seed=0)
    out_f = PD.ProDither(cfg).process(y)
    el, ef = out_l - y, out_f - y
    d = {k: (band_db(el, *k) - band_db(ef, *k)) for k in ((50, 1000), (1000, 5000), (15000, 20000))}
    print(f" Lipshitz minus flat (dB): <1k={d[(50,1000)]:+.1f}  1-5k={d[(1000,5000)]:+.1f}  15-20k={d[(15000,20000)]:+.1f}")
    results.append(gate("1-5 kHz floor >=10 dB below flat TPDF", d[(1000, 5000)] <= -10.0, f"{d[(1000,5000)]:+.1f} dB"))
    nl, _ = spur_count(el); results.append(gate("no spurs in 16-bit error", nl == 0, f"{nl} lines"))

    # ---- DETERMINISM ----
    print("\n== DETERMINISM ==")
    y8a = front_end(x, True, -48.0, 7); y8b = front_end(x, True, -48.0, 7)
    oa = back_end(y8a, -23.0, -1.0, 7)[1]; ob = back_end(y8b, -23.0, -1.0, 7)[1]
    results.append(gate("same seeds -> bit-identical output", np.array_equal(oa, ob), f"max diff {np.max(abs(oa-ob)):.1e}"))

    print(f"\nSUMMARY: {sum(results)}/{len(results)} gates passed.  WER (Whisper) still required on a real vocal.")
    return 0 if all(results) else 1

if __name__ == "__main__":
    sys.exit(main())
