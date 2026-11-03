import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal
exec(open('aura_v_edd_harness.py').read().split('# ---------- TEST 1')[0])   # reuse blocks (SR, vantage, thermal_noise, rms_scale)

COEFFS = np.array([2.033, -2.165, 1.959, -1.590, 0.6149])

def tpdf(n, lsb):
    return (rng.random(n) - rng.random(n)) * lsb

def prodither(x, bits=16, order="as_pasted", sign=+1, dither=True):
    """TPDF + 5-tap error feedback. order='as_pasted' reproduces error_history[::-1] pairing
    (coeff[0] on the OLDEST of 5 errors); 'newest_first' pairs coeff[0] with e[n-1]."""
    q = 2 ** (bits - 1); lsb = 1 / q
    d = tpdf(len(x), lsb) if dither else np.zeros(len(x))
    y = np.empty_like(x); hist = np.zeros(5)      # hist[0] = newest error
    c = COEFFS if order == "newest_first" else COEFFS[::-1]
    for n in range(len(x)):
        fb = sign * np.dot(c, hist)
        t = x[n] + d[n] + fb
        y[n] = np.round(t * q) / q
        hist = np.roll(hist, 1); hist[0] = y[n] - t
    return y

# ---- NTF check: noise transfer function 1 + sign*sum c_k z^-k
w = np.linspace(0, np.pi, 2048)
print("=== ProDither NTF (theory) ===")
variants = {}
for order in ("as_pasted", "newest_first"):
    c = COEFFS if order == "newest_first" else COEFFS[::-1]
    for sign in (+1, -1):
        h = 1 + sign * sum(c[k] * np.exp(-1j * w * (k + 1)) for k in range(5))
        H = 20 * np.log10(abs(h) + 1e-12); f = w / np.pi * SR / 2
        band = lambda a, b: np.mean(H[(f >= a) & (f <= b)])
        print(f"{order:13s} sign={sign:+d}: 1-5k={band(1000,5000):6.1f} dB  15-20k={band(15000,20000):6.1f} dB  <1k={band(50,1000):6.1f} dB  max={H.max():5.1f} dB")
        variants[(order, sign)] = H

# ---- GATE 3B
print("\n=== GATE 3B: 1 kHz @ -60 dBFS ===")
n = SR * 3; tt = np.arange(n) / SR
x = 10 ** (-60 / 20) * np.sin(2 * np.pi * 1000 * tt)
def spurs(y, tag, bits):
    f, P = signal.welch(y, SR, nperseg=2**15, window="blackmanharris")
    db = 10 * np.log10(P + 1e-30); i1 = np.argmin(abs(f - 1000)); fund = db[i1 - 2:i1 + 3].max()
    res = []
    for k in range(2, 12):
        i = np.argmin(abs(f - 1000 * k))
        loc = np.r_[db[i - 60:i - 6], db[i + 7:i + 61]]
        res.append((db[i - 3:i + 4].max() - fund, db[i - 3:i + 4].max() - np.median(loc)))
    worst = max(res, key=lambda r: r[0])
    real = [r for r in res if r[1] > 6]
    band = lambda a, b: 10 * np.log10(np.mean(P[(f >= a) & (f <= b)]) + 1e-30)
    print(f"{bits}b {tag:26s} worst harm={worst[0]:6.1f} dBc (margin over local floor {worst[1]:+5.1f} dB, "
          f"real spurs={len(real)})  floor 1-5k={band(1000,5000):6.1f}  15-20k={band(15000,20000):6.1f} dB/Hz")
    return f, P
plots = {}
for bits in (16, 24):
    q = 2 ** (bits - 1)
    plots[f"{bits} plain"] = spurs(np.round(x * q) / q, "undithered", bits)
    plots[f"{bits} TPDF flat"] = spurs(prodither(x, bits, sign=0), "TPDF only (no shaping)", bits)
    for order, sign in (("as_pasted", +1), ("as_pasted", -1), ("newest_first", +1), ("newest_first", -1)):
        y = prodither(x, bits, order, sign)
        if not np.all(np.isfinite(y)) or np.max(abs(y)) > 0.5:
            print(f"{bits}b ProDither {order} sign{sign:+d}: UNSTABLE/OVERLOAD"); continue
        plots[f"{bits} {order} {sign:+d}"] = spurs(y, f"ProDither {order} {sign:+d}", bits)

# TPDF level check
d = tpdf(10**6, 1.0)
print(f"\nTPDF as pasted: peak-to-peak={d.max()-d.min():.2f} LSB, RMS={d.std():.3f} LSB (spec said 1 LSB RMS)")

# ---- ALIASING METRIC: non-harmonic energy of Vantage chain
print("\n=== ALIASING METRIC: non-harmonic energy / fundamental (1 kHz, 0.5 amp) ===")
t2 = np.arange(SR * 2) / SR; s = 0.5 * np.sin(2 * np.pi * 1000 * t2)
def alias_db(y):
    Y = np.abs(np.fft.rfft(y * signal.windows.blackmanharris(len(y)))) ** 2
    f = np.fft.rfftfreq(len(y), 1 / SR); mask = np.ones(len(f), bool)
    for k in range(0, 23): mask &= abs(f - 1000 * k) > 6
    i1 = np.argmin(abs(f - 1000)); fund = Y[i1 - 3:i1 + 4].sum()
    return 10 * np.log10(Y[mask].sum() / fund), 10 * np.log10(Y[mask & (f > 15000)].sum() / fund)
b, a = signal.butter(8, 0.4 * SR / 4 / (SR / 2))   # AA lowpass before decimation (~4.4 kHz)
cfgs = {
 "asym curve only": vantage(s, bit_depth=None, decim=1),
 "+8-bit": vantage(s, decim=1),
 "+decim/4 ZOH (full chain)": vantage(s),
 "+decim/2 ZOH": vantage(s, decim=2),
 "AA-LPF then full chain": vantage(signal.lfilter(b, a, vantage(s, bit_depth=None, decim=1)), a=1, bias=0, decim=4) ,
}
for k, y in cfgs.items():
    tot, hf = alias_db(y); print(f"{k:28s} non-harmonic = {tot:6.1f} dBc   (>15 kHz: {hf:6.1f} dBc)")

# ---- GATE 3A SWEEP
print("\n=== GATE 3A SWEEP: 8-bit, Jowiphon dither BEFORE quantizer, 5 seeds ===")
def worst_real(y):
    f, P = signal.welch(y, SR, nperseg=2**15, window="blackmanharris"); db = 10*np.log10(P+1e-30)
    i1 = np.argmin(abs(f-1000)); fund = db[i1-2:i1+3].max(); out = []
    for k in range(2, 12):
        i = np.argmin(abs(f-1000*k)); loc = np.r_[db[i-60:i-6], db[i+7:i+61]]
        pk = db[i-3:i+4].max()
        if pk - np.median(loc) > 6: out.append(pk - fund)
    return max(out) if out else None
nn = SR * 2; t3 = np.arange(nn) / SR
print("sig dBFS | noise dBFS rms | undithered spur dBc | dithered: real-spur dBc (n/5 seeds with real spur)")
for sd in (-36, -30, -24):
    xs = 10 ** (sd / 20) * np.sin(2 * np.pi * 1000 * t3)
    base = worst_real(np.round(xs * 128) / 128)
    for ndb in (-54, -48, -42):
        r = []
        for _ in range(5):
            nz = rms_scale(thermal_noise(nn, flicker_corner=100), ndb)
            r.append(worst_real(np.round((xs + nz) * 128) / 128))
        vals = [v for v in r if v is not None]
        print(f"{sd:8d} | {ndb:14d} | {base if base is not None else float('nan'):19.1f} | "
              f"{(np.mean(vals) if vals else float('nan')):6.1f}  ({len(vals)}/5)")

fig, ax = plt.subplots(figsize=(10, 5))
for k in ("16 plain", "16 TPDF flat", "16 newest_first +1", "16 as_pasted +1"):
    if k in plots:
        f, P = plots[k]; ax.semilogx(f[1:], 10*np.log10(P[1:]+1e-30), lw=.7, label=k)
ax.set_xlim(100, 22050); ax.set_ylim(-190, -50); ax.legend(); ax.grid(True, which="both", alpha=.3)
ax.set_title("Gate 3B: 16-bit, 1 kHz @ -60 dBFS"); plt.savefig("aura_v_gate3b.png", dpi=130)
