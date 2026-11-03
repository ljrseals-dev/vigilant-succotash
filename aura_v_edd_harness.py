import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal

SR = 44100
rng = np.random.default_rng(1234)

# ---------- DSP blocks ----------
def vantage(x, a=2.0, bias=0.3, bit_depth=8, decim=4):
    """Asymmetric tanh -> bit-depth reduction -> sample-rate decimation (no AA filter)."""
    y = np.tanh(a * x + bias) - np.tanh(bias)           # DC-corrected asymmetric curve
    if bit_depth:
        q = 2 ** (bit_depth - 1)
        y = np.round(y * q) / q
    if decim > 1:                                       # zero-order hold: true aliasing
        y = np.repeat(y[::decim], decim)[: len(x)]
    return y

def vantage_curve_only(x, a=2.0, bias=0.3):
    return vantage(x, a, bias, bit_depth=None, decim=1)

def harmonic_levels(y, f0, sr, n=6):
    N = len(y)
    w = signal.windows.blackmanharris(N)
    S = np.abs(np.fft.rfft(y * w)) ** 2
    f = np.fft.rfftfreq(N, 1 / sr)
    lv = {}
    for k in range(1, n + 1):
        i = np.argmin(abs(f - k * f0))
        lv[k] = S[i - 3:i + 4].sum()
    return lv, f, S

def thd_report(y, f0=1000, tag=""):
    lv, f, S = harmonic_levels(y, f0, SR)
    even = lv[2] + lv[4] + lv[6]
    odd = lv[3] + lv[5]
    thd = np.sqrt(sum(lv[k] for k in range(2, 7)) / lv[1]) * 100
    db = {k: 10 * np.log10(lv[k] / lv[1]) for k in lv}
    print(f"[{tag}] THD(2-6)={thd:.2f}%  H2={db[2]:.1f} H3={db[3]:.1f} H4={db[4]:.1f} H5={db[5]:.1f} dBc"
          f"  even/odd(power)={even/odd:.2f}")
    return thd, even / odd, f, S

def slope_db_oct(f, P, lo, hi):
    m = (f >= lo) & (f <= hi)
    p = np.polyfit(np.log2(f[m]), 10 * np.log10(P[m]), 1)
    return p[0]

def thermal_noise(n, sr=SR, flicker_corner=None, flicker_gain=1.0):
    """Johnson floor = white. Optional flicker = true 1/f (-3 dB/oct) shaped below the corner."""
    white = rng.standard_normal(n)
    if flicker_corner is None:
        return white
    # 1/f via frequency-domain shaping of an independent white source
    W = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / sr)
    f[0] = f[1]
    shape = np.where(f < flicker_corner, np.sqrt(flicker_corner / f), 1.0)  # PSD ~ 1/f below corner, 1 above
    shape = shape * 1.0
    # flicker adds to white: white power 1, flicker power = gain*(corner/f - 1)+ below corner
    Fl = W * np.sqrt(np.maximum(flicker_gain * (flicker_corner / f - 1), 0))
    Fl[f >= flicker_corner] = 0
    flick = np.fft.irfft(Fl, n)
    return white + flick

def pink_noise(n):
    W = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR); f[0] = f[1]
    return np.fft.irfft(W / np.sqrt(f), n)

def rms_scale(x, dbfs):
    return x / np.sqrt(np.mean(x**2)) * 10 ** (dbfs / 20)

# ---------- TEST 1: THD / harmonic profile ----------
t = np.arange(SR * 2) / SR
sine = 0.5 * np.sin(2 * np.pi * 1000 * t)
print("=== TEST 1: Vantage Matrix harmonics (1 kHz, 0.5 amp, a=2.0, bias=0.3) ===")
res = {}
res["symmetric tanh (bias=0)"] = thd_report(vantage_curve_only(sine, bias=0.0), tag="sym tanh ")
res["asym tanh (curve only)"] = thd_report(vantage_curve_only(sine, bias=0.3), tag="asym tanh")
res["asym + 8bit + /4 decim"] = thd_report(vantage(sine), tag="full chain")
res["asym + 8bit (no decim)"] = thd_report(vantage(sine, decim=1), tag="8bit only ")
res["asym + decim only"] = thd_report(vantage(sine, bit_depth=None), tag="decim only")

fig, ax = plt.subplots(2, 1, figsize=(10, 7))
for name in ["symmetric tanh (bias=0)", "asym tanh (curve only)", "asym + 8bit + /4 decim"]:
    _, _, f, S = res[name]
    ax[0].semilogx(f[1:], 10 * np.log10(S[1:] / S.max() + 1e-20), label=name, lw=0.8)
ax[0].set_xlim(100, 22050); ax[0].set_ylim(-140, 5); ax[0].legend(); ax[0].grid(True, which="both", alpha=.3)
ax[0].set_title("Test 1: Vantage Matrix spectra (1 kHz in)"); ax[0].set_ylabel("dB rel. peak")

# ---------- TEST 2: spectral slope ----------
print("\n=== TEST 2: Jowiphon spectral slope (10 s) ===")
N = SR * 10
gens = {
    "white (Johnson only)": thermal_noise(N),
    "Jowiphon (white + 1/f below 100 Hz)": thermal_noise(N, flicker_corner=100, flicker_gain=1.0),
    "pink (reference reject)": pink_noise(N),
    "earlier script: white+1-pole LP @100Hz": None,
}
w = rng.standard_normal(N)
b, a = signal.butter(1, 100 / (SR / 2))
y = w + signal.lfilter(b, a, w)
gens["earlier script: white+1-pole LP @100Hz"] = y
out = {}
for name, x in gens.items():
    f, P = signal.welch(x, SR, nperseg=2**16)
    s_hi = slope_db_oct(f, P, 300, 10000)
    s_lo = slope_db_oct(f, P, 5, 60)
    s_all = slope_db_oct(f, P, 5, 20000)
    out[name] = (f, P)
    verdict = "REJECT (pink-like)" if abs(s_all + 3) < 0.75 else "pass"
    print(f"{name:42s} slope >300Hz={s_hi:+.2f}  5-60Hz={s_lo:+.2f}  overall={s_all:+.2f} dB/oct  -> {verdict}")
for name, (f, P) in out.items():
    ax[1].semilogx(f[1:], 10 * np.log10(P[1:]), label=name, lw=0.8)
ax[1].set_xlim(5, 22050); ax[1].legend(fontsize=7); ax[1].grid(True, which="both", alpha=.3)
ax[1].set_title("Test 2: noise PSD"); ax[1].set_xlabel("Hz"); ax[1].set_ylabel("dB/Hz (rel.)")
plt.tight_layout(); plt.savefig("aura_v_tests_1_2.png", dpi=130); plt.close()

# ---------- TEST 3: quantization masking ----------
print("\n=== TEST 3: Quantization masking (1 kHz @ -60 dBFS peak) ===")
n = SR * 4
tt = np.arange(n) / SR
quiet = 10 ** (-60 / 20) * np.sin(2 * np.pi * 1000 * tt)
noise = rms_scale(thermal_noise(n, flicker_corner=100), -64)   # -64 dBFS RMS

def quant(x, bits):
    q = 2 ** (bits - 1)
    return np.round(x * q) / q

def analyze(y, tag):
    f, P = signal.welch(y, SR, nperseg=2**15)
    bin1k = np.argmin(abs(f - 1000))
    sig = 10 * np.log10(P[bin1k - 1:bin1k + 2].sum())
    m = (abs(f - 1000) > 200) & (f > 200) & (f < 15000)
    floor = 10 * np.log10(np.median(P[m]))
    print(f"{tag:46s} peak={sig:6.1f} dB  median floor={floor:6.1f} dB  SNR~{sig-floor:5.1f} dB")
    return f, P

for bits in (8, 16):
    lsb = 2 ** -(bits - 1)
    print(f"-- {bits}-bit: LSB = {20*np.log10(lsb):.1f} dBFS ; signal peak = {quiet.max()/lsb:.2f} LSB")
    qn = quant(quiet, bits)
    print(f"   nonzero output samples: {np.count_nonzero(qn)}/{n}")
    analyze(qn, f"{bits}-bit, no dither")
    analyze(qn + noise, f"{bits}-bit, noise added AFTER quantizer")
    analyze(quant(quiet + noise, bits), f"{bits}-bit, noise added BEFORE quantizer")

# reference: ProDither level must sit below Jowiphon floor
print("\nProDither-below-floor gate: ProDither not present in this harness -> UNTESTED")
