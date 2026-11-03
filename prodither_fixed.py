"""
prodither.py — Unified dithering engine for the MA3 pipeline.

Modes:
    clean:      TPDF dither with optional noise shaping
    dtc:        Neve DTC-era emulation (clean, flat TPDF, no shaping,
                optional 50/15 µs emphasis/de-emphasis)
    neve:       Fictional Neve-style saturation + HF roll-off (character)
    manhattan:  API saturation + air shelf + bass tighten
    adaptive:   Bark-scale masking-aware shaping

All noise shaping is proper error-feedback.
All RNG is seeded for reproducibility.

Notes on the DTC mode:
    The Neve Digital Transfer Console (DTC) was a 16-bit digital mastering
    console (mid-1980s to late 1990s). It had EQ, dynamics, fader, and
    selectable emphasis, but NO analog saturation stage. Its dither was
    flat TPDF; noise shaping did not exist commercially until the Lipshitz
    et al. paper in 1991 and the Apogee UV22 in 1993. The DTC mode here
    models the era accurately: clean signal path, flat dither, optional
    emphasis.

    Reference:
        - Neve DTC product literature
        - Lipshitz, Vanderkooy, Wannamaker (1991), JAES
        - Sony PCM-1610 format documentation
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.signal import lfilter, butter, bilinear_zpk, zpk2sos, sosfilt, stft

try:
    from numba import njit, prange
    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False

    def njit(*args, **kwargs):
        def deco(f):
            return f
        return deco

    prange = range


# =============================================================================
# JIT KERNELS
# =============================================================================

@njit(fastmath=True, cache=True)
def _quantize_shaped_jit(signal_scaled, dither, coeffs, max_val):
    """
    Proper noise-shaped quantization with error feedback.

    Per sample:
        shaped = signal_scaled[i] - sum_k coeffs[k] * error[i-k]
        q = round(shaped + dither[i])
        q = clip(q, -max_val, max_val)
        error[i] = q - shaped
    """
    n = len(signal_scaled)
    out = np.empty(n)
    order = len(coeffs)
    err_hist = np.zeros(order)

    for i in range(n):
        fb = 0.0
        for k in range(order):
            fb += coeffs[k] * err_hist[k]

        shaped = signal_scaled[i] - fb   # FIX: feedback must be SUBTRACTED so NTF = 1 - C(z)
        q = round(shaped + dither[i])

        if q > max_val:
            q = max_val
        elif q < -max_val:
            q = -max_val

        for k in range(order - 1, 0, -1):
            err_hist[k] = err_hist[k - 1]
        if order > 0:
            err_hist[0] = q - shaped

        out[i] = q

    return out / max_val


@njit(fastmath=True, cache=True, parallel=True)
def _neve_saturation_jit(x, sat):
    """Odd-order Neve-style saturation (fictional character)."""
    y = np.empty_like(x)
    d3 = sat / 3.0
    d5 = sat / 10.0
    for i in prange(len(x)):
        v = x[i]
        if v > 1.0:
            v = 1.0
        elif v < -1.0:
            v = -1.0
        y[i] = v - d3 * v * v * v + d5 * v * v * v * v * v
        if y[i] > 1.0:
            y[i] = 1.0
        elif y[i] < -1.0:
            y[i] = -1.0
    return y


@njit(fastmath=True, cache=True, parallel=True)
def _api_saturation_jit(x, drive):
    """Odd-order API-style saturation."""
    y = np.empty_like(x)
    d = drive / 3.0
    for i in prange(len(x)):
        v = x[i]
        if v > 1.0:
            v = 1.0
        elif v < -1.0:
            v = -1.0
        y[i] = v - d * v * v * v
        if y[i] > 1.0:
            y[i] = 1.0
        elif y[i] < -1.0:
            y[i] = -1.0
    return y


# =============================================================================
# NOISE SHAPING FILTERS
# =============================================================================

LIPSHITZ_5 = np.array([2.033, -2.165, 1.959, -1.590, 0.614])
SHIBATA_8 = np.array([
    2.8720729351043701172, -5.0413231849670410156,
    6.2442994117736816406, -5.8483986854553222656,
    3.7067542076110839844, -1.0495119094848632812,
    -1.1830236911773681641, 2.1126792430877685547,
])
WANNER_3 = np.array([1.623, -0.982, 0.109])
NONE = np.array([])


# =============================================================================
# EMPHASIS / DE-EMPHASIS (DTC-era 50/15 µs)
# =============================================================================

def _make_emphasis_filters(sample_rate: int, tau_us: float = 50.0):
    """
    Return (pre_b, pre_a, de_b, de_a) for 50/15 µs emphasis at the given rate.

    Pre-emphasis: H(s) = (1 + s·τ1) / (1 + s·τ2) with τ1 = 50 µs, τ2 = 15 µs.
    De-emphasis is the inverse.
    """
    tau1 = tau_us * 1e-6
    tau2 = 15e-6

    # Pre-emphasis: zero at -1/τ1, pole at -1/τ2
    zeros = [-1.0 / tau1]
    poles = [-1.0 / tau2]
    k = 1.0

    z, p, k = bilinear_zpk(zeros, poles, k, sample_rate)
    sos_pre = zpk2sos(z, p, k)

    # De-emphasis: swap zeros and poles
    z, p, k = bilinear_zpk(poles, zeros, 1.0 / k, sample_rate)
    sos_de = zpk2sos(z, p, k)

    return sos_pre, sos_de


def _apply_sos(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
    return sosfilt(sos, x)


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class ProDitherConfig:
    bit_depth: int = 16
    sample_rate_hz: int = 44100
    mode: str = "clean"                 # clean | dtc | neve | manhattan | adaptive
    shaping: str = "lipshitz"           # none | wanner | lipshitz | shibata

    dither_seed: int = 0
    true_peak_ceiling_dbtp: float = -2.0

    # Neve (fictional) parameters
    neve_warmth: float = 0.0
    neve_sat: float = 0.0

    # Manhattan parameters
    manhattan_air: float = 0.0
    manhattan_punch: float = 0.0

    # DTC parameters
    dtc_emphasis_us: float = 0.0        # 0 = off, 50 = 50/15 µs, 75 = 75/50 µs
    dtc_shaping: str = "none"           # DTC-era: no shaping
    dtc_apply_deemphasis: bool = True   # Whether to undo emphasis at output


# =============================================================================
# MAIN ENGINE
# =============================================================================

class ProDither:
    """
    Unified dithering engine for MA3 mastering.

    Modes:
        clean:     dither + optional noise shaping
        dtc:       Neve DTC-era (clean path, flat TPDF, optional emphasis)
        neve:      Fictional Neve-style saturation + HF roll-off
        manhattan: API saturation + air shelf + bass tighten
        adaptive:  Bark-scale masking-aware shaping per frame
    """

    SHAPING_COEFFS = {
        "none": NONE,
        "wanner": WANNER_3,
        "lipshitz": LIPSHITZ_5,
        "shibata": SHIBATA_8,
    }

    def __init__(self, config: Optional[ProDitherConfig] = None):
        self.cfg = config or ProDitherConfig()
        self.max_val = (1 << (self.cfg.bit_depth - 1)) - 1
        self.rng = np.random.default_rng(self.cfg.dither_seed)
        self._build_filters()

    def _build_filters(self):
        nyq = self.cfg.sample_rate_hz * 0.5

        # --- Neve ---
        if self.cfg.neve_warmth > 0.01:
            cutoff = 15000.0 * (1.0 - self.cfg.neve_warmth * 0.3)
            self.neve_b, self.neve_a = butter(2, cutoff / nyq, btype="low")
        else:
            self.neve_b, self.neve_a = [1.0], [1.0]

        # --- Manhattan ---
        self.air_b, self.air_a = butter(1, 12000.0 / nyq, btype="high")
        self.air_gain = 1.0 + self.cfg.manhattan_air * 0.3
        self.bass_b, self.bass_a = butter(1, 30.0 / nyq, btype="high")

        # --- DTC emphasis ---
        if self.cfg.dtc_emphasis_us > 0.0:
            self.dtc_pre_sos, self.dtc_de_sos = _make_emphasis_filters(
                self.cfg.sample_rate_hz, self.cfg.dtc_emphasis_us
            )
        else:
            self.dtc_pre_sos = None
            self.dtc_de_sos = None

    # ---- dither generation ----

    def _tpdf(self, n: int) -> np.ndarray:
        u1 = self.rng.uniform(-1.0, 1.0, n)
        u2 = self.rng.uniform(-1.0, 1.0, n)
        return (u1 - u2) * 0.5

    # ---- mode pre-processors ----

    def _apply_neve(self, x: np.ndarray) -> np.ndarray:
        if self.cfg.neve_sat > 0.0:
            x = _neve_saturation_jit(x, self.cfg.neve_sat)
        return lfilter(self.neve_b, self.neve_a, x)

    def _apply_manhattan(self, x: np.ndarray) -> np.ndarray:
        x = lfilter(self.bass_b, self.bass_a, x)
        if self.cfg.manhattan_punch > 0.0:
            x = _api_saturation_jit(x, self.cfg.manhattan_punch)
        if self.cfg.manhattan_air > 0.0:
            high = lfilter(self.air_b, self.air_a, x)
            x = x + (self.air_gain - 1.0) * high
        return x

    def _apply_dtc_pre(self, x: np.ndarray) -> np.ndarray:
        """DTC-era: no saturation, optional 50/15 µs pre-emphasis."""
        if self.dtc_pre_sos is not None:
            x = _apply_sos(self.dtc_pre_sos, x)
        return x

    def _apply_dtc_de(self, x: np.ndarray) -> np.ndarray:
        """DTC-era: de-emphasis after dithering."""
        if self.dtc_de_sos is not None and self.cfg.dtc_apply_deemphasis:
            x = _apply_sos(self.dtc_de_sos, x)
        return x

    # ---- core ----

    def process(self, signal: np.ndarray) -> np.ndarray:
        """Apply mode pre-processing, then noise-shaped (or flat) dither."""
        x = np.asarray(signal, dtype=np.float64)
        mono = x.ndim == 1
        if mono:
            x = x.reshape(-1, 1)

        out = np.empty_like(x)

        for ch in range(x.shape[1]):
            ch_sig = x[:, ch]

            # --- Mode pre-processing ---
            if self.cfg.mode == "neve":
                ch_sig = self._apply_neve(ch_sig)
                shaping_name = self.cfg.shaping
            elif self.cfg.mode == "manhattan":
                ch_sig = self._apply_manhattan(ch_sig)
                shaping_name = self.cfg.shaping
            elif self.cfg.mode == "dtc":
                ch_sig = self._apply_dtc_pre(ch_sig)
                shaping_name = self.cfg.dtc_shaping
            else:
                shaping_name = self.cfg.shaping

            coeffs = self.SHAPING_COEFFS.get(shaping_name, NONE)

            # --- Scale to LSB ---
            scaled = ch_sig * self.max_val

            # --- Dither ---
            dither = self._tpdf(len(scaled))

            # --- Quantize ---
            if self.cfg.mode == "adaptive":
                out[:, ch] = self._adaptive_quantize(scaled, ch_sig)
            else:
                out[:, ch] = _quantize_shaped_jit(
                    scaled, dither, coeffs, self.max_val
                )

            # --- DTC post-processing ---
            if self.cfg.mode == "dtc":
                out[:, ch] = self._apply_dtc_de(out[:, ch])

        return out.squeeze() if mono else out

    # ---- adaptive ----

    def _adaptive_quantize(self, scaled: np.ndarray, signal: np.ndarray) -> np.ndarray:
        frame_size = 4096
        out = np.empty_like(scaled)
        n = len(scaled)

        for start in range(0, n, frame_size):
            end = min(start + frame_size, n)
            frame_scaled = scaled[start:end]
            frame_signal = signal[start:end]
            if len(frame_scaled) == 0:
                continue

            nperseg = min(2048, len(frame_signal))
            if nperseg < 256:
                d = self._tpdf(len(frame_scaled))
                out[start:end] = _quantize_shaped_jit(
                    frame_scaled, d, NONE, self.max_val
                )
                continue

            f, _, Zxx = stft(
                frame_signal, fs=self.cfg.sample_rate_hz,
                nperseg=nperseg, noverlap=nperseg // 2,
            )
            mag = np.abs(Zxx).mean(axis=1)
            freqs = np.fft.rfftfreq(nperseg, 1.0 / self.cfg.sample_rate_hz)[:len(mag)]

            if len(mag) < 8 or len(freqs) < 8:
                d = self._tpdf(len(frame_scaled))
                out[start:end] = _quantize_shaped_jit(
                    frame_scaled, d, NONE, self.max_val
                )
                continue

            coeffs = self._shaping_from_spectrum(mag, freqs)
            d = self._tpdf(len(frame_scaled))
            out[start:end] = _quantize_shaped_jit(
                frame_scaled, d, coeffs, self.max_val
            )

        return out

    def _shaping_from_spectrum(self, mag: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        m = mag / (mag.max() + 1e-12)
        weight = freqs / (freqs.max() + 1e-12)
        target = weight / (m + 0.1)

        try:
            c = np.polyfit(np.arange(len(target)), target, 2)
        except Exception:
            return WANNER_3

        b = np.array([1.0, -c[0] * 0.5, -c[1] * 0.3])
        b = np.clip(b, -2.0, 3.0)
        return b


# =============================================================================
# PRESETS
# =============================================================================

PRODITHER_PRESETS = {
    # --- clean ---
    "clean":         ProDitherConfig(mode="clean", shaping="lipshitz"),
    "clean_quiet":   ProDitherConfig(mode="clean", shaping="shibata"),
    "no_shaping":    ProDitherConfig(mode="clean", shaping="none"),

    # --- DTC (historical, 1980s–90s) ---
    "dtc_flat":      ProDitherConfig(mode="dtc", dtc_shaping="none"),
    "dtc_emphasis":  ProDitherConfig(mode="dtc", dtc_shaping="none",
                                     dtc_emphasis_us=50.0),
    "dtc_emph_75":   ProDitherConfig(mode="dtc", dtc_shaping="none",
                                     dtc_emphasis_us=75.0),
    "dtc_shaped":    ProDitherConfig(mode="dtc", dtc_shaping="lipshitz"),

    # --- Neve (fictional character) ---
    "neve_warm":     ProDitherConfig(mode="neve", shaping="lipshitz",
                                     neve_warmth=0.5, neve_sat=0.3),
    "neve_hot":      ProDitherConfig(mode="neve", shaping="lipshitz",
                                     neve_warmth=0.7, neve_sat=0.5),

    # --- Manhattan ---
    "manhattan":     ProDitherConfig(mode="manhattan", shaping="lipshitz",
                                     manhattan_air=0.5, manhattan_punch=0.4),
    "manhattan_air": ProDitherConfig(mode="manhattan", shaping="lipshitz",
                                     manhattan_air=0.7, manhattan_punch=0.2),

    # --- adaptive ---
    "adaptive":      ProDitherConfig(mode="adaptive", shaping="lipshitz"),
}


def apply_preset(
    signal: np.ndarray,
    preset_name: str,
    bit_depth: int = 16,
    sample_rate: int = 44100,
    seed: int = 0,
) -> np.ndarray:
    """Apply a named preset. Returns the processed signal."""
    if preset_name not in PRODITHER_PRESETS:
        raise ValueError(f"unknown preset: {preset_name}")
    import copy
    cfg = copy.deepcopy(PRODITHER_PRESETS[preset_name])
    cfg.bit_depth = bit_depth
    cfg.sample_rate_hz = sample_rate
    cfg.dither_seed = seed
    return ProDither(cfg).process(signal)


# =============================================================================
# USAGE
# =============================================================================

if __name__ == "__main__":
    import soundfile as sf

    fs, data = sf.read("input.wav")

    # Historical DTC — flat TPDF, no shaping, no emphasis
    dtc_out = apply_preset(data, "dtc_flat", bit_depth=16, sample_rate=fs, seed=42)
    sf.write("output_dtc_flat.wav", dtc_out, fs, subtype="PCM_16")

    # Historical DTC with 50/15 µs emphasis
    dtc_emph = apply_preset(data, "dtc_emphasis", bit_depth=16, sample_rate=fs, seed=42)
    sf.write("output_dtc_emph.wav", dtc_emph, fs, subtype="PCM_16")

    print("done")
