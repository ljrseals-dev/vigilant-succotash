#!/usr/bin/env python3
"""
jowiphon_noise — generate a noise bed and gain-stage it to a broadcast loudness standard.

Signal chain:
    [SOURCE] → [SHAPE] → [HPF + FADER] → [MEASURE] → [GAIN] → [CEIL] → OUT
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants / standards
# ---------------------------------------------------------------------------

STANDARDS = {
    "ebu_r128":     {"label": "EBU R 128",              "region": "Europe / International", "lufs": -23.0, "tp": -1.0},
    "atsc_a85":     {"label": "ATSC A/85",              "region": "North America",          "lufs": -24.0, "tp": -2.0},
    "arib_tr_b32":  {"label": "ARIB TR-B32",            "region": "Japan",                  "lufs": -24.0, "tp": -1.0},
    "as11_dpp":     {"label": "AS-11 / DPP",            "region": "UK",                     "lufs": -23.0, "tp": -2.0},
    "bbc":          {"label": "BBC",                    "region": "UK",                     "lufs": -23.0, "tp": -1.0},
    "nordic":       {"label": "Nordic (NRK/SVT)",       "region": "Nordic",                 "lufs": -23.0, "tp": -1.0},
    "nfl":          {"label": "NFL / sports",           "region": "US",                     "lufs": -24.0, "tp": -2.0},
    "podcast":      {"label": "Podcast (platform-safe)","region": "Global",                 "lufs": -16.0, "tp": -1.0},
    "podcast_mono": {"label": "Podcast (mono)",         "region": "Global",                 "lufs": -19.0, "tp": -1.0},
    "streaming":    {"label": "Streaming (music / VOD)","region": "Global",                 "lufs": -14.0, "tp": -1.0},
    "youtube":      {"label": "YouTube",                "region": "Global",                 "lufs": -14.0, "tp": -1.0},
    "apple_music":  {"label": "Apple Music",            "region": "Global",                 "lufs": -16.0, "tp": -1.0},
    "cinema":       {"label": "Cinema / theatrical stem","region": "Global",                "lufs": -27.0, "tp": -1.0},
    "itu_bs1770":   {"label": "ITU-R BS.1770-4",        "region": "International",          "lufs": -23.0, "tp": -1.0},
}

DEFAULT_TARGET_LUFS = -23.0
DEFAULT_CEILING_DBTP = -1.0
DEFAULT_SR = 48000
DEFAULT_SECONDS = 10.0
DEFAULT_HPF_HZ = 38.0
DEFAULT_SEED = 0

# Shaper normalising constants so output RMS matches shape_none.
# Input u ~ Uniform[-1, 1] has RMS = 1/√3.
# Shaped RMS values:
#   sqrt:    RMS(√|u|) = √(E[|u|]) = √(1/2)
#   squared: RMS(u²)   = √(E[u⁴]) = √(1/5)
# Factor = input_RMS / shaped_RMS so the product lands back at 1/√3:
#   sqrt    → √(1/3) / √(1/2) = √(2/3) ≈ 0.816
#   squared → √(1/3) / √(1/5) = √(5/3) ≈ 1.291
SQRT_2_OVER_3 = math.sqrt(2.0 / 3.0)
SQRT_5_OVER_3 = math.sqrt(5.0 / 3.0)


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

def white_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Flat spectrum, uniform on [-1, 1]."""
    return rng.uniform(-1.0, 1.0, size=n).astype(np.float64)


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Approximate −3 dB/oct via Voss-McCartney-style filtered white."""
    # Simple 1/f approximation: successive integration + high-pass-ish mix
    white = rng.standard_normal(n).astype(np.float64)
    # Cascade of first-order filters approximating pink
    b0 = 0.99765
    b1 = 0.0990460
    b2 = 0.0324985
    b3 = 0.00385336
    b4 = 0.000512
    y = np.zeros(n, dtype=np.float64)
    x1 = x2 = x3 = x4 = x5 = x6 = 0.0
    for i in range(n):
        white_i = white[i]
        x1 = 0.99886 * x1 + white_i * 0.0555179
        x2 = 0.99332 * x2 + white_i * 0.0750759
        x3 = 0.96900 * x3 + white_i * 0.1538520
        x4 = 0.86650 * x4 + white_i * 0.3104856
        x5 = 0.55000 * x5 + white_i * 0.5329522
        x6 = -0.7616 * x6 - white_i * 0.0168980
        y[i] = x1 + x2 + x3 + x4 + x5 + x6 + white_i * 0.5362
    # Normalise to roughly unit peak / RMS comparable to white
    y *= 0.11
    # Soft clip to [-1, 1] range similar to other sources
    peak = np.max(np.abs(y)) + 1e-12
    return (y / peak).astype(np.float64)


def brown_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """−6 dB/oct: leaky integrator of white + DC blocker."""
    white = rng.standard_normal(n).astype(np.float64)
    # Leaky integrator (alpha close to 1)
    alpha = 0.995
    y = np.zeros(n, dtype=np.float64)
    acc = 0.0
    for i in range(n):
        acc = alpha * acc + white[i]
        y[i] = acc
    # DC blocker (high-pass at very low frequency)
    R = 0.995
    out = np.zeros(n, dtype=np.float64)
    prev_x = prev_y = 0.0
    for i in range(n):
        out[i] = y[i] - prev_x + R * prev_y
        prev_x = y[i]
        prev_y = out[i]
    # Scale to roughly unit peak
    peak = np.max(np.abs(out)) + 1e-12
    return (out / peak).astype(np.float64)


def thermal_noise(n: int, sr: int, rng: np.random.Generator,
                  corner_hz: float = 100.0, f_min_hz: float = 1.0) -> np.ndarray:
    """
    Johnson-Nyquist floor + flicker (1/f) below a corner.

    Target one-sided PSD:   S(f) = 1            for f >= corner   (white, 0 dB/oct)
                            S(f) = corner / f   for f <  corner   (-3 dB/oct)
    Built by shaping Gaussian white noise in the frequency domain, so the white
    floor and the 1/f segment join continuously at the corner. f_min_hz bounds
    the 1/f rise so subsonic energy stays finite.
    """
    white = rng.standard_normal(n)
    X = np.fft.rfft(white)
    f = np.fft.rfftfreq(n, 1.0 / sr)
    f_eff = np.maximum(f, f_min_hz)
    shape = np.sqrt(np.maximum(1.0, corner_hz / f_eff))
    y = np.fft.irfft(X * shape, n)
    peak = np.max(np.abs(y)) + 1e-12
    return (y / peak).astype(np.float64)


def photocell_noise(n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    """
    Model of a selenium photocell under varying illumination.

    Three physical components:
      1. Pink grain floor  – continuous microscopic resistance fluctuations
      2. Carrier drift     – slow random walk of mean resistance
      3. Discrete events   – sparse, decaying clicks when grains change state
    """
    t = np.arange(n, dtype=np.float64) / sr

    # 1. Pink grain floor
    grain = pink_noise(n, rng) * 0.35

    # 2. Carrier drift (very slow random walk, low-pass)
    walk = np.cumsum(rng.standard_normal(n)) * 0.0008
    # Smooth further
    for _ in range(3):
        walk = np.convolve(walk, np.ones(5) / 5.0, mode="same")
    drift = walk * 0.15

    # 3. Sparse decaying events
    event_rate = 8.0  # average events per second
    n_events = max(1, int(event_rate * n / sr))
    event_times = rng.uniform(0, n / sr, size=n_events)
    events = np.zeros(n, dtype=np.float64)
    for et in event_times:
        idx = int(et * sr)
        if idx >= n:
            continue
        # exponential decay, random polarity and amplitude
        length = min(int(0.08 * sr), n - idx)
        if length < 2:
            continue
        amp = rng.uniform(0.15, 0.55) * rng.choice([-1.0, 1.0])
        tau = rng.uniform(0.008, 0.035)
        decay = amp * np.exp(-np.arange(length) / (tau * sr))
        events[idx:idx + length] += decay

    out = grain + drift + events
    peak = np.max(np.abs(out)) + 1e-12
    return (out / peak).astype(np.float64)


# ---------------------------------------------------------------------------
# Shapers (RMS-preserving)
# ---------------------------------------------------------------------------

def shape_none(x: np.ndarray) -> np.ndarray:
    return x.copy()


def shape_sqrt(x: np.ndarray) -> np.ndarray:
    """Compress dynamic range; quiet parts rise relative to peaks."""
    return np.copysign(np.sqrt(np.abs(x)), x) * SQRT_2_OVER_3


def shape_squared(x: np.ndarray) -> np.ndarray:
    """Expand dynamic range; quiet parts fall toward zero."""
    return np.copysign(x * x, x) * SQRT_5_OVER_3


SHAPERS = {
    "none": shape_none,
    "sqrt": shape_sqrt,
    "squared": shape_squared,
}


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

class Biquad:
    """Direct-form-II transposed biquad, block processing."""

    def __init__(self, b: np.ndarray, a: np.ndarray):
        self.b = np.asarray(b, dtype=np.float64)
        self.a = np.asarray(a, dtype=np.float64)
        self.z1 = 0.0
        self.z2 = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        y = np.empty_like(x)
        b0, b1, b2 = self.b
        a1, a2 = self.a[1], self.a[2]  # a0 normalised to 1
        z1, z2 = self.z1, self.z2
        for i in range(len(x)):
            xi = x[i]
            yi = b0 * xi + z1
            z1 = b1 * xi - a1 * yi + z2
            z2 = b2 * xi - a2 * yi
            y[i] = yi
        self.z1, self.z2 = z1, z2
        return y

    def reset(self) -> None:
        self.z1 = self.z2 = 0.0


def make_highpass(fc: float, sr: int) -> Biquad:
    """Butterworth 2nd-order high-pass."""
    w0 = 2.0 * math.pi * fc / sr
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)
    alpha = sin_w0 / (2.0 * math.sqrt(2.0))  # Q = 1/√2
    b0 = (1.0 + cos_w0) / 2.0
    b1 = -(1.0 + cos_w0)
    b2 = (1.0 + cos_w0) / 2.0
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha
    b = np.array([b0 / a0, b1 / a0, b2 / a0])
    a = np.array([1.0, a1 / a0, a2 / a0])
    return Biquad(b, a)


def make_k_weighting(sr: int) -> Tuple[Biquad, Biquad]:
    """
    BS.1770 K-weighting: pre-filter + RLB stage.
    Coefficients computed for the actual sample rate.
    """
    # Stage 1 – high-shelf pre-filter (approx 4 kHz shelf)
    # Reference coefficients for 48 kHz, bilinear-warped for other rates
    f0 = 1681.974450955533
    G = 3.999843853973347
    Q = 0.7071752369554196
    K = math.tan(math.pi * f0 / sr)
    Vh = 10 ** (G / 20.0)
    Vb = Vh ** 0.4996667741545416
    a0_ = 1.0 + K / Q + K * K
    b0 = (Vh + Vb * K / Q + K * K) / a0_
    b1 = 2.0 * (K * K - Vh) / a0_
    b2 = (Vh - Vb * K / Q + K * K) / a0_
    a1 = 2.0 * (K * K - 1.0) / a0_
    a2 = (1.0 - K / Q + K * K) / a0_
    pre = Biquad(np.array([b0, b1, b2]), np.array([1.0, a1, a2]))

    # Stage 2 – RLB high-pass
    f0 = 38.13547087602444
    Q = 0.5003270373238773
    K = math.tan(math.pi * f0 / sr)
    a0_ = 1.0 + K / Q + K * K
    b0 = 1.0
    b1 = -2.0
    b2 = 1.0
    a1 = 2.0 * (K * K - 1.0) / a0_
    a2 = (1.0 - K / Q + K * K) / a0_
    # Standard BS.1770 RLB numerator is [1, -2, 1] (NOT divided by a0).
    rlb = Biquad(np.array([b0, b1, b2]), np.array([1.0, a1, a2]))
    return pre, rlb


# ---------------------------------------------------------------------------
# Loudness measurement (BS.1770)
# ---------------------------------------------------------------------------

def integrated_lufs(x: np.ndarray, sr: int) -> float:
    """
    Two-pass gated integrated loudness per BS.1770-4.
    Returns LUFS, or -inf for silence / undefined.
    """
    if len(x) == 0:
        return float("-inf")

    pre, rlb = make_k_weighting(sr)
    y = rlb.process(pre.process(x.astype(np.float64)))

    block_ms = 400
    hop_ms = 100
    block = int(sr * block_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    if block < 1:
        block = 1
    if hop < 1:
        hop = 1

    n_blocks = 1 + max(0, (len(y) - block) // hop)
    if n_blocks < 1:
        return float("-inf")

    powers = []
    for i in range(n_blocks):
        start = i * hop
        end = start + block
        if end > len(y):
            break
        seg = y[start:end]
        p = np.mean(seg * seg)
        powers.append(p)

    if not powers:
        return float("-inf")

    powers = np.asarray(powers, dtype=np.float64)
    # Absolute gate −70 LUFS
    abs_thresh = 10 ** ((-70.0 - 0.691) / 10.0)  # power domain
    gated = powers[powers > abs_thresh]
    if len(gated) == 0:
        return float("-inf")

    # Relative gate −10 LU relative to gated mean
    mean_power = np.mean(gated)
    rel_thresh = mean_power * 10 ** (-10.0 / 10.0)
    gated2 = powers[powers > rel_thresh]
    if len(gated2) == 0:
        return float("-inf")

    lufs = -0.691 + 10.0 * math.log10(np.mean(gated2) + 1e-30)
    return float(lufs)


def _sinc_kernel(oversample: int, half_width: int = 16) -> np.ndarray:
    """Windowed-sinc low-pass for polyphase upsampling (Kaiser-ish raised-cosine)."""
    # Taps at intervals of 1/oversample relative to original sample rate
    n = np.arange(-half_width * oversample, half_width * oversample + 1, dtype=np.float64)
    # Normalized frequency cutoff at Nyquist of original (π/oversample in upsampled domain)
    x = n / oversample
    # sinc
    kernel = np.sinc(x)
    # Raised-cosine / Hann window
    window = 0.5 + 0.5 * np.cos(math.pi * n / (half_width * oversample))
    kernel *= window
    # Unity gain at DC
    kernel /= np.sum(kernel) / oversample
    return kernel


def true_peak_dbtp(x: np.ndarray, sr: int, oversample: int = 4) -> float:
    """
    True-peak estimate via windowed-sinc oversampling (ITU-R BS.1770 style).
    Linear interpolation under-reads inter-sample peaks; this does not.
    """
    if len(x) == 0:
        return float("-inf")
    x = np.asarray(x, dtype=np.float64)
    kernel = _sinc_kernel(oversample)
    # Zero-stuff then convolve (same as polyphase filter)
    n = len(x)
    stuffed = np.zeros(n * oversample, dtype=np.float64)
    stuffed[::oversample] = x
    # Full convolution; we only need the magnitude peak
    up = np.convolve(stuffed, kernel, mode="same")
    peak = float(np.max(np.abs(up)))
    if peak < 1e-30:
        return float("-inf")
    return 20.0 * math.log10(peak)


# ---------------------------------------------------------------------------
# Gain & ceiling
# ---------------------------------------------------------------------------

def _smooth_gain(target_lin: float, n: int, sr: int,
                 attack_ms: float = 1.0, release_ms: float = 40.0) -> np.ndarray:
    """One-pole smoother from 0 → target_lin."""
    att = math.exp(-1.0 / (attack_ms * 0.001 * sr))
    rel = math.exp(-1.0 / (release_ms * 0.001 * sr))
    g = np.empty(n, dtype=np.float64)
    cur = 0.0
    for i in range(n):
        if target_lin > cur:
            cur = att * cur + (1.0 - att) * target_lin
        else:
            cur = rel * cur + (1.0 - rel) * target_lin
        g[i] = cur
    return g


def apply_gain(x: np.ndarray, gain_db: float, sr: int) -> np.ndarray:
    """Apply smoothed linear gain."""
    lin = 10.0 ** (gain_db / 20.0)
    envelope = _smooth_gain(lin, len(x), sr)
    return x * envelope


def apply_ceiling(
    x: np.ndarray,
    ceiling_lin: float,
    sr: int,
) -> Tuple[np.ndarray, bool]:
    """
    True-peak trim: if the oversampled peak exceeds ceiling, apply a constant
    scale so the true peak lands exactly at ceiling. This preserves waveform
    shape (no hard clipping) at the cost of a small integrated-loudness shift
    when the TP constraint is binding. Returns (signal, was_trimmed).
    """
    tp_db = true_peak_dbtp(x, sr)
    if not math.isfinite(tp_db):
        return x.copy(), False
    tp_lin = 10.0 ** (tp_db / 20.0)
    if tp_lin <= ceiling_lin:
        return x.copy(), False
    scale = ceiling_lin / (tp_lin + 1e-12)
    return x * scale, True


# ---------------------------------------------------------------------------
# Main render path
# ---------------------------------------------------------------------------

@dataclass
class RenderResult:
    audio: np.ndarray
    sr: int
    measured_before: float
    measured_after: float
    gain_db: float
    trimmed: bool
    target_lufs: float
    ceiling_dbtp: float


def render(
    kind: str = "pink",
    shape: str = "none",
    seconds: float = DEFAULT_SECONDS,
    sr: int = DEFAULT_SR,
    seed: int = DEFAULT_SEED,
    hpf_hz: float = DEFAULT_HPF_HZ,
    fader_db: float = 0.0,
    target_lufs: float = DEFAULT_TARGET_LUFS,
    ceiling_dbtp: float = DEFAULT_CEILING_DBTP,
    corner_hz: float = 100.0,
) -> RenderResult:
    """
    Full signal chain:
        source → shape → optional HPF + fader → measure → gain → ceiling
    """
    rng = np.random.default_rng(seed)
    n = int(round(seconds * sr))
    if n < 1:
        raise ValueError("duration produces zero samples")

    # Source
    if kind == "white":
        raw = white_noise(n, rng)
    elif kind == "pink":
        raw = pink_noise(n, rng)
    elif kind == "brown":
        raw = brown_noise(n, rng)
    elif kind == "photocell":
        raw = photocell_noise(n, sr, rng)
    elif kind == "thermal":
        raw = thermal_noise(n, sr, rng, corner_hz)
    else:
        raise ValueError(f"unknown kind: {kind}")

    # Shape
    if shape not in SHAPERS:
        raise ValueError(f"unknown shape: {shape}")
    raw = SHAPERS[shape](raw)

    # HPF (skip when fc is already covered by K-weighting RLB ≈ 38 Hz)
    if hpf_hz > 40.0:
        hp = make_highpass(hpf_hz, sr)
        raw = hp.process(raw)
    # else: measurement filter already removes DC / subsonic

    # Fader (pre-gain trim)
    if abs(fader_db) > 1e-6:
        raw = raw * (10.0 ** (fader_db / 20.0))

    # Measure
    measured_before = integrated_lufs(raw, sr)
    if not math.isfinite(measured_before):
        # Silence or undefined – leave at zero gain
        gain_db = 0.0
    else:
        gain_db = target_lufs - measured_before

    # Apply gain (smoothed) then true-peak ceiling trim
    out = apply_gain(raw, gain_db, sr)
    ceiling_lin = 10.0 ** (ceiling_dbtp / 20.0)
    out, trimmed = apply_ceiling(out, ceiling_lin, sr)

    measured_after = integrated_lufs(out, sr)

    return RenderResult(
        audio=out.astype(np.float32),
        sr=sr,
        measured_before=measured_before,
        measured_after=measured_after,
        gain_db=gain_db,
        trimmed=trimmed,
        target_lufs=target_lufs,
        ceiling_dbtp=ceiling_dbtp,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def list_standards() -> None:
    print(f"{'key':<14} {'label':<28} {'region':<22} {'LUFS':>6} {'dBTP':>6}")
    print("-" * 80)
    for k, v in STANDARDS.items():
        print(f"{k:<14} {v['label']:<28} {v['region']:<22} {v['lufs']:>6.1f} {v['tp']:>6.1f}")
    print()
    print("Notes:")
    print("  −24 is the North-American television anchor (ATSC A/85).")
    print("  European broadcast is −23 (EBU R 128).")
    print("  Cinema is genuinely separate at −27 because the room is calibrated.")


def write_wav(path: str, audio: np.ndarray, sr: int) -> None:
    """Minimal WAV writer (float32 PCM) – no external dependency beyond numpy."""
    import wave
    import struct

    # Convert float32 −1…1 to int16 for broad compatibility
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(
        description="Generate a noise bed and gain-stage it to a broadcast loudness standard.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python jowiphon_noise.py --list-standards
  python jowiphon_noise.py --standard ebu_r128 -t 10 -o bed.wav
  python jowiphon_noise.py --standard podcast --kind photocell --shape sqrt -t 15 -o bed.wav
  python jowiphon_noise.py --target -18 --tp -1.5 --kind brown -t 5 -o bed.wav
""",
    )
    p.add_argument("--standard", "-s", choices=list(STANDARDS.keys()),
                   help="delivery standard; sets target and ceiling")
    p.add_argument("--target", type=float, default=None,
                   help="target loudness in LUFS (overrides standard)")
    p.add_argument("--tp", type=float, default=None,
                   help="true-peak ceiling in dBTP (overrides standard)")
    p.add_argument("--kind", "-k", choices=["white", "pink", "brown", "photocell", "thermal"],
                   default="pink", help="noise source (default: pink)")
    p.add_argument("--corner", type=float, default=100.0,
                   help="1/f corner in Hz for --kind thermal (default: 100)")
    p.add_argument("--shape", choices=list(SHAPERS.keys()), default="none",
                   help="waveshaper (default: none)")
    p.add_argument("--hpf", type=float, default=DEFAULT_HPF_HZ,
                   help=f"high-pass frequency in Hz; ≤40 skips filter (default: {DEFAULT_HPF_HZ})")
    p.add_argument("--fader", type=float, default=0.0,
                   help="trim in dB before the gain stage (default: 0)")
    p.add_argument("--seconds", "-t", type=float, default=DEFAULT_SECONDS,
                   help=f"length in seconds (default: {DEFAULT_SECONDS})")
    p.add_argument("--sr", type=int, default=DEFAULT_SR,
                   help=f"sample rate (default: {DEFAULT_SR})")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help="RNG seed for reproducibility (default: 0)")
    p.add_argument("--out", "-o", default="noise.wav",
                   help="output WAV path (default: noise.wav)")
    p.add_argument("--list-standards", action="store_true",
                   help="print the standard table and exit")

    args = p.parse_args(argv)

    if args.list_standards:
        list_standards()
        return 0

    target = DEFAULT_TARGET_LUFS
    ceiling = DEFAULT_CEILING_DBTP
    if args.standard:
        s = STANDARDS[args.standard]
        target = s["lufs"]
        ceiling = s["tp"]
    if args.target is not None:
        target = args.target
    if args.tp is not None:
        ceiling = args.tp

    try:
        result = render(
            kind=args.kind,
            shape=args.shape,
            seconds=args.seconds,
            sr=args.sr,
            seed=args.seed,
            hpf_hz=args.hpf,
            fader_db=args.fader,
            target_lufs=target,
            ceiling_dbtp=ceiling,
            corner_hz=args.corner,
        )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    write_wav(args.out, result.audio, result.sr)

    err = result.measured_after - result.target_lufs if math.isfinite(result.measured_after) else float("nan")
    print(f"wrote {args.out}")
    print(f"  kind={args.kind}  shape={args.shape}  sr={result.sr}  duration={args.seconds:.2f}s")
    print(f"  target={result.target_lufs:.1f} LUFS  ceiling={result.ceiling_dbtp:.1f} dBTP")
    print(f"  before={result.measured_before:.2f} LUFS  gain={result.gain_db:+.2f} dB")
    print(f"  after ={result.measured_after:.2f} LUFS  error={err:+.2f} LU  trimmed={result.trimmed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
