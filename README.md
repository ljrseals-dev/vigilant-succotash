This report provides a detailed technical analysis of the three Python modules—jowiphon_noise.py, prodither_fixed.py, and the integration test aura_v_integration_test.py—which together form a mastering-grade signal chain for noise generation, loudness normalisation, and word-length reduction. The critique focuses on the mathematical correctness, architectural decisions, and the validity of the test gates. The requested "Seals and Windsor" reference could not be located in the literature; the analysis therefore draws on the standard authorities in the field, principally the work of Lipshitz, Vanderkooy, and Wannamaker on minimally audible noise shaping and the ITU-R BS.1770 recommendation for true-peak estimation.

---

1. System Architecture and Signal Flow

The integration test defines a 44.1 kHz chain intended to model a complete delivery path: asym tanh → [AA FIR] → Jowiphon thermal dither → 8-bit quantize → /4 ZOH → Jowiphon BS.1770 gain → true-peak trim → ProDither 16-bit. Two variants are tested: Chain A without an anti-aliasing filter and Chain B with a 511-tap linear-phase FIR at a cutoff of 0.45 \times (f_s / 4).

The chain is not a single continuous graph but a sequence of discrete processing objects. This modularity is a strength for testing, but it introduces two architectural considerations:

1. Latency declaration: The AA FIR in Chain B introduces a group delay of (N-1)/2 = 255 samples. The test correctly prints this latency for graph declaration. However, the mode="same" convolution in aa() centres the filter, meaning the latency is not explicitly compensated before the subsequent 4× decimation. In a real-time graph this would require an explicit delay node; in the offline test it only affects the relative alignment of the reference and the processed signal in the spur gate.
2. State isolation: Each call to front_end and back_end creates fresh random generators from the supplied seeds. This is essential for the determinism gate, but it means the modules themselves do not maintain internal state across calls. The ProDither class, however, does maintain an internal RNG state, which is correctly reset by constructing a new instance per call in back_end.

---

2. jowiphon_noise.py: Noise Generation and Loudness Normalisation

2.1 Noise Sources

The module provides five sources: white, pink, brown, photocell, and thermal. The thermal noise implementation is the most technically interesting:

thermal_noise synthesises a one-sided PSD with a white floor above a corner frequency f_c and a 1/f segment below it: S(f) = 1 for f \ge f_c, and S(f) = f_c / f for f < f_c. The shaping is applied in the frequency domain via an FFT, which produces a stationary Gaussian process with the correct second-order statistics. The f_{\min} parameter bounds the 1/f rise to keep subsonic energy finite.

The photocell source is a phenomenological model combining pink grain, a slow random walk (carrier drift), and sparse decaying events. It is not derived from a physical device model but serves as a plausible "character" source for the noise bed.

2.2 Loudness Measurement: BS.1770 Implementation

integrated_lufs is a two-pass implementation of the BS.1770-4 gated loudness algorithm. The K-weighting filters are computed at the actual sample rate by bilinear warping of the analogue prototype, which is correct. The gating logic follows the standard:

1. Absolute gate: blocks below −70 LUFS are discarded.
2. Relative gate: blocks more than 10 LU below the mean of the absolute-gated blocks are discarded.
3. The final loudness is  -0.691 + 10 \log_{10}(\text{mean power of relative-gated blocks}).

The implementation is faithful. One minor issue is the use of 1e-30 in the logarithm to avoid log(0). This is a pragmatic guard, but it means silence returns a finite value (approximately −300 LUFS) rather than -inf. The render function checks math.isfinite and handles this correctly, so the end-to-end behaviour is robust.

2.3 True-Peak Estimation

true_peak_dbtp uses 4× polyphase oversampling with a windowed-sinc kernel. This aligns with the BS.1770 Annex 2 methodology, which specifies at least 4× oversampling for content up to 48 kHz. The kernel is a raised-cosine windowed sinc with a half-width of 16 original samples, which is generous and provides good stopband attenuation. The convolution is performed in the time domain via zero-stuffing and np.convolve, which is computationally heavier than an FFT-based approach but exact and acceptable for offline use.

2.4 Gain and Ceiling

apply_gain uses a one-pole smoother with separate attack (1 ms) and release (40 ms) constants. This is a reasonable choice for a gain stage that is not intended to be a limiter; it prevents the gain change from introducing a click. The ceiling stage, however, is not a limiter in the conventional sense:

apply_ceiling measures the true peak and, if it exceeds the ceiling, applies a constant scale factor to the entire signal so that the true peak lands exactly at the ceiling. This preserves waveform shape but shifts the integrated loudness. The test correctly accounts for this with the condition or (trimmed and TP <= tp + 0.1).

This is a deliberate design choice: it avoids the distortion of hard clipping at the cost of a loudness error when the true-peak constraint is binding. For a mastering chain, this is defensible, but it must be understood that "true-peak trim" here is not a dynamic process.

---

3. prodither_fixed.py: Dither and Noise Shaping Engine

3.1 Quantisation Kernel

The core quantiser is _quantize_shaped_jit:

```python
shaped = signal_scaled[i] - fb
q = round(shaped + dither[i])
error[i] = q - shaped
```

This implements proper error-feedback noise shaping. The feedback term fb is subtracted from the signal, so the noise transfer function (NTF) is 1 - C(z), where C(z) is the FIR shaping filter. This is the correct sign convention: to shape the error away from sensitive bands, the feedback must be subtracted. The error history is updated with the quantisation error of the shaped signal, which is the standard architecture.

3.2 TPDF Dither

The _tpdf method generates triangular dither by taking the difference of two uniform variates on [-1, 1) and scaling by 0.5. The resulting distribution has a peak deviation of ±1 LSB and a variance of q^2/4, which is the theoretically correct TPDF dither for a quantiser with step size q. This is the appropriate choice for high-quality audio: it fully decorrelates the error from the signal, linearising the quantiser at the cost of a 4.77 dB increase in noise power relative to rounding.

3.3 Noise-Shaping Filters

The module provides three fixed shaping filters:

· Lipshitz 5th-order (LIPSHITZ_5): the coefficients from the seminal paper on minimally audible noise shaping.
· Wannamaker 3rd-order (WANNER_3): a lower-order alternative.
· Shibata 8th-order (SHIBATA_8): a higher-order, aggressive shaper.

The use of the Lipshitz coefficients is historically and technically appropriate. The paper's central thesis—that the noise shaper should approximate the inverse of the 15-phon audibility curve—is the foundation of all modern psychoacoustically optimal dithering. The integration test's gate that the 1–5 kHz shaped floor is at least 10 dB below the flat TPDF floor is a direct empirical validation of this principle.

3.4 DTC Mode and Historical Accuracy

The dtc mode is a historical emulation of the Neve Digital Transfer Console. The docstring correctly notes that the DTC had no noise shaping and used flat TPDF dither. The mode therefore sets dtc_shaping = "none" and optionally applies 50/15 µs emphasis. This is historically accurate: commercial noise shaping did not appear until after the 1991 Lipshitz paper. The emphasis filters are derived via bilinear transformation of the analogue pre-emphasis network, which is the correct approach for a digital emulation.

The neve and manhattan modes are explicitly fictional character models. The neve mode uses odd-order polynomial saturation followed by a low-pass filter; the manhattan mode uses API-style saturation plus an air shelf and bass tighten. These are not claimed to be circuit-accurate, and the docstring is transparent about this.

3.5 Adaptive Mode

The adaptive mode performs a short-time Fourier transform per frame, computes a masking-weighted target spectrum, and fits a second-order polynomial to generate shaping coefficients. This is a simplified version of a Bark-scale masking model. The polynomial fit is clipped to [-2, 3] to prevent instability. While not a full psychoacoustic model, it demonstrates the principle of signal-dependent shaping. A more rigorous implementation would use the actual Bark scale and a proper spreading function, but for a test harness the current approach is sufficient to exercise the code path.

---

4. aura_v_integration_test.py: Test Gates and Validity

The integration test defines five gate categories. Each is examined below for technical validity.

4.1 Spur Gate (8-bit Stage)

The test isolates the 8-bit quantiser by subtracting the output of an unquantised reference chain from the quantised chain. It then counts narrowband spectral lines in the error signal using a local-median prominence detector.

This is a sound methodology for detecting dither effectiveness. The undithered chain must show lines (because the error is correlated with the 1 kHz tone), while the dithered chain must show zero lines at −48 dBFS. The test requires that the undithered case have n0 > 0 as a sensitivity check. This guards against a false pass: if the undithered case also shows no lines, the detector is not sensitive enough, and the gate fails. This is rigorous.

The use of five seeds for the dithered counts (range(5)) checks that the result is not a lucky seed. The gate passes only if the sum of counts across all seeds is zero. This is a strong condition: it requires that the dither reliably bury the spurs for every tested seed.

4.2 Loudness and True-Peak Gates

For each target ( −23 and −14 LUFS) and each chain (A and B), the test checks:

· Loudness error within ±0.5 LU, or the true-peak constraint is binding and the TP is at or below the ceiling.
· True peak is at or below ceiling + 0.1 dB.

The tolerance of ±0.5 LU is reasonable for a 6-second speech-like signal; BS.1770 gating on short segments can exhibit this much variance. The TP tolerance of +0.1 dB accounts for numerical rounding in the oversampled peak estimate. The logical or for the loudness gate correctly handles the case where the constant-scale ceiling trim has shifted the loudness: if the trim is binding, the loudness target is subordinated to the TP ceiling, which is the correct priority for a delivery chain.

4.3 Noise-Floor Gate (16-bit Stage)

The test compares the error of the Lipshitz-shaped 16-bit output with that of a flat TPDF 16-bit output. It computes the difference in band power for three bands: 50–1000 Hz, 1–5 kHz, and 15–20 kHz. The gate requires the 1–5 kHz difference to be ≤ −10 dB.

This is a direct test of the shaping filter's efficacy. Because the Lipshitz filter is designed to minimise audibility in the midrange, a 10 dB reduction in the 1–5 kHz error power relative to flat dither is a conservative but meaningful threshold. The test also counts spurs in the 16-bit error and requires zero, which confirms that the TPDF dither is successfully linearising the quantiser.

4.4 Determinism Gate

The test runs the full Chain B twice with identical seeds and checks for bit-identical output. This is a necessary condition for reproducibility. The modules use np.random.default_rng with explicit seeds, and the ProDither instance is recreated per call, so the RNG state is reset. The gate is straightforward but essential.

4.5 Limitations of the Test Suite

The test explicitly notes that WER (Word Error Rate) is not measured here. This is a significant omission for a chain intended for speech or vocal delivery. Dither and noise shaping can interact with perceptual codecs and with the human auditory system in ways that are not captured by spectral spurs or band-power differences. A real vocal through a perceptual codec (Opus, AAC) must be evaluated with a listening test or an objective intelligibility metric. The test acknowledges this in the final summary.

Another limitation is the absence of a null test against an ideal quantiser. The spur gate compares against an unquantised reference, which is useful, but a null test against a high-precision floating-point model of the same chain would reveal any systematic errors in the fixed-point emulation.

---

5. Design Indents: Structural Observations

1. Separation of concerns: jowiphon_noise handles noise generation and loudness/TP, while prodither_fixed handles word-length reduction and dithering. The integration test orchestrates them. This separation is clean and testable.
2. Immutable configuration: ProDitherConfig is a dataclass with sensible defaults. The presets are defined as module-level constants. This makes the engine easy to reconfigure without mutating global state.
3. Numba acceleration: The @njit decorators on the quantisation and saturation kernels provide a significant speedup when Numba is available, with a graceful fallback to pure Python. The cache=True option is appropriate for repeated runs.
4. Error handling: The code raises ValueError for unknown source kinds, shapes, and presets. The render function checks for zero-length signals and undefined loudness. This is adequate for a library module.

---

6. Conclusion

The three modules form a coherent, technically sound testbed for a mastering-grade dither and loudness chain. The BS.1770 loudness measurement and true-peak estimation are implemented according to the relevant standards. The TPDF dither and error-feedback noise shaping follow the classical theory of Lipshitz, Vanderkooy, and Wannamaker. The integration test is rigorous in its spur and noise-floor gates, with sensitivity checks and multi-seed verification.

The primary caveats are:

· The "true-peak trim" is a constant-scale operation, not a dynamic limiter, and its loudness side-effect is correctly accounted for in the test logic.
· The test suite does not evaluate perceptual intelligibility (WER) or codec interaction.
· The requested "Seals and Windsor" report could not be located; the analysis therefore relies on the standard peer-reviewed literature in the field.

The code is well-structured, the tests are meaningful, and the design is faithful to the underlying signal-processing theory.

---

References

1. Lipshitz, S. P., Vanderkooy, J., & Wannamaker, R. A. (1991). Minimally Audible Noise Shaping. Journal of the Audio Engineering Society, 39(11), 836–852. 
2. ITU-R BS.1770-4 (and -5). Algorithms to measure audio programme loudness and true-peak audio level. International Telecommunication Union. 
3. Hicks, C. (1995). The Application of Dither and Noise-Shaping to Nyquist-Rate Digital Audio: an Introduction. Cambridge University Engineering Department. 
4. Johnson, J. B. (1928). Thermal Agitation of Electricity in Conductors. Physical Review, 32(1), 97–109. (Johnson–Nyquist noise; 1/f flicker noise.) 
