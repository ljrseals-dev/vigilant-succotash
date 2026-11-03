import numpy as np
from scipy import signal
exec(open('aura_v_edd_harness2.py').read().split('# ---- NTF check')[0])
FS_EFF = SR/4                      # 11025 Hz after /4 decimation; Nyquist 5512.5 Hz
t = np.arange(SR*2)/SR; s = 0.5*np.sin(2*np.pi*1000*t)

def curve(x,a=2.0,b=0.3): return np.tanh(a*x+b)-np.tanh(b)
def q8(x): return np.round(x*128)/128
def zoh(x,D=4): return np.repeat(x[::D],D)[:len(x)]
def aa(x, D=4, fc_frac=0.45):     # linear-phase FIR, cutoff at 0.45*fs_eff
    h = signal.firwin(511, fc_frac*SR/D, fs=SR); return signal.fftconvolve(x,h,mode="same")

def band_report(y, tag, nyq=FS_EFF/2):
    """True fold-back metric: energy in 0..FS_EFF/2 at NON-harmonic bins of 1 kHz, rel. to fundamental.
       ZOH images sit at m*FS_EFF +/- f (>= 10.0 kHz for a 1 kHz tone) so they cannot land in this band."""
    Y = np.abs(np.fft.rfft(y*signal.windows.blackmanharris(len(y))))**2
    f = np.fft.rfftfreq(len(y),1/SR); i1=np.argmin(abs(f-1000)); fund=Y[i1-3:i1+4].sum()
    m = (f>20)&(f<nyq)
    for k in range(1,6): m &= abs(f-1000*k)>8
    print(f"{tag:46s} fold-back (0-{nyq:.0f} Hz, non-harm) = {10*np.log10(Y[m].sum()/fund+1e-30):7.1f} dBc")

print("=== TRUE FOLD-BACK METRIC (baseband only) ===")
c = curve(s)
band_report(q8(c),            "curve + 8-bit, no decimation (reference)")
band_report(q8(zoh(c)),       "curve -> ZOH/4 -> 8-bit (no AA)")
band_report(zoh(q8(c)),       "curve -> 8-bit -> ZOH/4 (spec order, no AA)")
band_report(zoh(q8(aa(c))),   "curve -> AA FIR -> 8-bit -> ZOH/4")
band_report(zoh(q8(aa(c,fc_frac=0.40))), "  same, AA cutoff 0.40*fs_eff")
band_report(zoh(aa(c)),       "curve -> AA FIR -> ZOH/4 (no 8-bit)")
band_report(zoh(c),           "curve -> ZOH/4 (no AA, no 8-bit)")

# harmonic content that survives AA (formant-band cost)
print("\n=== Harmonic levels after chain (dBc re 1 kHz) ===")
def harm(y,tag):
    Y=np.abs(np.fft.rfft(y*signal.windows.blackmanharris(len(y))))**2; f=np.fft.rfftfreq(len(y),1/SR)
    L=lambda k:Y[np.argmin(abs(f-1000*k))-3:np.argmin(abs(f-1000*k))+4].sum()
    print(f"{tag:34s} "+"  ".join(f"H{k}={10*np.log10(L(k)/L(1)+1e-30):6.1f}" for k in (2,3,4,5,6)))
harm(zoh(q8(c)),"no AA"); harm(zoh(q8(aa(c))),"with AA")

# as-pasted ProDither stability facts
x = 10**(-60/20)*np.sin(2*np.pi*1000*np.arange(SR*3)/SR)
for order,sign in (("as_pasted",+1),("newest_first",-1)):
    y = prodither(x,16,order,sign); print(f"ProDither {order} {sign:+d}: max|y|={abs(y).max():.2e} (full scale=1), finite={np.all(np.isfinite(y))}")
