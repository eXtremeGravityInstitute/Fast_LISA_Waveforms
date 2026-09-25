# Copyright (C) Neil Cornish and Noah Pearson.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Experimental modified gaussian "WDG" transform"""

import argparse
import math
import numpy as np
import sys

def tw_freq(x, NF, NT, fs, q=8, window_choice="meyer"):
    """transform time domain to time-frequency using the frequency domain version of the transform
    * x: time series data
    * NF, NT: number of frequency, time pixels
    * fs: sampling frequency
    * q: Wilson frequency-tail truncation in half-band units.

    The Meyer window used by the C codes is compact in frequency, so q does
    not affect that branch. The modified Gaussian/Wilson window is formally
    infinite in both time and frequency. For the frequency-domain transform we
    therefore keep q half-bands of its exponentially decaying frequency tail.
    """

    N = NF*NT

    dt = 1/fs
    Tobs = N*dt

    dataf = np.fft.fft(x)*dt
    return fw_freq(dataf, NF, NT, fs, q=q, window_choice=window_choice)

def load_table(path):
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data

def uniform_spacing(values, path, label):
    values = np.asarray(values, dtype=float)
    diffs = np.diff(values)
    if diffs.size == 0:
        raise ValueError(f"{path} needs at least two rows to infer {label} spacing")

    spacing = np.median(diffs)
    if spacing <= 0:
        raise ValueError(f"{path} has non-positive {label} spacing")

    atol = max(1.0e-12, 1.0e-8*abs(spacing))
    if not np.allclose(diffs, spacing, rtol=1.0e-6, atol=atol):
        raise ValueError(f"{path} has non-uniform {label} spacing")

    return spacing

def infer_fs_from_time_column(times, path):
    dt = uniform_spacing(times, path, "time")
    return 1.0/dt

def infer_fs_from_frequency_column(freqs, N, path):
    df = uniform_spacing(freqs, path, "frequency")
    expected = df*np.arange(1, N//2)
    atol = max(1.0e-9, 1.0e-6*abs(df))
    if not np.allclose(freqs, expected, rtol=1.0e-6, atol=atol):
        raise ValueError(
            f"{path} frequency column is not consistent with bins 1..N/2-1 "
            f"for N={N}"
        )
    return N*df

def choose_sample_rate(requested_fs, inferred_fs, source):
    if requested_fs is None:
        return inferred_fs
    if requested_fs <= 0:
        raise ValueError("--fs must be positive")
    if not np.isclose(requested_fs, inferred_fs, rtol=1.0e-6, atol=0.0):
        print(
            f"Warning: --fs={requested_fs:.12g} Hz differs from "
            f"{source}-column inference {inferred_fs:.12g} Hz; using --fs.",
            file=sys.stderr,
        )
    return requested_fs

def read_time_data(path, NF, NT):
    data = load_table(path)
    if data.shape[0] != NF*NT or data.shape[1] < 2:
        raise ValueError(f"{path} does not match NF*NT={NF*NT} two-column time data")
    inferred_fs = infer_fs_from_time_column(data[:, 0], path)
    return data[:, 1], inferred_fs

def read_freq_data(path, NF, NT):
    rows = load_table(path)
    N = NF*NT
    if rows.shape[0] != N//2-1 or rows.shape[1] < 3:
        raise ValueError(f"{path} does not match N/2-1={N//2-1} three-column frequency data")

    inferred_fs = infer_fs_from_frequency_column(rows[:, 0], N, path)
    dataf = np.zeros(N, dtype=np.complex128)
    dataf[1:N//2] = rows[:, 1] + 1j*rows[:, 2]
    dataf[N//2+1:] = np.conj(dataf[1:N//2][::-1])
    return dataf, inferred_fs

def fw_freq(dataf, NF, NT, fs, q=8, window_choice="meyer"):
    """Frequency-domain WDM transform using the C-code conventions.

    For Meyer, this is the same compact-support Eq. 17 implementation as
    wdm_viafreq.c. For Wilson, the Gaussian frequency-domain window is not
    compact: samples outside the principal |l| < NT/2 band still contribute.
    Those tail samples are accumulated into the NT-point inverse FFT modulo NT.
    This is the frequency-domain analogue of truncating the long time-domain
    Meyer window in wdm_transform.c.
    """

    N = NF*NT
    dt = 1/fs
    Tobs = N*dt

    # wavelet pixels
    DF = 1/(2*dt*NF)

    # angular frequency spacing
    domega = 2*np.pi/Tobs

    # get window
    if window_choice == 'meyer':
        max_l = NT//2
        # frequencies to evaluate
        fn = np.arange(max_l+1)/Tobs
        phif = np.zeros(NT//2+1)
        for i in range(NT//2+1):
            w = i*domega
            phif[i] = meyer_fd(w, DF)
    elif window_choice == 'wilson':
        if q < 1:
            raise ValueError("q must be at least 1 for the Wilson window")

        # The Wilson/modified Gaussian profile has exponential frequency tails.
        # Keeping only the principal half-band is not enough: the missing tails
        # produce percent-level Parseval errors. Increasing q rapidly suppresses
        # the truncation error; for the bundled test data q=2 gives ~5e-4,
        # q=4 gives ~1e-5, and q=8 gives ~1e-8.
        max_l = q*NT//2
        fn = np.arange(max_l+1)/Tobs
        nu = 0.5
        phif = wilson_fd(fn,DF,nu)

        # wilson_fd() constructs a dimensionless profile. Match the physical
        # normalization used by the Meyer window, Phi ~ 1/sqrt(DeltaOmega), so
        # Parseval is measured in the same units as the C transforms.
        phif = np.real_if_close(phif, tol=1000).astype(float)/np.sqrt(2*np.pi*DF)
    else:
        raise Exception('not a valid window choice')

    # object to store wavelet coefficients
    wave = np.zeros((NT, NF+1))
    scale = 2*np.sqrt(np.pi)/Tobs

    # loop through frequencies and apply Meyer wavelet window
    for m in range(NF+1):
        DX = np.zeros(NT,dtype=np.complex128)

        for l in range(-max_l,max_l):
            n = l + NT//2 # index range [0,NT-1]
            jj = l + m*NT//2

            if window_choice == 'wilson':
                # Tails beyond the principal NT samples have the same Fourier
                # phase on the WDM time grid up to periodic wrapping, so they
                # add into the same length-NT inverse FFT bins. This is the
                # essential difference from the compact Meyer branch below.
                DX[l % NT] += dataf[jj % N]*phif[abs(l)]
            else:
                if m == 0 or m == NF:
                    DX[n] = dataf[jj % N]*phif[abs(l)]
                elif jj > 0 and jj < N//2:
                    DX[n] = dataf[jj]*phif[abs(l)]

        # inverse transform
        data = np.fft.ifft(DX)*NT

        for n in range(NT):
            if m == 0 or m == NF:
                if (n+m) % 2 == 0:
                    wave[n,m] = scale*np.real(data[n])/np.sqrt(2.0)
            elif m % 2 == 0:
                if (n+m) % 2 == 0:
                    wave[n,m] = scale*np.real(data[n])
                else:
                    wave[n,m] = scale*np.imag(data[n])
            else:
                if (n+m) % 2 == 0:
                    wave[n,m] = scale*np.real(data[n])
                else:
                    wave[n,m] = -scale*np.imag(data[n])

    return wave

def write_gnuplot_data(path, wave, NF, NT, fs):
    dt = 1/fs
    DT = dt*NF
    DF = 1/(2*dt*NF)

    with open(path, "w", encoding="utf-8") as out:
        for m in range(NF+1):
            if m == 0:
                fplot = 0.25*DF
            elif m == NF:
                fplot = (NF-0.25)*DF
            else:
                fplot = m*DF

            for n in range(NT):
                if (m == 0 or m == NF) and (n+m) % 2 != 0:
                    continue
                out.write(f"{n*DT:e} {fplot:e} {wave[n,m]:.14e}\n")
            out.write("\n")

def write_gnuplot_script(path, data_path, output_png, NF, NT, fs, zmax):
    dt = 1/fs
    Tobs = NF*NT*dt
    with open(path, "w", encoding="utf-8") as out:
        out.write("set term png enhanced truecolor crop font Helvetica 18  size 1200,800\n")
        out.write(f"set output '{output_png}'\n")
        out.write("set pm3d map corners2color c1\n")
        out.write("set ylabel 'f (Hz)'\n")
        out.write("set xlabel 't (s)'\n")
        out.write(f"set xrange [0:{Tobs:e}]\n")
        out.write(f"set yrange [0:{0.5/dt:e}]\n")
        out.write(f"set cbrange [{-zmax:e}:{zmax:e}]\n")
        out.write("set palette defined (0 '#b2182b', 1 '#ef8a62', 2 '#fddbc7', 3 '#ffffff', 4 '#d1e5f0', 5 '#67a9cf', 6 '#2166ac')\n")
        out.write(f"splot '{data_path}' using 1:2:3 notitle\n")

def rounded_color_scale(wave):
    ymax = np.max(np.abs(wave))
    if ymax == 0:
        return 1.0
    decade = 10.0**np.floor(np.log10(ymax))
    return np.ceil(ymax/decade)*decade

def main():
    parser = argparse.ArgumentParser(description="Compute a WDM transform and write gnuplot-ready output.")
    parser.add_argument("filename")
    parser.add_argument("domain", choices=["time", "freq", "0", "1"], help="input type: time/0 or freq/1")
    parser.add_argument("--nf", type=int, default=256)
    parser.add_argument("--nt", type=int, default=256)
    parser.add_argument(
        "--fs",
        type=float,
        default=None,
        help="sample rate in Hz; omit to infer it from the input file's first column",
    )
    parser.add_argument("--q", type=int, default=8, help="Wilson frequency-tail truncation in half-band units")
    parser.add_argument("--window", choices=["meyer", "wilson"], default="meyer")
    parser.add_argument("--data-out", default="BinaryPy.dat")
    parser.add_argument("--script-out", default="tranpy.gnu")
    parser.add_argument("--png-out", default="tranpy.png")
    args = parser.parse_args()

    if args.domain in ("time", "0"):
        x, inferred_fs = read_time_data(args.filename, args.nf, args.nt)
        fs = choose_sample_rate(args.fs, inferred_fs, "time")
        input_power = np.sum(x*x)/fs
        wave = tw_freq(x, args.nf, args.nt, fs, q=args.q, window_choice=args.window)
    else:
        dataf, inferred_fs = read_freq_data(args.filename, args.nf, args.nt)
        fs = choose_sample_rate(args.fs, inferred_fs, "frequency")
        input_power = 2*np.sum(np.abs(dataf[1:args.nf*args.nt//2])**2)/(args.nf*args.nt/fs)
        wave = fw_freq(dataf, args.nf, args.nt, fs, q=args.q, window_choice=args.window)

    total_power = np.sum(wave*wave)
    print(f"Sample rate {fs:e} Hz")
    print(f"Total input power {input_power:e}")
    print(f"Total power {total_power:f}")
    print(f"Parseval fractional error {(total_power-input_power)/input_power:e}")

    write_gnuplot_data(args.data_out, wave, args.nf, args.nt, fs)
    write_gnuplot_script(args.script_out, args.data_out, args.png_out, args.nf, args.nt, fs, rounded_color_scale(wave))
    print(f"Wrote {args.data_out}")
    print(f"Wrote {args.script_out}")

def meyer_fd(w,DF):
    """function to compute the FD Meyer window function
    - takes angular frequency argument (single value)"""
    
    DOmega = 2*np.pi*DF
    
    # choose the wavelet morphological parameters -> 2A+B = DOmega
    #A,B,d = DOmega/4,DOmega/2,4
    
    # NEW PARAMETER CHOICE - June 2025
    A,B,d = 0,DOmega,6

    if abs(w) < A:
        phi = 1/np.sqrt(DOmega)
        
    elif A <= abs(w) and abs(w) <= A+B:
        
        arg = (abs(w)-A)/B
        
        # normalized incomplete Beta function
        nu = beta_inc_integer(d, arg)

        
        phi = np.cos(np.pi*nu/2)/np.sqrt(DOmega)
        
    else:
        phi = 0
        
    return phi

def beta_inc_integer(d, x):
    """Regularized incomplete beta I_x(d,d), for positive integer d."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    return sum(
        math.comb(2*d-1, j) * x**j * (1.0-x)**(2*d-1-j)
        for j in range(d, 2*d)
    )

def wilson_fd(fn,DF,nu):
    """function to compute the 'Wilson' window (modified gaussians) in the frequency domain
    * fn: list of frequencies
    * DF: wavelet frequency pixel span
    * nu: gaussian width shape parameter (0.5 is TF symmetric)"""

    # number of terms in nm sum; n = (-N,N), m = (-M,M)
    M = 20
    N = 20
    Kmax = 40

    # index lattice
    pairs = [(m, n) for m in range(-M, M + 1)
                    for n in range(-N, N + 1)]

    P = len(pairs)
    center_index = pairs.index((0, 0))
    
    # compute prefactor
    A, B, t_min, s_min, t_max, s_max = frame_bounds(nu)
    
    A_nu = A
    B_nu = B
    
    alpha = 2.0 / (A_nu + B_nu)

    b = np.zeros(P, dtype=complex)
    b[center_index] = 1.0

    a = np.zeros(P, dtype=complex)

    c_k = 1.0
    
    m_arr = np.array([p[0] for p in pairs], dtype=int)
    n_arr = np.array([p[1] for p in pairs], dtype=int)
    
    m = m_arr[:, None]
    n = n_arr[:, None]
    mp = m_arr[None, :]
    np_ = n_arr[None, :]
    
    Omega = np.exp(
        1j * np.pi * (mp - m) * (n + np_) / 2
        - np.pi * nu * (n - np_)**2 / 2
        - np.pi * (m - mp)**2 / (8 * nu)
    )

    for k in range(Kmax + 1):
        if k > 0:
            c_k *= (2 * k - 1) / (2 * k)

        a += c_k * b

        # Equivalent to applying [I - 2P/(A+B)]
        b = b - alpha * (Omega @ b)

    prefactor = 2.0*np.sqrt(1.0 / (A_nu + B_nu))
    
    fn = np.asarray(fn)/DF
    out = np.zeros_like(fn, dtype=complex)

    for coeff, m, n in zip(a, m_arr, n_arr):
        out += coeff * g_mn_x(fn, m, n, nu)
        # out += coeff * g_mn_hat(fn, m, n, nu)

    return prefactor * out

def frame_bounds(nu, Nt=100, Ns=100, L=20):
    ts = np.linspace(0, 1, Nt, endpoint=False)
    ss = np.linspace(0, 1, Ns, endpoint=False)

    A = np.inf
    B = -np.inf
    
    t_min = s_min = None
    t_max = s_max = None

    for t in ts:
        Z0 = np.array([zak_g(t, s, nu, L=L) for s in ss])
        Z1 = np.array([zak_g(t, s + 0.5, nu, L=L) for s in ss])

        F = np.abs(Z0)**2 + np.abs(Z1)**2

        j_min = np.argmin(F)
        j_max = np.argmax(F)

        if F[j_min] < A:
            A = F[j_min]
            t_min = t
            s_min = ss[j_min]

        if F[j_max] > B:
            B = F[j_max]
            t_max = t
            s_max = ss[j_max]

    return A, B, t_min, s_min, t_max, s_max

def g_nu(x, nu):
    return (2 * nu)**0.25 * np.exp(-np.pi * nu * x**2)

def g_mn_x(x, m, n, nu):
    return np.exp(1j * np.pi * m * x) * g_nu(x - n, nu)

def g_mn_hat(y, m, n, nu):
    sign = 1.0 if (m * n) % 2 == 0 else -1.0
    return sign * np.exp(2j * np.pi * y * n) * g_nu(y + m / 2, 1 / nu)
    
def zak_g(t, s, nu, L=20):
    z = 0.0j
    for k in range(-L, L + 1):
        z += g_nu(t + k, nu) * np.exp(2j * np.pi * k * s)
    return z

if __name__ == "__main__":
    main()
