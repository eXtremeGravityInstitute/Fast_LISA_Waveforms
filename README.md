# Sparse LISA waveform templates (public beta)

This beta provides fast LISA X/Y/Z time-delay-interferometry (TDI) responses
and sparse Meyer-wavelet (WDM) coefficients for four source families:

| Source | C | Python |
| --- | --- | --- |
| Circular Galactic binaries | `c/galactic_binary/response.c` | `python/galactic_binary_tdi_wdm.py` |
| Extreme-mass-ratio inspirals | - | `python/few_tdi_wdm.py` (requires FEW) |
| Aligned-spin massive black holes (IMRPhenomTHM) | `c/thm/PhenomTHM_TDI.c` | `python/phenomthm_tdi_wdm.py` |
| Precessing massive black holes (IMRPhenomTPHM) | `c/thm/PhenomTPHM_TDI.c` | `python/phenomtphm_tdi_wdm.py` |

The THM/TPHM waveform ports are LAL-free; the FEW driver calls the separately
installed FastEMRIWaveforms package. The fast response samples the source and
LISA geometry sparsely rather than generating a full-cadence waveform for
routine use. The default THM/TPHM WDM path uses overlapping, heterodyned
complex FFT blocks and one combined merger/ringdown FFT. TDI-1 and directed-
delay TDI-2 are available; the command-line default is TDI-1.

The algorithms and conventions are described in
[Time-frequency analysis for LISA: Fast waveform templates](https://arxiv.org/abs/2609.28316).
This is research software and a beta release, not a calibrated LISA analysis
pipeline.

## Requirements

- C99 compiler and GNU Scientific Library (GSL), including `gsl-config`, for
  the C programs.
- Python 3.12 (the tested version) with NumPy and SciPy; Numba is strongly
  recommended for repeated fast calls. The Python ports do not require
  LALSuite, PyCBC, JAX, or Phentax.
- [FastEMRIWaveforms (FEW)](https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms)
  in the Python environment for `few_tdi_wdm.py` only. FEW is not bundled.

Run the commands below from the repository root. The Python modules are in
`python/`; C sources and their Makefiles are in `c/thm/` and
`c/galactic_binary/`. This is a source distribution: no generated waveforms,
Mojito files, virtual environments, FEW, or Phentax are bundled.
The supplied defaults are reference examples, not a universal source catalog.
Use `--help` on each driver for parameters and output controls.

## Python quick start

```sh
python python/galactic_binary_tdi_wdm.py --tdi-generation 2 --no-write
python python/phenomthm_tdi_wdm.py --modes default --tdi-generation 2 \
  --coefficient-backend python --wdm-method partitioned-fft \
  --sparse-only --no-write
python python/phenomtphm_tdi_wdm.py --modes default --channels XYZ \
  --tdi-generation 2 --coefficient-backend python \
  --wdm-method partitioned-fft --timing
python python/few_tdi_wdm.py --modes selected --tdi-generation 2 \
  --fast-method block-fft --fast-only --no-write
```

Remove `--no-write` to retain sparse WDM coefficients where supported.
The TPHM driver returns its result in memory through
`generate_tphm_tdi_wdm`; its command-line example prints a summary rather
than writing a full-cadence waveform. The FEW block-FFT route requires more
work than the single-mode example; first calls also include FEW/Numba setup.

## C quick start

With GSL on the compiler's search path:

```sh
make -C c/thm
./c/thm/PhenomTHM_TDI --modes default --wdm-all --tdi-generation 2 \
  --no-diagnostics --sparse-files

make -C c/galactic_binary
(cd c/galactic_binary && ./response --sparse-hetF --tdi2)
```

The precessing C path is exposed by `PhenomTPHM_TDI.h` and uses the THM
response and intrinsic sources as well as `IMRPhenomTPHM_Precession.c` and
`IMRPhenomTPHM_Rotation.c`. `PhenomTPHM_TDI_test.c` is an executable example
of that API. Use `./c/thm/PhenomTHM_TDI --modes 22pair` for the current C
2,2-pair path.

## Result format

The primary result is a sparse **real WDM transform of the TDI X, Y, or Z
response**, not a full-cadence time series. Each channel's `track_pixels` file
is whitespace-delimited text with an optional `# n m wdm` header and three
columns: zero-based time-pixel index `n`, zero-based frequency-layer index
`m`, and the **signed** WDM coefficient. Rows not present in the sparse list
have zero coefficient; the third column is not power or SNR squared. Its square
is unweighted pixel power, before applying a noise model.

For grid settings `nt`, `nf`, and sample cadence `dt` (seconds), the WDM
time-pixel spacing is `DT = nf * dt`, the frequency-layer spacing is
`DF = 1 / (2 * nf * dt)` Hz, and `Tobs = nt * DT`. An interior pixel `(n,m)`
is centered at approximately `t = n * DT` seconds and `f = m * DF` Hz;
the DC (`m=0`) and Nyquist (`m=nf`) layers have special edge conventions.
The time coordinate is the SSB output/data timestamp. A field named
`detector_time` in some Python response grids is instead a retarded
guiding-center **source argument**, not another output timestamp.

| Driver | Main files or in-memory result |
| --- | --- |
| C THM | `track_pixels_THM_X.dat`, `_Y.dat`, `_Z.dat` in the working directory, each `n m wdm`. `--diagnostics` can add dense transform and plotting files. |
| C Galactic binary (`--sparse-hetF`) | `ucb_hetF_X_track_pixels.dat` (and Y/Z), each `n m wdm`. `ucb_hetF_pixels.dat` has `k list n m X Y Z`, with `list = m * Nt + n`. |
| Python Galactic binary | In `--output-dir` (default `galactic_binary_validation/`): `track_pixels_X.dat` (and Y/Z), `track_pixels_XYZ.dat` with `n m X Y Z` on the union support (missing channel values filled with zero), and `summary.csv` as `quantity,value`. |
| Python THM / 2,2 | `<prefix>_<channel>_track_pixels.dat`, with `n m wdm`; `<prefix>` is set by `--write-prefix`. Dense `<prefix>_<channel>_wtranfast.dat` is written unless `--sparse-only` is set. On the SPA or frequency-only path, `--write-frequency-domain` adds `*_freq_ap.dat` with `f amplitude phase real imag` columns; the default partitioned-FFT path does not write those per-carrier files. |
| Python FEW | In `--output-dir` (default `few_tdi_wdm_validation/`): `track_pixels_fast.dat` with `n m fast_wdm` and `summary.csv`. Direct-reference comparison tables, mode lists, and endpoint diagnostics depend on the selected validation options. |
| C / Python TPHM | The C API returns a `THMSparseWDMTriplet` containing per-channel `n`, `m`, and `value` arrays. Python `generate_tphm_tdi_wdm(...)` returns a `TPHMTDIWDMResult` whose `channels["X"]` (likewise Y/Z) is a `WDMChannel`. The Python TPHM CLI prints counts and optional timings but does not write coefficient files. |

Python `WDMChannel` objects expose aligned `listn`, `listm`, and `values`
arrays, plus `dense(shape)` for a zero-filled array of shape `(nt, nf+1)`
indexed as `[n, m]`. THM/TPHM result objects also expose nonuniform
`intrinsic` and `tdi` grids for inspecting amplitude, phase, and the response;
these are not full-cadence waveform samples. `--no-write` suppresses files in
drivers that support it, but still prints a run summary.

For source waveforms before TDI, `IMRPhenomTHM.evaluate_times(times, ...)`
returns a mapping from `(ell, m)` to `ModeSeries`. Each series has sampled
`amplitude`, `phase` (radians), and angular frequency `omega` (radians/second),
with complex mode `hlm = amplitude * exp(-1j * phase)` and ordinary frequency
`omega / (2*pi)` Hz. `IMRPhenomTPHM.evaluate_times(times, ...)` instead
returns a `TPHMWaveform`: its complex `strain` is `hplus - 1j*hcross`, and
`carriers[(ell, abs(m))]` holds the projected carrier quadratures and AP
samples. In either `evaluate_times` call, the arrays correspond to the supplied
source-time samples; they are not automatically resampled to the WDM cadence.
The C `IMRPhenomTHMEvaluateGrid` output is mode-major:
`samples[mode_index*n + i]` holds `amplitude`, `phase`, dimensionless `omega`
(`d phase / d(t/M)`), and complex `hlm` at sample `i`. The C
`IMRPhenomTPHMEvaluateGrid` instead writes sample-major complex observer
`strain[i] = hplus[i] - I*hcross[i]`; optional mode and Euler-angle arrays
are described in its header. The bare C TPHM intrinsic evaluation does not
apply a distance normalization.

## Scope and validation

The WDM grid, time cadence, and output naming are controlled by each driver;
full-cadence construction is reserved for validation. For the
reference THM source, both the C and Python TDI-2 partitioned-FFT outputs
have been compared with a full-cadence Mojito-orbit TDI/WDM calculation in
X/Y/Z. Those comparisons test the complete response and transform, but do not
establish accuracy for every source, orbit, or parameter extreme.

TDI-2 follows the directed-delay/PyTDI convention. Its default C evaluation
uses reference-time polarization projections and a Taylor-expanded delay
chain; the fully retarded numerical construction remains an opt-in cross-check
(`--tdi2-exact`). The TPHM Tapestry route and chirplet lookup are experimental
alternatives; use the partitioned FFT for the THM/TPHM reference path.

Fisher matrices, likelihood samplers, JAX/Phentax development code, Mojito
data, and internal LaTeX notes are outside this beta's scope. Numerical orbit
files and FEW must be obtained separately where needed.

## Provenance and license

The THM/TPHM intrinsic ports retain attribution to the LALSimulation
IMRPhenomTHM/TPHM authors and the local port changes. Their source notices
are GPL-2.0-or-later; the THM/TPHM response modules use GPL-3.0-or-later.
Keep the component notices, `NOTICE.md`, root `LICENSE` (GPLv3), and both
versioned license texts in `LICENSES/` with any combined source distribution.
Third-party packages, including FEW and GSL, have their own licenses.
