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
  --coefficient-backend python --wdm-method partitioned-fft --no-write
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

## Scope and validation

The standard output is a *sparse* list of occupied WDM pixels and their
coefficients. The WDM grid, time cadence, and output naming are controlled by
each driver; full-cadence construction is reserved for validation. For the
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
