#!/usr/bin/env python3
# Python port of the GPL-3.0-or-later PhenomT_TDI.c response by Neil Cornish.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fast LISA-TDI WDM generator for IMRPhenomT 2,2 and UCB sources.

This is a stripped Python port of the production path in ``PhenomT_TDI.c``:

1. build a sparse intrinsic IMRPhenomT 2,2 amplitude/phase grid and
   interpolate it onto the detector-response grid,
2. apply the fast unequal-arm Michelson TDI response on that grid,
3. map each TDI channel to frequency-domain amplitude/phase using the SPA,
   then blend endpoint WDM packets with a short FFT whose TDI delays are
   evaluated directly at the endpoint sample times,
4. assemble the active Meyer-WDM packets.

It intentionally omits the development diagnostics and alternative MBH WDM
paths.  The optional ``--source-model ucb`` branch follows the narrow-band
galactic-binary model and heterodyned frequency-domain WDM path in
``GB/response.c``.  The ``--source-model eccentric`` branch generalizes the
UCB model to the low-eccentricity, non-evolving-orbit harmonic expansion of
Moreno-Garrido et al./Seto, with phenomenological frequency, eccentricity, and
periastron-advance evolution.  The module API is parameterized and returns
arrays rather than writing files.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import math
import time
from typing import TYPE_CHECKING, Iterable, Sequence

import numpy as np
from scipy.interpolate import Akima1DInterpolator, CubicSpline
from scipy.special import betainc

from phenomt22 import GPSEC, TSUN, IMRPhenomT22

if TYPE_CHECKING:
    from phenomthm_tdi_wdm import THMTDIWDMResult

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]

        def decorate(func):
            return func

        return decorate


PI = math.pi

AU = 1.4959787e11
SECSYR = 3.15581498e7
PARSEC = 3.08568025e16
CLIGHT = 2.99792458e8
SQ3 = 1.7320508075688773
EC = 0.0048241852175
FM = 3.168753575e-8
FSTAR = 1.908538e-2

DTMAX = 2.0e5
AP_DTM_MAX = 1.0e4
DTMIN = 1.0
DPHASE_TRANSFER = 0.1
DPHASE_DEFAULT = 0.5
AP_MAX_PHASE_STEP = 100.0
AP_PHASE_CURVATURE_TOL = 0.25
AP_FREQ_REL_CURVATURE_TOL = 2.0e-2
AP_FREQ_STEP_REL_TOL = 5.0e-2
AP_TRANSFER_PHASE_STEP = 0.1
AP_STEP_GROWTH_LIMIT = 1.2
AP_STEP_SHRINK_FACTOR = 0.8
AP_STEP_BRACKET_ITERATIONS = 9
INTRINSIC_DTM_MAX = 2.0e5
INTRINSIC_MAX_PHASE_STEP = 100.0
INTRINSIC_PHASE_CURVATURE_TOL = 1.0e-2
INTRINSIC_FREQ_REL_CURVATURE_TOL = 5.0e-3
INTRINSIC_FREQ_STEP_REL_TOL = 5.0e-2
INTRINSIC_AMP_STEP_REL_TOL = 2.5e-1
INTRINSIC_AMP_REL_CURVATURE_TOL = 2.0e-2
INTRINSIC_STEP_GROWTH_LIMIT = 1.5
INTRINSIC_STEP_SHRINK_FACTOR = 0.8
INTRINSIC_TDI_SWITCH_TRANSFER_FRACTION = 0.5
INTRINSIC_TDI_SWITCH_TAIL_SECONDS = 2.0 * INTRINSIC_DTM_MAX
WDM_TIMESCAN_PAD_PIXELS = 0
CONSTELLATION_LIGHT_TIME_SECONDS = 500.0
CONSTELLATION_MIN_PADDING_SECONDS = CONSTELLATION_LIGHT_TIME_SECONDS
NOMINAL_ARM_LIGHT_TIME_SECONDS = 1.0 / (2.0 * PI * FSTAR)
TDI_MAX_PATH_DELAY_SECONDS = 4.0 * 1.02 * NOMINAL_ARM_LIGHT_TIME_SECONDS
RESPONSE_EARLY_MARGIN_SECONDS = CONSTELLATION_LIGHT_TIME_SECONDS
RESPONSE_LATE_MARGIN_SECONDS = CONSTELLATION_LIGHT_TIME_SECONDS + TDI_MAX_PATH_DELAY_SECONDS
INTRINSIC_TDI_SWITCH_GUARD_SECONDS = AP_DTM_MAX + CONSTELLATION_LIGHT_TIME_SECONDS
WAVEFORM_PRE_PADDING_SECONDS = 0.0
T_CUT_FREQ = -150.0
SHORTFFT_MERGER_TAPER_MARGIN_SECONDS = 0.0
DEFAULT_WDM_BLEND_ENDPOINT = True
DEFAULT_WDM_BLEND_HALF_WIDTH_LAYERS = 4.0

DEFAULT_NF = 4096
DEFAULT_NT = 4096
DEFAULT_DT = 1.875
DEFAULT_NX = 6.0
DEFAULT_BFRAC = 1.0
DEFAULT_MULT = 8
DEFAULT_UCB_FREQUENCY_HZ = 5.0e-3 + 0.6 / (2.0 * DEFAULT_DT * DEFAULT_NF)
ECCENTRIC_MAX_E = 0.3
ECCENTRIC_DEFAULT_NMAX = 4


@dataclass(frozen=True)
class SourceParams:
    """Intrinsic and extrinsic source parameters.

    ``m1_solar`` and ``m2_solar`` are converted to seconds internally. The mass
    hierarchy is enforced to match the LAL/IMRPhenomT convention.
    """

    m1_solar: float = 2.0e5
    m2_solar: float = 1.0e5
    chi1: float = 0.42
    chi2: float = 0.85
    phic: float = 0.0
    tc: float = 3.0e7
    distance_gpc: float = 1.0
    # Historical name: the response uses sin(value) as sky cos(theta), as for latitude.
    ecliptic_colatitude: float = 2.31
    ecliptic_longitude: float = 0.57
    polarization: float = 0.4
    cos_inclination: float = 0.3

    def masses_seconds(self) -> tuple[float, float, float, float]:
        m1 = self.m1_solar * TSUN
        m2 = self.m2_solar * TSUN
        chi1 = self.chi1
        chi2 = self.chi2
        if m2 > m1:
            m1, m2 = m2, m1
            chi1, chi2 = chi2, chi1
        return m1, m2, chi1, chi2


@dataclass(frozen=True)
class UCBSourceParams:
    """Nearly monochromatic galactic-binary source used by ``response.c``.

    The parameter order mirrors ``response.c``:
    ``f0, costh, phi, Amp, cosi, psi, phi0, fdot, fddot``.  If ``amplitude``,
    ``fdot``, or ``fddot`` are omitted, the same leading-order chirp estimates
    used in that C driver are computed from the component masses and distance.
    Times are measured from the start of the observation, not from a merger.
    """

    frequency_hz: float = DEFAULT_UCB_FREQUENCY_HZ
    ecliptic_costheta: float = -0.783326909627
    ecliptic_longitude: float = 3.0
    amplitude: float | None = None
    cos_inclination: float = 0.0707372016677
    polarization: float = 0.8
    phi0: float = 1.2
    fdot: float | None = None
    fddot: float | None = None
    m1_solar: float = 0.6
    m2_solar: float = 0.7
    distance_kpc: float = 1.0

    def masses_seconds(self) -> tuple[float, float]:
        return self.m1_solar * TSUN, self.m2_solar * TSUN

    def chirp_mass_seconds(self) -> float:
        m1, m2 = self.masses_seconds()
        mtot = m1 + m2
        return (m1 * m2) ** (3.0 / 5.0) / (mtot ** (1.0 / 5.0))

    def frequency_derivatives(self) -> tuple[float, float]:
        if self.fdot is not None and self.fddot is not None:
            return self.fdot, self.fddot
        mc = self.chirp_mass_seconds()
        f0 = self.frequency_hz
        fdot = 96.0 * PI ** (8.0 / 3.0) / 5.0 * mc ** (5.0 / 3.0) * f0 ** (11.0 / 3.0)
        fddot = (11.0 / 3.0) * fdot * fdot / f0
        if self.fdot is not None:
            fdot = self.fdot
        if self.fddot is not None:
            fddot = self.fddot
        return fdot, fddot

    def amplitude0(self) -> float:
        if self.amplitude is not None:
            return self.amplitude
        mc = self.chirp_mass_seconds()
        dl_seconds = self.distance_kpc * 1.0e3 * PARSEC / CLIGHT
        return 4.0 * mc ** (5.0 / 3.0) * (PI * self.frequency_hz) ** (2.0 / 3.0) / dl_seconds

    def evaluate(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        f0 = self.frequency_hz
        fdot, fddot = self.frequency_derivatives()
        amp0 = self.amplitude0()
        phase = self.phi0 + 2.0 * PI * (f0 * times + 0.5 * fdot * times * times + (1.0 / 6.0) * fddot * times * times * times)
        frequency = f0 + fdot * times + 0.5 * fddot * times * times
        # This is the same leading-order chirping-amplitude approximation used
        # in response.c.  The fuller A \propto f^(2/3) expression is left for a
        # later waveform-model upgrade.
        amplitude = amp0 * (1.0 + (2.0 / 3.0) * fdot / f0 * times)
        return phase, amplitude, frequency


ECCENTRIC_COEFFICIENTS: dict[tuple[str, int], tuple[tuple[int, float], ...]] = {
    # A_n multiplies the central nM term in h_+.
    ("central", 1): ((1, 0.5), (3, -1.0 / 16.0), (5, 1.0 / 384.0)),
    ("central", 2): ((2, 0.5), (4, -1.0 / 6.0), (6, 1.0 / 48.0)),
    ("central", 3): ((3, 9.0 / 16.0), (5, -81.0 / 256.0), (7, 729.0 / 10240.0)),
    ("central", 4): ((4, 2.0 / 3.0), (6, -8.0 / 15.0), (8, 8.0 / 45.0)),
    # ``plus`` is the phase nM + 2Phi sideband, coefficient (S_n-C_n)/2.
    ("plus", 1): ((3, 7.0 / 96.0), (5, 47.0 / 1536.0), (7, 1091.0 / 61440.0)),
    ("plus", 2): ((4, 1.0 / 16.0), (6, 11.0 / 480.0), (8, 173.0 / 11520.0)),
    ("plus", 3): ((5, 153.0 / 2560.0), (7, 63.0 / 4096.0), (9, 15507.0 / 1146880.0)),
    ("plus", 4): ((6, 11.0 / 180.0), (8, 17.0 / 2520.0), (10, 53.0 / 4032.0)),
    # ``minus`` is the phase nM - 2Phi sideband, coefficient (S_n+C_n)/2.
    # It contains the circular n=2 carrier in the e -> 0 limit.
    ("minus", 1): ((1, -3.0 / 4.0), (3, 13.0 / 32.0), (5, 5.0 / 768.0)),
    ("minus", 2): ((0, 1.0), (2, -5.0 / 2.0), (4, 23.0 / 16.0), (6, -65.0 / 288.0)),
    ("minus", 3): ((1, 9.0 / 4.0), (3, -171.0 / 32.0), (5, 963.0 / 256.0)),
    ("minus", 4): ((2, 4.0), (4, -10.0), (6, 101.0 / 12.0), (8, -1177.0 / 360.0)),
}


def _eccentric_poly(kind: str, harmonic: int, eccentricity: np.ndarray) -> np.ndarray:
    out = np.zeros_like(eccentricity, dtype=np.float64)
    for power, coefficient in ECCENTRIC_COEFFICIENTS[(kind, harmonic)]:
        out += coefficient * eccentricity**power
    return out


@dataclass(frozen=True)
class EccentricCarrierGrid:
    label: str
    harmonic: int
    kind: str
    intrinsic: "IntrinsicGrid"
    aplus: float
    across: float


@dataclass(frozen=True)
class EccentricUCBSourceParams:
    """Low-eccentricity generalization of the circular UCB source.

    ``frequency_hz`` is the observed circular carrier used by the existing UCB
    model.  With the convention used here, that carrier is the ``n=2`` minus
    sideband, ``f_2 - 2 delta_f``.  ``delta_f`` is therefore the periastron
    advance frequency, and the sideband phase uses
    ``2 Phi(t) = periastron_phase0 + 4 pi integral delta_f(t) dt``.  This keeps
    the ``eccentricity -> 0, delta_f -> 0`` limit identical to ``UCBSourceParams``.
    """

    frequency_hz: float = DEFAULT_UCB_FREQUENCY_HZ
    ecliptic_costheta: float = -0.783326909627
    ecliptic_longitude: float = 3.0
    amplitude: float | None = None
    cos_inclination: float = 0.0707372016677
    polarization: float = 0.8
    phi0: float = 1.2
    fdot: float | None = None
    fddot: float | None = None
    m1_solar: float = 0.6
    m2_solar: float = 0.7
    distance_kpc: float = 1.0
    eccentricity0: float = 0.0
    edot: float = 0.0
    delta_f0: float = 0.0
    delta_fdot: float = 0.0
    periastron_phase0: float = 0.0
    nmax: int = ECCENTRIC_DEFAULT_NMAX

    def masses_seconds(self) -> tuple[float, float]:
        return self.m1_solar * TSUN, self.m2_solar * TSUN

    def chirp_mass_seconds(self) -> float:
        m1, m2 = self.masses_seconds()
        mtot = m1 + m2
        return (m1 * m2) ** (3.0 / 5.0) / (mtot ** (1.0 / 5.0))

    def frequency_derivatives(self) -> tuple[float, float]:
        return UCBSourceParams(
            frequency_hz=self.frequency_hz,
            fdot=self.fdot,
            fddot=self.fddot,
            m1_solar=self.m1_solar,
            m2_solar=self.m2_solar,
        ).frequency_derivatives()

    def amplitude0(self) -> float:
        return UCBSourceParams(
            frequency_hz=self.frequency_hz,
            amplitude=self.amplitude,
            m1_solar=self.m1_solar,
            m2_solar=self.m2_solar,
            distance_kpc=self.distance_kpc,
        ).amplitude0()

    def eccentricity(self, times: np.ndarray) -> np.ndarray:
        e = self.eccentricity0 + self.edot * times
        emin = float(np.min(e))
        emax = float(np.max(e))
        if emin < -1.0e-14:
            raise ValueError(f"eccentricity became negative: min(e)={emin:.6e}")
        if emax > ECCENTRIC_MAX_E + 1.0e-14:
            raise ValueError(
                f"eccentricity expansion is validated only through e={ECCENTRIC_MAX_E}; max(e)={emax:.6e}"
            )
        return np.maximum(e, 0.0)

    def phase_integrals(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        fdot, fddot = self.frequency_derivatives()
        circular = self.frequency_hz * times + 0.5 * fdot * times * times + (1.0 / 6.0) * fddot * times * times * times
        periastron = self.delta_f0 * times + 0.5 * self.delta_fdot * times * times
        return circular, periastron

    def carrier_frequency(self, times: np.ndarray, harmonic: int, kind: str) -> np.ndarray:
        fdot, fddot = self.frequency_derivatives()
        circular_frequency = self.frequency_hz + fdot * times + 0.5 * fddot * times * times
        delta_frequency = self.delta_f0 + self.delta_fdot * times
        f2 = circular_frequency + 2.0 * delta_frequency
        fn = 0.5 * float(harmonic) * f2
        if kind == "plus":
            return fn + 2.0 * delta_frequency
        if kind == "minus":
            return fn - 2.0 * delta_frequency
        return fn

    def carrier_phase(self, times: np.ndarray, harmonic: int, kind: str) -> np.ndarray:
        circular, periastron = self.phase_integrals(times)
        mean_phase2 = self.phi0 + self.periastron_phase0 + 2.0 * PI * (circular + 2.0 * periastron)
        phase = 0.5 * float(harmonic) * mean_phase2
        two_phi = self.periastron_phase0 + 4.0 * PI * periastron
        if kind == "plus":
            phase = phase + two_phi
        elif kind == "minus":
            phase = phase - two_phi
        return phase

    def carrier_geometry(self, kind: str) -> tuple[float, float]:
        cosi = self.cos_inclination
        if kind == "central":
            return 0.5 * (1.0 - cosi * cosi), 0.0
        return 0.5 * (1.0 + cosi * cosi), -cosi

    def carrier_grids(self, waveform_time: np.ndarray, response_time: np.ndarray, shape: "WDMShape") -> list[EccentricCarrierGrid]:
        if self.nmax < 1 or self.nmax > 4:
            raise ValueError("--ecc-nmax must be between 1 and 4")
        e = self.eccentricity(waveform_time)
        amp0 = self.amplitude0()
        fdot, _fddot = self.frequency_derivatives()
        base_amplitude = amp0 * (1.0 + (2.0 / 3.0) * fdot / self.frequency_hz * waveform_time)
        carriers: list[EccentricCarrierGrid] = []
        for harmonic in range(1, self.nmax + 1):
            for kind in ("central", "plus", "minus"):
                coefficient = _eccentric_poly(kind, harmonic, e)
                amplitude = base_amplitude * coefficient
                if np.max(np.abs(amplitude)) <= abs(amp0) * 1.0e-12:
                    continue
                phase = self.carrier_phase(waveform_time, harmonic, kind)
                frequency = self.carrier_frequency(waveform_time, harmonic, kind)
                fmin = float(np.min(self.carrier_frequency(np.array([0.0, shape.Tobs], dtype=np.float64), harmonic, kind)))
                fmax = float(np.max(self.carrier_frequency(np.array([0.0, shape.Tobs], dtype=np.float64), harmonic, kind)))
                kx, kw = wdm_band_for_frequency_range(fmin, fmax, shape)
                setup = np.array([float(kx), float(kw), float(kx) * shape.DF, shape.DT / float(kw + 1)], dtype=np.float64)
                intrinsic = IntrinsicGrid(waveform_time, response_time, amplitude, phase, frequency, setup, None)
                aplus, across = self.carrier_geometry(kind)
                carriers.append(EccentricCarrierGrid(f"n{harmonic}_{kind}", harmonic, kind, intrinsic, aplus, across))
        return carriers


@dataclass(frozen=True)
class WDMShape:
    nf: int = DEFAULT_NF
    nt: int = DEFAULT_NT
    dt: float = DEFAULT_DT
    bfrac: float = DEFAULT_BFRAC
    nx: float = DEFAULT_NX
    mult: int = DEFAULT_MULT

    @property
    def n(self) -> int:
        return self.nf * self.nt

    @property
    def DT(self) -> float:
        return self.dt * self.nf

    @property
    def DF(self) -> float:
        return 1.0 / (2.0 * self.dt * self.nf)

    @property
    def OM(self) -> float:
        return PI / self.dt

    @property
    def DOM(self) -> float:
        return self.OM / self.nf

    @property
    def B(self) -> float:
        return self.bfrac * self.DOM

    @property
    def A(self) -> float:
        return (self.DOM - self.B) / 2.0

    @property
    def insDOM(self) -> float:
        return 1.0 / math.sqrt(self.DOM)

    @property
    def FB(self) -> float:
        return (self.A + self.B) / (2.0 * PI)

    @property
    def DFA(self) -> float:
        return self.A / (2.0 * PI)

    @property
    def Tfilt(self) -> float:
        return self.dt * float(self.mult * 2 * self.nf)

    @property
    def Tobs(self) -> float:
        return self.dt * float(self.nt * self.nf)


@dataclass
class IntrinsicGrid:
    bary_time: np.ndarray
    detector_time: np.ndarray
    amplitude: np.ndarray
    phase: np.ndarray
    frequency: np.ndarray
    setup: np.ndarray
    model: object | None
    model_time: np.ndarray | None = None
    response_switch_detector_time: float | None = None
    exact_response_samples: int = 0


@dataclass
class TDIGrid:
    # ``time`` is the public SSB output timestamp.  ``detector_time`` is kept
    # for compatibility, but it is not another output timestamp: it is the
    # guiding-center retarded waveform argument u=t-k.r_0(t).
    time: np.ndarray
    detector_time: np.ndarray
    reference_phase: np.ndarray
    bary_phase: np.ndarray
    amplitude: dict[str, np.ndarray]
    phase_offset: dict[str, np.ndarray]

    def channel_phase(self, channel: str) -> np.ndarray:
        return self.reference_phase + self.phase_offset[channel]

    @property
    def ssb_output_time(self) -> np.ndarray:
        return self.time

    @property
    def center_source_time(self) -> np.ndarray:
        return self.detector_time


@dataclass
class WDMChannel:
    freq: np.ndarray
    phase: np.ndarray
    amplitude: np.ndarray
    nmid: np.ndarray
    nsize: np.ndarray
    listn: np.ndarray
    listm: np.ndarray
    values: np.ndarray

    def dense(self, shape: WDMShape) -> np.ndarray:
        out = np.zeros((shape.nt, shape.nf + 1), dtype=np.float64)
        out[self.listn, self.listm] = self.values
        return out


@dataclass
class TDIWDMResult:
    source: object
    shape: WDMShape
    intrinsic: IntrinsicGrid
    tdi: TDIGrid
    nmid: np.ndarray
    nsize: np.ndarray
    channels: dict[str, WDMChannel]
    timings: dict[str, float]


def spacecraft(t: float) -> np.ndarray:
    alpha = 2.0 * PI * t * FM
    sa = math.sin(alpha)
    ca = math.cos(alpha)
    out = np.empty((3, 3), dtype=np.float64)
    for i, beta in enumerate((0.0, 2.0 * PI / 3.0, 4.0 * PI / 3.0)):
        sb = math.sin(beta)
        cb = math.cos(beta)
        out[i, 0] = AU * ca + AU * EC * (sa * ca * sb - (1.0 + sa * sa) * cb)
        out[i, 1] = AU * sa + AU * EC * (sa * ca * cb - (1.0 + ca * ca) * sb)
        out[i, 2] = -SQ3 * AU * EC * (ca * cb + sa * sb)
    return out


def constellation(tarray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nc = tarray.size
    pos = np.empty((3, 3, nc), dtype=np.float64)
    arms = np.empty((3, 3, nc), dtype=np.float64)
    lengths = np.empty((3, nc), dtype=np.float64)

    for n, t in enumerate(tarray):
        p = spacecraft(float(t)) / CLIGHT
        pos[:, :, n] = p
        arms[0, :, n] = p[1] - p[2]
        arms[1, :, n] = p[2] - p[0]
        arms[2, :, n] = p[0] - p[1]
        for i in range(3):
            lengths[i, n] = np.linalg.norm(arms[i, :, n])
            arms[i, :, n] /= lengths[i, n]
    return lengths, pos, arms


def build_constellation_splines(shape: WDMShape) -> tuple[np.ndarray, list[CubicSpline], list[CubicSpline], list[CubicSpline]]:
    nc = int(200.0 * shape.Tobs / SECSYR)
    if nc < 20:
        nc = 20
    dtx = shape.Tobs / float(nc - 1)
    if dtx < CONSTELLATION_MIN_PADDING_SECONDS:
        dtx = CONSTELLATION_MIN_PADDING_SECONDS
    dtc = (shape.Tobs + 2.0 * dtx) / float(nc - 1)
    tarray = -dtx + dtc * np.arange(nc, dtype=np.float64)
    lengths, pos, arms = constellation(tarray)

    l_splines = [CubicSpline(tarray, lengths[i], bc_type="natural") for i in range(3)]
    p_splines = [CubicSpline(tarray, pos[i, j], bc_type="natural") for i in range(3) for j in range(3)]
    v_splines = [CubicSpline(tarray, arms[i, j], bc_type="natural") for i in range(3) for j in range(3)]
    return tarray, l_splines, p_splines, v_splines


def sky_vectors(source: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if hasattr(source, "ecliptic_costheta"):
        costh = float(getattr(source, "ecliptic_costheta"))
    else:
        costh = math.sin(float(getattr(source, "ecliptic_colatitude")))
    phi = source.ecliptic_longitude
    costh = max(-1.0, min(1.0, costh))
    sinth = math.sqrt(1.0 - costh * costh)
    cosph = math.cos(phi)
    sinph = math.sin(phi)
    u = np.array([-costh * cosph, -costh * sinph, sinth], dtype=np.float64)
    v = np.array([sinph, -cosph, 0.0], dtype=np.float64)
    kv = np.array([-sinth * cosph, -sinth * sinph, -costh], dtype=np.float64)
    return u, v, kv


def _stack_spline_coefficients(splines: list[CubicSpline]) -> np.ndarray:
    return np.ascontiguousarray(np.stack([s.c for s in splines], axis=0), dtype=np.float64)


def make_ap_spline(x: np.ndarray, y: np.ndarray):
    """Shape-preserving AP spline used for waveform and TDI amplitude/phase.

    This mirrors the current C driver, which switched the AP splines to Akima
    interpolation to reduce overshoot near merger and TDI transfer-function
    notches.  The constellation/orbit splines remain ordinary cubic splines.
    """

    return Akima1DInterpolator(x, y, extrapolate=True)


# Hot scalar spline loops use SciPy's stored piecewise-polynomial coefficients
# directly. This keeps the interpolation identical while avoiding thousands of
# Python/SciPy scalar calls inside the TDI and barycentric-time calculations.
@njit(cache=True)
def _spline_interval_jit(x: np.ndarray, t: float) -> int:
    n = x.size
    if t <= x[0]:
        return 0
    if t >= x[n - 1]:
        return n - 2
    lo = 0
    hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if x[mid] <= t:
            lo = mid
        else:
            hi = mid
    return lo


@njit(cache=True)
def _spline_eval_at_jit(x: np.ndarray, coeffs: np.ndarray, t: float, i: int) -> float:
    dx = t - x[i]
    return ((coeffs[0, i] * dx + coeffs[1, i]) * dx + coeffs[2, i]) * dx + coeffs[3, i]


@njit(cache=True)
def _spline_eval_jit(x: np.ndarray, coeffs: np.ndarray, t: float) -> float:
    return _spline_eval_at_jit(x, coeffs, t, _spline_interval_jit(x, t))


@njit(cache=True)
def _spline_eval_stack_at_jit(x: np.ndarray, coeffs: np.ndarray, series: int, t: float, i: int) -> float:
    dx = t - x[i]
    return ((coeffs[series, 0, i] * dx + coeffs[series, 1, i]) * dx + coeffs[series, 2, i]) * dx + coeffs[series, 3, i]


@njit(cache=True)
def _unwrap_jit(phi: np.ndarray) -> np.ndarray:
    out = np.empty(phi.size, dtype=np.float64)
    twopi = 2.0 * PI
    v = phi[0]
    for i in range(phi.size):
        u = phi[i]
        q = round(abs(u - v) / twopi)
        if q > 0:
            if v > u:
                u += q * twopi
            else:
                u -= q * twopi
        v = u
        out[i] = u
    return out


@njit(cache=True)
def _extract_ap_jit(m: np.ndarray, mf: np.ndarray, reference_phase: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = m.size
    amp = np.empty(n, dtype=np.float64)
    flip = np.ones(n, dtype=np.float64)
    pjump = np.zeros(n, dtype=np.float64)
    for i in range(n):
        amp[i] = math.sqrt(m[i] * m[i] + mf[i] * mf[i])

    i = 1
    while i < n - 1:
        flip[i] = flip[i - 1]
        pjump[i] = pjump[i - 1]
        if amp[i] < amp[i - 1] and amp[i] < amp[i + 1]:
            d1 = amp[i + 1] + amp[i - 1] - 2.0 * amp[i]
            d2 = -amp[i + 1] + amp[i - 1] - 2.0 * amp[i]
            d3 = -amp[i + 1] + amp[i - 1] + 2.0 * amp[i]
            if abs(d1) > 0.0 and abs(d2 / d1) < 0.1:
                flip[i + 1] = -flip[i]
                pjump[i + 1] = pjump[i] + PI
                i += 1
            elif abs(d1) > 0.0 and abs(d3 / d1) < 0.1:
                flip[i] = -flip[i - 1]
                pjump[i] = pjump[i - 1] + PI
        i += 1

    flip[n - 1] = flip[n - 2]
    pjump[n - 1] = pjump[n - 2]

    signed_amp = np.empty(n, dtype=np.float64)
    phase_offset = np.empty(n, dtype=np.float64)
    twopi = 2.0 * PI
    for i in range(n):
        signed_amp[i] = flip[i] * amp[i]
        rem = reference_phase[i] - math.floor(reference_phase[i] / twopi) * twopi
        phase_offset[i] = -math.atan2(mf[i], m[i]) + pjump[i] - rem
    return signed_amp, _unwrap_jit(phase_offset)


@njit(cache=True)
def _hphc_coeff_jit(
    t: float,
    wave_x: np.ndarray,
    amp_coeffs: np.ndarray,
    phase_coeffs: np.ndarray,
    aplus: float,
    across: float,
    cos2psi: float,
    sin2psi: float,
) -> tuple[float, float, float, float]:
    amp = _spline_eval_jit(wave_x, amp_coeffs, t)
    phase = _spline_eval_jit(wave_x, phase_coeffs, t)
    cp = math.cos(phase)
    sp = math.sin(phase)
    hp = amp * (aplus * cos2psi * cp + across * sin2psi * sp)
    hc = amp * (across * cos2psi * sp - aplus * sin2psi * cp)
    hpf = amp * (-aplus * cos2psi * sp + across * sin2psi * cp)
    hcf = amp * (across * cos2psi * cp + aplus * sin2psi * sp)
    return hp, hc, hpf, hcf


@njit(cache=True)
def _tdi_spline_coeff_jit(
    a: int,
    b: int,
    c: int,
    t: float,
    wave_x: np.ndarray,
    amp_coeffs: np.ndarray,
    phase_coeffs: np.ndarray,
    aplus: float,
    across: float,
    cos2psi: float,
    sin2psi: float,
    app: np.ndarray,
    apm: np.ndarray,
    acp: np.ndarray,
    acm: np.ndarray,
    kr: np.ndarray,
    larm: np.ndarray,
) -> tuple[float, float]:
    m = 0.0
    mf = 0.0

    hp, hc, hpf, hcf = _hphc_coeff_jit(t - kr[a] - 2.0 * larm[c] - 2.0 * larm[b], wave_x, amp_coeffs, phase_coeffs, aplus, across, cos2psi, sin2psi)
    coefp = app[c] - apm[b]
    coefc = acp[c] - acm[b]
    m += hp * coefp + hc * coefc
    mf += hpf * coefp + hcf * coefc

    hp, hc, hpf, hcf = _hphc_coeff_jit(t - kr[b] - larm[c] - 2.0 * larm[b], wave_x, amp_coeffs, phase_coeffs, aplus, across, cos2psi, sin2psi)
    coefp = -app[c] + apm[c]
    coefc = -acp[c] + acm[c]
    m += hp * coefp + hc * coefc
    mf += hpf * coefp + hcf * coefc

    hp, hc, hpf, hcf = _hphc_coeff_jit(t - kr[c] - larm[b] - 2.0 * larm[c], wave_x, amp_coeffs, phase_coeffs, aplus, across, cos2psi, sin2psi)
    coefp = apm[b] - app[b]
    coefc = acm[b] - acp[b]
    m += hp * coefp + hc * coefc
    mf += hpf * coefp + hcf * coefc

    hp, hc, hpf, hcf = _hphc_coeff_jit(t - kr[a] - 2.0 * larm[b], wave_x, amp_coeffs, phase_coeffs, aplus, across, cos2psi, sin2psi)
    coefp = -apm[c] + apm[b]
    coefc = -acm[c] + acm[b]
    m += hp * coefp + hc * coefc
    mf += hpf * coefp + hcf * coefc

    hp, hc, hpf, hcf = _hphc_coeff_jit(t - kr[a] - 2.0 * larm[c], wave_x, amp_coeffs, phase_coeffs, aplus, across, cos2psi, sin2psi)
    coefp = app[b] - app[c]
    coefc = acp[b] - acp[c]
    m += hp * coefp + hc * coefc
    mf += hpf * coefp + hcf * coefc

    hp, hc, hpf, hcf = _hphc_coeff_jit(t - kr[c] - larm[b], wave_x, amp_coeffs, phase_coeffs, aplus, across, cos2psi, sin2psi)
    coefp = -apm[b] + app[b]
    coefc = -acm[b] + acp[b]
    m += hp * coefp + hc * coefc
    mf += hpf * coefp + hcf * coefc

    hp, hc, hpf, hcf = _hphc_coeff_jit(t - kr[b] - larm[c], wave_x, amp_coeffs, phase_coeffs, aplus, across, cos2psi, sin2psi)
    coefp = app[c] - apm[c]
    coefc = acp[c] - acm[c]
    m += hp * coefp + hc * coefc
    mf += hpf * coefp + hcf * coefc

    hp, hc, hpf, hcf = _hphc_coeff_jit(t - kr[a], wave_x, amp_coeffs, phase_coeffs, aplus, across, cos2psi, sin2psi)
    coefp = -app[b] + apm[c]
    coefc = -acp[b] + acm[c]
    m += hp * coefp + hc * coefc
    mf += hpf * coefp + hcf * coefc

    return m, mf


@njit(cache=True)
def _fast_response_jit(
    times: np.ndarray,
    const_x: np.ndarray,
    l_coeffs: np.ndarray,
    p_coeffs: np.ndarray,
    v_coeffs: np.ndarray,
    wave_x: np.ndarray,
    amp_coeffs: np.ndarray,
    phase_coeffs: np.ndarray,
    kv: np.ndarray,
    eplus: np.ndarray,
    ecross: np.ndarray,
    aplus: float,
    across: float,
    cos2psi: float,
    sin2psi: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nt = times.size
    x = np.empty(nt, dtype=np.float64)
    xf = np.empty(nt, dtype=np.float64)
    y = np.empty(nt, dtype=np.float64)
    yf = np.empty(nt, dtype=np.float64)
    z = np.empty(nt, dtype=np.float64)
    zf = np.empty(nt, dtype=np.float64)

    larm = np.empty(3, dtype=np.float64)
    kr = np.empty(3, dtype=np.float64)
    kn = np.empty(3, dtype=np.float64)
    app = np.empty(3, dtype=np.float64)
    apm = np.empty(3, dtype=np.float64)
    acp = np.empty(3, dtype=np.float64)
    acm = np.empty(3, dtype=np.float64)
    arms = np.empty((3, 3), dtype=np.float64)

    for n in range(nt):
        t = times[n]
        interval = _spline_interval_jit(const_x, t)
        for i in range(3):
            larm[i] = _spline_eval_stack_at_jit(const_x, l_coeffs, i, t, interval)
            p0 = _spline_eval_stack_at_jit(const_x, p_coeffs, 3 * i, t, interval)
            p1 = _spline_eval_stack_at_jit(const_x, p_coeffs, 3 * i + 1, t, interval)
            p2 = _spline_eval_stack_at_jit(const_x, p_coeffs, 3 * i + 2, t, interval)
            arms[i, 0] = _spline_eval_stack_at_jit(const_x, v_coeffs, 3 * i, t, interval)
            arms[i, 1] = _spline_eval_stack_at_jit(const_x, v_coeffs, 3 * i + 1, t, interval)
            arms[i, 2] = _spline_eval_stack_at_jit(const_x, v_coeffs, 3 * i + 2, t, interval)
            kr[i] = p0 * kv[0] + p1 * kv[1] + p2 * kv[2]
            kn[i] = arms[i, 0] * kv[0] + arms[i, 1] * kv[1] + arms[i, 2] * kv[2]

        for i in range(3):
            plus = 0.0
            cross = 0.0
            for j in range(3):
                for k in range(3):
                    armprod = arms[i, j] * arms[i, k]
                    plus += armprod * eplus[j, k]
                    cross += armprod * ecross[j, k]
            app[i] = 0.5 * plus / (1.0 + kn[i])
            apm[i] = 0.5 * plus / (1.0 - kn[i])
            acp[i] = 0.5 * cross / (1.0 + kn[i])
            acm[i] = 0.5 * cross / (1.0 - kn[i])

        x[n], xf[n] = _tdi_spline_coeff_jit(0, 1, 2, t, wave_x, amp_coeffs, phase_coeffs, aplus, across, cos2psi, sin2psi, app, apm, acp, acm, kr, larm)
        y[n], yf[n] = _tdi_spline_coeff_jit(1, 2, 0, t, wave_x, amp_coeffs, phase_coeffs, aplus, across, cos2psi, sin2psi, app, apm, acp, acm, kr, larm)
        z[n], zf[n] = _tdi_spline_coeff_jit(2, 0, 1, t, wave_x, amp_coeffs, phase_coeffs, aplus, across, cos2psi, sin2psi, app, apm, acp, acm, kr, larm)

    return x, xf, y, yf, z, zf


@njit(cache=True)
def _spacecraft0_kdot_jit(t: float, const_x: np.ndarray, p_coeffs: np.ndarray, kv: np.ndarray) -> float:
    interval = _spline_interval_jit(const_x, t)
    x0 = _spline_eval_stack_at_jit(const_x, p_coeffs, 0, t, interval)
    x1 = _spline_eval_stack_at_jit(const_x, p_coeffs, 1, t, interval)
    x2 = _spline_eval_stack_at_jit(const_x, p_coeffs, 2, t, interval)
    return x0 * kv[0] + x1 * kv[1] + x2 * kv[2]


@njit(cache=True)
def _barycenter_time_jit(detector_time: np.ndarray, const_x: np.ndarray, p_coeffs: np.ndarray, kv: np.ndarray) -> np.ndarray:
    out = np.empty(detector_time.size, dtype=np.float64)
    for n in range(detector_time.size):
        tspace = detector_time[n]
        tb = tspace + _spacecraft0_kdot_jit(tspace, const_x, p_coeffs, kv)
        out[n] = tspace + _spacecraft0_kdot_jit(tb, const_x, p_coeffs, kv)
    return out


@njit(cache=True)
def _barycenter_time_scalar_jit(tspace: float, const_x: np.ndarray, p_coeffs: np.ndarray, kv: np.ndarray) -> float:
    tb = tspace + _spacecraft0_kdot_jit(tspace, const_x, p_coeffs, kv)
    return tspace + _spacecraft0_kdot_jit(tb, const_x, p_coeffs, kv)


@njit(cache=True)
def _spacecraft0_reference_time_jit(tarray: np.ndarray, const_x: np.ndarray, p_coeffs: np.ndarray, kv: np.ndarray) -> np.ndarray:
    out = np.empty(tarray.size, dtype=np.float64)
    for n in range(tarray.size):
        t = tarray[n]
        out[n] = t - _spacecraft0_kdot_jit(t, const_x, p_coeffs, kv)
    return out


def eval_spacecraft0_position(p_splines: list[CubicSpline], t: float) -> np.ndarray:
    return np.array([p_splines[0](t), p_splines[1](t), p_splines[2](t)], dtype=np.float64)


def ssb_output_time_from_center_source_time(
    center_source_time: np.ndarray,
    source: SourceParams,
    p_splines: list[CubicSpline],
) -> np.ndarray:
    """Invert u=t-k.r_0(t), returning SSB output time t for each u."""

    _, _, kv = sky_vectors(source)
    if NUMBA_AVAILABLE:
        return _barycenter_time_jit(
            np.ascontiguousarray(center_source_time, dtype=np.float64),
            np.ascontiguousarray(p_splines[0].x, dtype=np.float64),
            _stack_spline_coefficients(p_splines),
            np.ascontiguousarray(kv, dtype=np.float64),
        )
    out = np.empty_like(center_source_time)
    for n, tspace in enumerate(center_source_time):
        x = eval_spacecraft0_position(p_splines, float(tspace))
        tb = float(tspace) + float(np.dot(x, kv))
        x = eval_spacecraft0_position(p_splines, tb)
        out[n] = float(tspace) + float(np.dot(x, kv))
    return out


def barycenter_time(
    detector_time: np.ndarray,
    source: SourceParams,
    p_splines: list[CubicSpline],
) -> np.ndarray:
    """Compatibility name for :func:`ssb_output_time_from_center_source_time`."""

    return ssb_output_time_from_center_source_time(detector_time, source, p_splines)


def _barycenter_time_scalar_from_coeffs(tspace: float, const_x: np.ndarray, p_coeffs: np.ndarray, kv: np.ndarray) -> float:
    return float(_barycenter_time_scalar_jit(float(tspace), const_x, p_coeffs, kv))


def spacecraft0_reference_time(tarray: np.ndarray, source: object, p_splines: list[CubicSpline]) -> np.ndarray:
    """Reference time used by response.c for the spacecraft-0 carrier phase."""

    _, _, kv = sky_vectors(source)
    if NUMBA_AVAILABLE:
        return _spacecraft0_reference_time_jit(
            np.ascontiguousarray(tarray, dtype=np.float64),
            np.ascontiguousarray(p_splines[0].x, dtype=np.float64),
            _stack_spline_coefficients(p_splines),
            np.ascontiguousarray(kv, dtype=np.float64),
        )
    out = np.empty_like(tarray)
    for n, t in enumerate(tarray):
        x = eval_spacecraft0_position(p_splines, float(t))
        out[n] = float(t) - float(np.dot(x, kv))
    return out


def _frequency_hz_at_source_time(model: IMRPhenomT22, source_time: float, tc: float, total_mass: float) -> float:
    omega = model.omega22((source_time - tc) / total_mass)
    if not math.isfinite(omega):
        return 0.0
    return abs(omega) / (2.0 * PI * total_mass)


def _frequency_hz_at_ssb_knot_from_center_time(
    model: IMRPhenomT22,
    detector_time: float,
    tc: float,
    total_mass: float,
    const_x: np.ndarray,
    p_coeffs: np.ndarray,
    kv: np.ndarray,
) -> float:
    source_time = _barycenter_time_scalar_from_coeffs(detector_time, const_x, p_coeffs, kv)
    return _frequency_hz_at_source_time(model, source_time, tc, total_mass)


def _transfer_interval_fails(f0: float, fmid: float, f1: float, width: float, fmode_max: float) -> bool:
    if width <= DTMIN or FSTAR <= 0.0:
        return False
    f0 = abs(f0)
    fmid = abs(fmid)
    f1 = abs(f1)
    fmin = min(f0, fmid, f1)
    fmax = max(f0, fmid, f1)
    if not math.isfinite(fmin) or not math.isfinite(fmax) or fmax <= 0.0:
        return False
    if not math.isfinite(fmode_max) or fmode_max <= 0.0:
        fmode_max = fmax
    # fstar is the one-radian delay-phase scale, omega*L = f/fstar.  The
    # equal-arm delay zeros are at f = n*pi*fstar, not at n*fstar.
    zero_spacing = PI * FSTAR
    nmin = max(1, int(math.floor(fmin / zero_spacing)) - 1)
    nmax = int(math.floor(fmode_max / zero_spacing))
    if nmax < nmin:
        return False
    for n in range(nmin, nmax + 1):
        fc = float(n) * zero_spacing
        if fmin <= fc <= fmax:
            fscale = max(fmax, fc)
            max_width = AP_TRANSFER_PHASE_STEP / (2.0 * PI * fscale)
            if max_width < DTMIN:
                max_width = DTMIN
            if width > max_width:
                return True
    return False


def _frequency_interval_ok(f0: float, fmid: float, f1: float, width: float, fmode_max: float) -> bool:
    f0 = abs(f0)
    fmid = abs(fmid)
    f1 = abs(f1)
    ferr = abs(fmid - 0.5 * (f0 + f1))
    fscale = max(f0, fmid, f1, 1.0e-8)
    if abs(f1 - f0) / fscale > AP_FREQ_STEP_REL_TOL:
        return False
    if _transfer_interval_fails(f0, fmid, f1, width, fmode_max):
        return False
    phase_err = 0.5 * PI * ferr * width
    if phase_err > AP_PHASE_CURVATURE_TOL:
        return False
    if ferr / fscale > AP_FREQ_REL_CURVATURE_TOL:
        return False
    return True


def _detector_interval_ok(
    model: IMRPhenomT22,
    t0: float,
    t1: float,
    tc: float,
    total_mass: float,
    fmode_max: float,
    const_x: np.ndarray,
    p_coeffs: np.ndarray,
    kv: np.ndarray,
) -> bool:
    width = t1 - t0
    if width <= DTMIN:
        return True
    tmid = 0.5 * (t0 + t1)

    # The returned AP knots are labelled by SSB output time.  Protect that
    # intrinsic interpolation grid first; its merger location differs from the
    # detector-center response merger by k.r_0, which can be hundreds of
    # seconds.
    f0 = _frequency_hz_at_ssb_knot_from_center_time(model, t0, tc, total_mass, const_x, p_coeffs, kv)
    f1 = _frequency_hz_at_ssb_knot_from_center_time(model, t1, tc, total_mass, const_x, p_coeffs, kv)
    fmid = _frequency_hz_at_ssb_knot_from_center_time(model, tmid, tc, total_mass, const_x, p_coeffs, kv)
    if not _frequency_interval_ok(f0, fmid, f1, width, fmode_max):
        return False

    # The TDI response itself samples h(t-k.r-Delta L).  In this planner t0,
    # tmid, and t1 are the guiding-center retarded/source times, so apply the
    # nominal arm delays directly to them.  Mapping this family to SSB time as
    # well would leave the physical detector-frame merger under-resolved.
    lnom = 1.0 / (2.0 * PI * FSTAR)
    for offset in (0.0, -lnom, -2.0 * lnom, -4.0 * lnom):
        f0 = _frequency_hz_at_source_time(model, t0 + offset, tc, total_mass)
        f1 = _frequency_hz_at_source_time(model, t1 + offset, tc, total_mass)
        fmid = _frequency_hz_at_source_time(model, tmid + offset, tc, total_mass)
        if not _frequency_interval_ok(f0, fmid, f1, width, fmode_max):
            return False
    return True


def _detector_adaptive_step(
    model: IMRPhenomT22,
    t: float,
    tstop: float,
    proposed_step: float,
    tc: float,
    total_mass: float,
    fmode_max: float,
    const_x: np.ndarray,
    p_coeffs: np.ndarray,
    kv: np.ndarray,
) -> float:
    step = proposed_step
    if not math.isfinite(step) or step <= 0.0:
        step = AP_DTM_MAX
    if step > AP_DTM_MAX:
        step = AP_DTM_MAX

    knot_frequency = _frequency_hz_at_ssb_knot_from_center_time(
        model, t, tc, total_mass, const_x, p_coeffs, kv
    )
    lnom = 1.0 / (2.0 * PI * FSTAR)
    response_frequency = max(
        _frequency_hz_at_source_time(model, t + offset, tc, total_mass)
        for offset in (0.0, -lnom, -2.0 * lnom, -4.0 * lnom)
    )
    max_omega = 2.0 * PI * max(knot_frequency, response_frequency)
    if max_omega > 0.0 and step > AP_MAX_PHASE_STEP / max_omega:
        step = AP_MAX_PHASE_STEP / max_omega
    if step < DTMIN:
        step = DTMIN
    if t + step > tstop:
        step = tstop - t
    if step < DTMIN:
        step = DTMIN

    if (
        step > DTMIN
        and t + step <= tstop
        and not _detector_interval_ok(model, t, t + step, tc, total_mass, fmode_max, const_x, p_coeffs, kv)
    ):
        failed_step = step
        while True:
            passing_step = max(DTMIN, failed_step * AP_STEP_SHRINK_FACTOR)
            if _detector_interval_ok(
                model, t, t + passing_step, tc, total_mass, fmode_max, const_x, p_coeffs, kv
            ):
                break
            failed_step = passing_step
            if passing_step <= DTMIN:
                break

        for _ in range(AP_STEP_BRACKET_ITERATIONS):
            if failed_step - passing_step <= 1.0e-6 * max(1.0, passing_step):
                break
            trial_step = 0.5 * (passing_step + failed_step)
            if _detector_interval_ok(
                model, t, t + trial_step, tc, total_mass, fmode_max, const_x, p_coeffs, kv
            ):
                passing_step = trial_step
            else:
                failed_step = trial_step
        step = passing_step
    return step


def _detector_step_proposal(last_step: float, previous_step: float, history_count: int) -> float:
    """Continue a decreasing spacing trend smoothly into the merger."""

    if not math.isfinite(last_step) or last_step <= 0.0:
        return AP_DTM_MAX
    if history_count >= 2 and math.isfinite(previous_step) and previous_step > 0.0 and last_step < previous_step:
        ratio = min(1.0, max(1.0 / AP_STEP_GROWTH_LIMIT, last_step / previous_step))
        return last_step * ratio
    return AP_STEP_GROWTH_LIMIT * last_step


def build_detector_adaptive_grid(
    source: SourceParams,
    model: IMRPhenomT22,
    total_mass: float,
    shape: WDMShape,
    p_splines: list[CubicSpline],
    constellation_tmin: float,
    constellation_tmax: float,
    nsmax: int,
) -> np.ndarray:
    _, _, kv = sky_vectors(source)
    const_x = np.ascontiguousarray(p_splines[0].x, dtype=np.float64)
    p_coeffs = _stack_spline_coefficients(p_splines)
    kv = np.ascontiguousarray(kv, dtype=np.float64)

    tstart = max(0.0, constellation_tmin + CONSTELLATION_LIGHT_TIME_SECONDS)
    tstop = min(
        shape.Tobs,
        source.tc + RESPONSE_LATE_MARGIN_SECONDS + 1000.0 * total_mass,
        constellation_tmax - CONSTELLATION_LIGHT_TIME_SECONDS,
    )
    if tstop <= tstart:
        raise ValueError(f"detector adaptive grid has empty support [{tstart:.15e}, {tstop:.15e}]")

    fmode_max = abs(model.omega_ring) / (2.0 * PI * total_mass)
    detector_time: list[float] = []
    t = tstart
    last_step = AP_DTM_MAX
    previous_step = AP_DTM_MAX
    history_count = 0
    while True:
        if len(detector_time) >= nsmax:
            raise RuntimeError("detector-adaptive waveform grid exceeded nsmax")
        detector_time.append(t)
        if t >= tstop:
            break
        step = _detector_adaptive_step(
            model,
            t,
            tstop,
            _detector_step_proposal(last_step, previous_step, history_count),
            source.tc,
            total_mass,
            fmode_max,
            const_x,
            p_coeffs,
            kv,
        )
        tnext = t + step
        if tnext <= t:
            tnext = t + DTMIN
        if tnext > tstop:
            tnext = tstop
        previous_step = last_step
        last_step = tnext - t
        history_count += 1
        t = tnext

    return np.asarray(detector_time, dtype=np.float64)


def transform_plan(
    chirp_mass: float,
    total_mass: float,
    tc: float,
    response_time: np.ndarray,
    omega_response: np.ndarray,
    t22: float,
    fring: float,
    fdamp: float,
    tmax: float,
    shape: WDMShape,
    *,
    center_source_time: np.ndarray | None = None,
) -> np.ndarray:
    """Plan the endpoint FFT on the SSB output-time coordinate.

    ``tc`` and ``t22`` are intrinsic/source times.  When the response is
    sampled at SSB times, map those events through the detector-center
    retarded coordinate before comparing them with ``response_time``.
    """

    tc_output = float(tc)
    t22_output = float(t22)
    if center_source_time is not None:
        center_source_time = np.asarray(center_source_time, dtype=np.float64)
        if center_source_time.shape != response_time.shape:
            raise ValueError("center_source_time must match response_time")
        if np.any(np.diff(center_source_time) <= 0.0):
            raise ValueError("center_source_time must be strictly increasing")
        tc_output = float(np.interp(tc, center_source_time, response_time))
        t22_output = float(np.interp(t22, center_source_time, response_time))

    setup = np.zeros(7, dtype=np.float64)
    dte = 1.0 / (6.0 * fring)
    if dte < shape.dt:
        dte = shape.dt
    dte = shape.dt * math.floor(dte / shape.dt)
    setup[0] = dte

    rise = 1000.0 * chirp_mass
    tes0 = t22_output - 2.0 * rise
    tes0 = min(tes0, t22_output - RESPONSE_EARLY_MARGIN_SECONDS - rise)
    tee = tc_output + RESPONSE_LATE_MARGIN_SECONDS + 1000.0 * total_mass
    damping_rate = abs(fdamp)
    if damping_rate > 0.0:
        tee = max(tee, tc_output + 10.0 / damping_rate)
    tee = max(tee, tmax)

    nend = math.floor(math.pow(2.0, math.floor(math.log2((tee - tes0) / dte)) + 1.0))
    setup[1] = nend
    tspan = nend * dte
    tes = dte * round(tee / dte) - tspan
    setup[2] = tes
    setup[3] = rise

    tjoin_target = t22_output - 0.25 * rise
    jj = int(np.searchsorted(response_time, tjoin_target, side="left"))
    jj = min(max(jj, 1), response_time.size - 2)
    setup[4] = float(jj)
    # Keep the continuous requested join time.  The AP-grid index above is
    # only for array slicing; using it as the frequency boundary causes whole
    # WDM layers to jump when a nearby adaptive knot moves.
    setup[5] = tjoin_target
    setup[6] = math.ceil(1.5 * fring * tspan)
    return setup


def _first_upward_crossing_time(
    time: np.ndarray,
    frequency: np.ndarray,
    threshold: float,
    stop_time: float,
) -> float | None:
    """Return the first upward crossing of ``threshold`` before ``stop_time``."""

    valid = np.isfinite(time) & np.isfinite(frequency) & (time <= stop_time)
    indices = np.flatnonzero(valid)
    if indices.size == 0:
        return None
    first = int(indices[0])
    if float(frequency[first]) >= threshold:
        return float(time[first])
    for left, right in zip(indices[:-1], indices[1:]):
        f0 = float(frequency[left])
        f1 = float(frequency[right])
        if f0 < threshold <= f1 and f1 > f0:
            fraction = (threshold - f0) / (f1 - f0)
            return float(time[left]) + fraction * float(time[right] - time[left])
    return None


def plan_endpoint_taper_flat_time(
    setup: np.ndarray,
    time: np.ndarray,
    channel_tracks: Sequence[Sequence[np.ndarray]],
    blend_lower_frequencies: Sequence[float],
    merger_time: float,
) -> np.ndarray:
    """Make the common endpoint taper flat before any blend can begin.

    For each channel, find the earliest carrier that reaches the lower edge of
    its frequency blend on the pre-merger rise.  The endpoint taper must be
    flat by the earliest such crossing.  Normally the power-of-two interval
    already starts early enough and only the roll-on is shortened.  Otherwise
    the interval is expanded while preserving its original late boundary.
    """

    planned = np.asarray(setup, dtype=np.float64).copy()
    crossing_times: list[float] = []
    for tracks, threshold in zip(channel_tracks, blend_lower_frequencies):
        for track in tracks:
            crossing = _first_upward_crossing_time(
                time, np.asarray(track, dtype=np.float64), float(threshold), merger_time
            )
            if crossing is not None:
                crossing_times.append(crossing)
    if not crossing_times:
        return planned

    target_flat = min(crossing_times)
    dte = float(planned[0])
    nshort = int(planned[1])
    start = float(planned[2])
    original_rise = float(planned[3])
    available_rise = target_flat - start
    if available_rise >= dte:
        planned[3] = min(original_rise, available_rise)
        return planned

    old_stop = start + dte * float(nshort)
    requested_start = target_flat - original_rise
    required = max(int(math.ceil((old_stop - requested_start) / dte)), nshort)
    expanded = 1
    while expanded < required:
        expanded *= 2
    old_frequency_stop = float(planned[6]) / (dte * float(nshort))
    planned[1] = float(expanded)
    planned[2] = dte * round(old_stop / dte) - dte * float(expanded)
    planned[3] = min(original_rise, target_flat - float(planned[2]))
    planned[6] = math.ceil(old_frequency_stop * dte * float(expanded))
    return planned


def _intrinsic_interval_ok(
    model: IMRPhenomT22,
    t0: float,
    t1: float,
    tc: float,
    total_mass: float,
) -> bool:
    """Derivative-free source-grid test used by the production C planner."""

    width = t1 - t0
    if width <= DTMIN:
        return True
    tmid = 0.5 * (t0 + t1)
    f0 = _frequency_hz_at_source_time(model, t0, tc, total_mass)
    fmid = _frequency_hz_at_source_time(model, tmid, tc, total_mass)
    f1 = _frequency_hz_at_source_time(model, t1, tc, total_mass)
    ferr = abs(fmid - 0.5 * (f0 + f1))
    fscale = max(abs(f0), abs(fmid), abs(f1), 1.0e-8)
    if abs(f1 - f0) / fscale > INTRINSIC_FREQ_STEP_REL_TOL:
        return False
    if 0.5 * PI * ferr * width > INTRINSIC_PHASE_CURVATURE_TOL:
        return False
    if ferr / fscale > INTRINSIC_FREQ_REL_CURVATURE_TOL:
        return False

    tau0 = (t0 - tc) / total_mass
    taumid = (tmid - tc) / total_mass
    tau1 = (t1 - tc) / total_mass
    a0 = abs(model.amplitude_complex(tau0))
    amid = abs(model.amplitude_complex(taumid))
    a1 = abs(model.amplitude_complex(tau1))
    ascale = max(a0, amid, a1)
    if math.isfinite(ascale) and ascale > 0.0:
        if abs(a1 - a0) / ascale > INTRINSIC_AMP_STEP_REL_TOL:
            return False
        if abs(amid - 0.5 * (a0 + a1)) / ascale > INTRINSIC_AMP_REL_CURVATURE_TOL:
            return False
    return True


def _intrinsic_adaptive_step(
    model: IMRPhenomT22,
    t: float,
    tstop: float,
    proposed_step: float,
    tc: float,
    total_mass: float,
) -> float:
    step = proposed_step
    if not math.isfinite(step) or step <= 0.0:
        step = INTRINSIC_DTM_MAX
    step = min(step, INTRINSIC_DTM_MAX)

    fnow = _frequency_hz_at_source_time(model, t, tc, total_mass)
    max_omega = 2.0 * PI * fnow
    if max_omega > 0.0:
        step = min(step, INTRINSIC_MAX_PHASE_STEP / max_omega)
    step = max(step, DTMIN)
    if t + step > tstop:
        step = max(tstop - t, DTMIN)

    while step > DTMIN and t + step <= tstop and not _intrinsic_interval_ok(
        model, t, t + step, tc, total_mass
    ):
        step = max(DTMIN, step * INTRINSIC_STEP_SHRINK_FACTOR)
    return step


def _unique_sorted_times(values: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if ordered.size < 2:
        return ordered
    keep = np.ones(ordered.size, dtype=bool)
    last = float(ordered[0])
    for i in range(1, ordered.size):
        value = float(ordered[i])
        tol = 1.0e-7 + 1.0e-13 * abs(value)
        if abs(value - last) <= tol:
            keep[i] = False
        else:
            last = value
    return ordered[keep]


def _build_intrinsic_model_grid(
    model: IMRPhenomT22,
    source_start: float,
    source_stop: float,
    tc: float,
    total_mass: float,
    capacity: int,
) -> np.ndarray:
    points: list[float] = []
    t = source_start
    last_step = INTRINSIC_DTM_MAX
    while True:
        if len(points) >= capacity:
            raise RuntimeError("intrinsic waveform grid exceeded capacity")
        points.append(t)
        if t >= source_stop:
            break
        step = _intrinsic_adaptive_step(
            model,
            t,
            source_stop,
            INTRINSIC_STEP_GROWTH_LIMIT * last_step,
            tc,
            total_mass,
        )
        tnext = min(t + step, source_stop)
        if tnext <= t:
            tnext = min(t + DTMIN, source_stop)
        last_step = tnext - t
        t = tnext
    return _unique_sorted_times(np.asarray(points, dtype=np.float64))


def _intrinsic_tdi_switch_source_time(
    model: IMRPhenomT22,
    model_time: np.ndarray,
    source_start: float,
    source_stop: float,
    tc: float,
    total_mass: float,
) -> float:
    trigger = INTRINSIC_TDI_SWITCH_TRANSFER_FRACTION * PI * FSTAR
    switch_time = math.inf
    fprev = _frequency_hz_at_source_time(model, float(model_time[0]), tc, total_mass)
    for i in range(1, model_time.size):
        t0 = float(model_time[i - 1])
        t1 = float(model_time[i])
        if 0.0 < t1 - t0 < AP_DTM_MAX:
            switch_time = t0
            break
        fnow = _frequency_hz_at_source_time(model, t1, tc, total_mass)
        if math.isfinite(fprev) and math.isfinite(fnow) and (
            (fprev < trigger <= fnow) or (fprev >= trigger and fnow > 0.0)
        ):
            if fnow != fprev:
                u = min(1.0, max(0.0, (trigger - fprev) / (fnow - fprev)))
                switch_time = t0 + u * (t1 - t0)
            else:
                switch_time = t0
            break
        fprev = fnow

    switch_time = min(switch_time, source_stop - INTRINSIC_TDI_SWITCH_TAIL_SECONDS)
    if not math.isfinite(switch_time):
        switch_time = source_start
    return min(source_stop, max(source_start, switch_time))


def _build_hybrid_response_grid(
    source: SourceParams,
    model: IMRPhenomT22,
    total_mass: float,
    shape: WDMShape,
    p_splines: list[CubicSpline],
    constellation_tmin: float,
    constellation_tmax: float,
    model_time: np.ndarray,
    nsmax: int,
) -> tuple[np.ndarray, float]:
    _, _, kv = sky_vectors(source)
    const_x = np.ascontiguousarray(p_splines[0].x, dtype=np.float64)
    p_coeffs = _stack_spline_coefficients(p_splines)
    kv = np.ascontiguousarray(kv, dtype=np.float64)

    tstart = max(0.0, constellation_tmin + CONSTELLATION_LIGHT_TIME_SECONDS)
    tstop = min(
        shape.Tobs,
        source.tc + RESPONSE_LATE_MARGIN_SECONDS + 1000.0 * total_mass,
        constellation_tmax - CONSTELLATION_LIGHT_TIME_SECONDS,
    )
    switch_source = _intrinsic_tdi_switch_source_time(
        model, model_time, float(model_time[0]), float(model_time[-1]), source.tc, total_mass
    )
    switch_detector = min(tstop, max(tstart, switch_source - INTRINSIC_TDI_SWITCH_GUARD_SECONDS))
    fmode_max = abs(model.omega_ring) / (2.0 * PI * total_mass)

    points: list[float] = []
    t = tstart
    last_step = AP_DTM_MAX
    previous_step = AP_DTM_MAX
    history_count = 0
    while True:
        if len(points) >= nsmax:
            raise RuntimeError("hybrid detector-response grid exceeded nsmax")
        points.append(t)
        if t >= tstop:
            break
        if t < switch_detector:
            step = min(AP_DTM_MAX, switch_detector - t)
        else:
            step = _detector_adaptive_step(
                model,
                t,
                tstop,
                _detector_step_proposal(last_step, previous_step, history_count),
                source.tc,
                total_mass,
                fmode_max,
                const_x,
                p_coeffs,
                kv,
            )
        tnext = min(t + max(step, DTMIN), tstop)
        if tnext <= t:
            tnext = min(t + DTMIN, tstop)
        if t >= switch_detector:
            previous_step = last_step
            last_step = tnext - t
            history_count += 1
        t = tnext
    return np.asarray(points, dtype=np.float64), switch_detector


def build_intrinsic_grid(
    source: SourceParams,
    shape: WDMShape,
    p_splines: list[CubicSpline],
    constellation_tmin: float,
    constellation_tmax: float,
    nsmax: int = 10000,
    coefficient_backend: str = "auto",
    intrinsic_backend: str = "local",
) -> IntrinsicGrid:
    """Build the current production sparse-intrinsic/hybrid-response AP grid."""

    m1, m2, chi1, chi2 = source.masses_seconds()
    if intrinsic_backend == "local":
        model = IMRPhenomT22.from_masses(m1, m2, chi1, chi2, coefficient_backend=coefficient_backend)
    elif intrinsic_backend == "phentax":
        # Import lazily so the normal NumPy/Numba path has no JAX dependency
        # or startup cost. Phentax22Adapter presents the same arbitrary-time
        # 2,2 amplitude/frequency/phase interface used by this planner.
        from phentax22_adapter import Phentax22Adapter

        model = Phentax22Adapter.from_masses(m1, m2, chi1, chi2)
    else:
        raise ValueError("intrinsic_backend must be 'local' or 'phentax'")
    planning_model = getattr(model, "planning_model", model)
    total_mass = m1 + m2
    eta = m1 * m2 / (total_mass * total_mass)
    chirp_mass = (m1 * m2) ** (3.0 / 5.0) / (total_mass ** (1.0 / 5.0))
    tc = source.tc

    tstart = max(0.0, constellation_tmin + CONSTELLATION_LIGHT_TIME_SECONDS)
    tstop = min(
        shape.Tobs,
        tc + RESPONSE_LATE_MARGIN_SECONDS + 1000.0 * total_mass,
        constellation_tmax - CONSTELLATION_LIGHT_TIME_SECONDS,
    )
    detector_bounds = np.asarray([tstart, tstop], dtype=np.float64)
    source_bounds = barycenter_time(detector_bounds, source, p_splines)
    model_time = _build_intrinsic_model_grid(
        planning_model,
        float(source_bounds[0]),
        float(source_bounds[1]),
        tc,
        total_mass,
        2 * nsmax + 4,
    )
    detector_time, switch_detector = _build_hybrid_response_grid(
        source,
        planning_model,
        total_mass,
        shape,
        p_splines,
        constellation_tmin,
        constellation_tmax,
        model_time,
        nsmax,
    )
    bary_time = barycenter_time(detector_time, source, p_splines)

    exact = np.zeros(detector_time.size, dtype=bool)
    exact[0] = True
    exact[-1] = True
    spacing = np.diff(detector_time)
    eps = 1.0e-10 * AP_DTM_MAX
    exact[:-1] |= spacing < AP_DTM_MAX - eps
    exact[1:] |= spacing < AP_DTM_MAX - eps
    model_time = _unique_sorted_times(np.concatenate([model_time, bary_time[exact]]))

    tau_model = (model_time - tc) / total_mass
    evaluated = model.evaluate_tau(tau_model)
    omega_model = np.asarray(evaluated["omega22"], dtype=np.float64) / total_mass
    amp_scale = math.sqrt(2.0) * eta * total_mass / (source.distance_gpc * GPSEC)
    amplitude_model = amp_scale * np.asarray(evaluated["amp_abs"], dtype=np.float64)
    frequency_model = omega_model / (2.0 * PI)

    omega_spline = CubicSpline(model_time, omega_model, bc_type="natural")
    phase_model = np.zeros_like(model_time)
    phase_model[0] = model.phase22(float(tau_model[0])) + source.phic - model.phase22(0.0)
    for i in range(1, model_time.size):
        phase_model[i] = phase_model[i - 1] + omega_spline.integrate(
            float(model_time[i - 1]), float(model_time[i])
        )

    amplitude = np.asarray(make_ap_spline(model_time, amplitude_model)(bary_time), dtype=np.float64)
    phase = np.asarray(make_ap_spline(model_time, phase_model)(bary_time), dtype=np.float64)
    frequency = np.asarray(make_ap_spline(model_time, frequency_model)(bary_time), dtype=np.float64)

    fring = model.omega_ring / (2.0 * PI * total_mass)
    fdamp = model.alpha1rd / total_mass
    setup = transform_plan(
        chirp_mass,
        total_mass,
        tc,
        bary_time,
        2.0 * PI * frequency,
        tc + total_mass * T_CUT_FREQ,
        fring,
        fdamp,
        float(bary_time[-1]),
        shape,
        center_source_time=detector_time,
    )
    return IntrinsicGrid(
        bary_time,
        detector_time,
        amplitude,
        phase,
        frequency,
        setup,
        model,
        model_time=model_time,
        response_switch_detector_time=switch_detector,
        exact_response_samples=int(np.count_nonzero(exact)),
    )


def build_ucb_intrinsic_grid(source: UCBSourceParams, shape: WDMShape) -> IntrinsicGrid:
    """Build the sparse galactic-binary AP grid used by response.c."""

    ns = int(200.0 * shape.Tobs / SECSYR)
    if ns < 20:
        ns = 20
    dtx = shape.Tobs / float(ns - 1)
    dtc = (shape.Tobs + 2.0 * dtx) / float(ns - 1)
    waveform_time = -dtx + dtc * np.arange(ns, dtype=np.float64)
    response_time = dtx * np.arange(ns, dtype=np.float64)
    phase, amplitude, frequency = source.evaluate(waveform_time)
    kx, kw = ucb_wdm_band(source, shape)
    setup = np.array([float(kx), float(kw), float(kx) * shape.DF, shape.DT / float(kw + 1)], dtype=np.float64)
    return IntrinsicGrid(waveform_time, response_time, amplitude, phase, frequency, setup, None)


def build_eccentric_ucb_carrier_grids(
    source: EccentricUCBSourceParams,
    shape: WDMShape,
    carrier_labels: Iterable[str] | None = None,
) -> list[EccentricCarrierGrid]:
    """Build sparse AP grids for all non-negligible eccentric carriers."""

    ns = int(200.0 * shape.Tobs / SECSYR)
    if ns < 20:
        ns = 20
    dtx = shape.Tobs / float(ns - 1)
    dtc = (shape.Tobs + 2.0 * dtx) / float(ns - 1)
    waveform_time = -dtx + dtc * np.arange(ns, dtype=np.float64)
    response_time = dtx * np.arange(ns, dtype=np.float64)
    carriers = source.carrier_grids(waveform_time, response_time, shape)
    if carrier_labels is not None:
        requested = {label.lower() for label in carrier_labels}
        carriers = [carrier for carrier in carriers if carrier.label.lower() in requested]
    if not carriers:
        raise ValueError("eccentric source produced no active carriers")
    return carriers


def compute_ucb_tdi_grid(
    source: UCBSourceParams | EccentricUCBSourceParams,
    intrinsic: IntrinsicGrid,
    l_splines: list[CubicSpline],
    p_splines: list[CubicSpline],
    v_splines: list[CubicSpline],
    aplus_override: float | None = None,
    across_override: float | None = None,
    tdi_generation: int = 1,
) -> TDIGrid:
    """Apply the fast TDI response to the galactic-binary AP spline."""

    amp_spline = CubicSpline(intrinsic.bary_time, intrinsic.amplitude, bc_type="natural")
    phase_spline = CubicSpline(intrinsic.bary_time, intrinsic.phase, bc_type="natural")
    times = intrinsic.detector_time
    reference_time = spacecraft0_reference_time(times, source, p_splines)
    phi_r = phase_spline(reference_time)
    phi_b = phase_spline(times)

    amp: dict[str, np.ndarray] = {}
    offset: dict[str, np.ndarray] = {}
    if tdi_generation == 2:
        from tdi2_response import complex_tdi2

        cosi = source.cos_inclination
        aplus = 0.5*(1.0+cosi*cosi) if aplus_override is None else aplus_override
        across = -cosi if across_override is None else across_override
        cos2psi = math.cos(2.0*source.polarization)
        sin2psi = math.sin(2.0*source.polarization)

        def polarizations(source_time: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            oscillator = amp_spline(source_time)*np.exp(1j*phase_spline(source_time))
            return (aplus*cos2psi-1j*across*sin2psi)*oscillator, \
                   (-aplus*sin2psi-1j*across*cos2psi)*oscillator

        analytic = complex_tdi2(times, math.asin(source.ecliptic_costheta),
                                source.ecliptic_longitude, p_splines,
                                polarizations)
        for channel, name in enumerate("XYZ"):
            amp[name], offset[name] = extract_ap(
                analytic[channel, 0].real, -analytic[channel, 0].imag, phi_r
            )
        return TDIGrid(times, reference_time, phi_r, phi_b, amp, offset)
    if tdi_generation != 1:
        raise ValueError("tdi_generation must be 1 or 2")
    if NUMBA_AVAILABLE:
        u, v, kv = sky_vectors(source)
        eplus = np.ascontiguousarray(np.outer(v, v) - np.outer(u, u), dtype=np.float64)
        ecross = np.ascontiguousarray(np.outer(u, v) + np.outer(v, u), dtype=np.float64)
        psi = source.polarization
        cos2psi = math.cos(2.0 * psi)
        sin2psi = math.sin(2.0 * psi)
        cosi = source.cos_inclination
        aplus = 0.5 * (1.0 + cosi * cosi) if aplus_override is None else aplus_override
        across = -cosi if across_override is None else across_override
        raw_arrays = _fast_response_jit(
            np.ascontiguousarray(times, dtype=np.float64),
            np.ascontiguousarray(l_splines[0].x, dtype=np.float64),
            _stack_spline_coefficients(l_splines),
            _stack_spline_coefficients(p_splines),
            _stack_spline_coefficients(v_splines),
            np.ascontiguousarray(amp_spline.x, dtype=np.float64),
            np.ascontiguousarray(amp_spline.c, dtype=np.float64),
            np.ascontiguousarray(phase_spline.c, dtype=np.float64),
            np.ascontiguousarray(kv, dtype=np.float64),
            eplus,
            ecross,
            aplus,
            across,
            cos2psi,
            sin2psi,
        )
        raw = {"X": (raw_arrays[0], raw_arrays[1]), "Y": (raw_arrays[2], raw_arrays[3]), "Z": (raw_arrays[4], raw_arrays[5])}
        for key, (m, mf) in raw.items():
            amp[key], offset[key] = _extract_ap_jit(m, mf, np.ascontiguousarray(phi_r, dtype=np.float64))
    else:
        raw = fast_response(
            times,
            source,
            l_splines,
            p_splines,
            v_splines,
            amp_spline,
            phase_spline,
            aplus_override=aplus_override,
            across_override=across_override,
        )
        for key, (m, mf) in raw.items():
            amp[key], offset[key] = extract_ap(m, mf, phi_r)
    return TDIGrid(times, reference_time, phi_r, phi_b, amp, offset)


def hphc(
    t: float,
    amp_spline: CubicSpline,
    phase_spline: CubicSpline,
    aplus: float,
    across: float,
    cos2psi: float,
    sin2psi: float,
) -> tuple[float, float, float, float]:
    amp = float(amp_spline(t))
    phase = float(phase_spline(t))
    cp = math.cos(phase)
    sp = math.sin(phase)
    hp = amp * (aplus * cos2psi * cp + across * sin2psi * sp)
    hc = amp * (across * cos2psi * sp - aplus * sin2psi * cp)
    hpf = amp * (-aplus * cos2psi * sp + across * sin2psi * cp)
    hcf = amp * (across * cos2psi * cp + aplus * sin2psi * sp)
    return hp, hc, hpf, hcf


def tdi_spline(
    a: int,
    b: int,
    c: int,
    t: float,
    amp_spline: CubicSpline,
    phase_spline: CubicSpline,
    aplus: float,
    across: float,
    cos2psi: float,
    sin2psi: float,
    app: np.ndarray,
    apm: np.ndarray,
    acp: np.ndarray,
    acm: np.ndarray,
    kr: np.ndarray,
    larm: np.ndarray,
) -> tuple[float, float]:
    terms = (
        (t - kr[a] - 2.0 * larm[c] - 2.0 * larm[b], +app[c], -apm[b], +acp[c], -acm[b]),
        (t - kr[b] - larm[c] - 2.0 * larm[b], -app[c], +apm[c], -acp[c], +acm[c]),
        (t - kr[c] - larm[b] - 2.0 * larm[c], +apm[b], -app[b], +acm[b], -acp[b]),
        (t - kr[a] - 2.0 * larm[b], -apm[c], +apm[b], -acm[c], +acm[b]),
        (t - kr[a] - 2.0 * larm[c], +app[b], -app[c], +acp[b], -acp[c]),
        (t - kr[c] - larm[b], -apm[b], +app[b], -acm[b], +acp[b]),
        (t - kr[b] - larm[c], +app[c], -apm[c], +acp[c], -acm[c]),
        (t - kr[a], -app[b], +apm[c], -acp[b], +acm[c]),
    )
    m = 0.0
    mf = 0.0
    for delay_t, hp1, hp2, hc1, hc2 in terms:
        hp, hc, hpf, hcf = hphc(delay_t, amp_spline, phase_spline, aplus, across, cos2psi, sin2psi)
        m += hp * (hp1 + hp2) + hc * (hc1 + hc2)
        mf += hpf * (hp1 + hp2) + hcf * (hc1 + hc2)
    return m, mf


def fast_response(
    times: np.ndarray,
    source: object,
    l_splines: list[CubicSpline],
    p_splines: list[CubicSpline],
    v_splines: list[CubicSpline],
    amp_spline: CubicSpline,
    phase_spline: CubicSpline,
    aplus_override: float | None = None,
    across_override: float | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    u, v, kv = sky_vectors(source)
    eplus = np.outer(v, v) - np.outer(u, u)
    ecross = np.outer(u, v) + np.outer(v, u)

    psi = source.polarization
    cos2psi = math.cos(2.0 * psi)
    sin2psi = math.sin(2.0 * psi)
    cosi = source.cos_inclination
    aplus = 0.5 * (1.0 + cosi * cosi) if aplus_override is None else aplus_override
    across = -cosi if across_override is None else across_override

    out = {key: (np.zeros(times.size), np.zeros(times.size)) for key in ("X", "Y", "Z")}
    triples = {"X": (0, 1, 2), "Y": (1, 2, 0), "Z": (2, 0, 1)}

    for n, t in enumerate(times):
        larm = np.array([s(float(t)) for s in l_splines], dtype=np.float64)
        pos = np.array([[p_splines[i * 3 + j](float(t)) for j in range(3)] for i in range(3)], dtype=np.float64)
        arms = np.array([[v_splines[i * 3 + j](float(t)) for j in range(3)] for i in range(3)], dtype=np.float64)
        kr = pos @ kv
        kn = arms @ kv
        plus = np.einsum("ij,ik,jk->i", arms, arms, eplus)
        cross = np.einsum("ij,ik,jk->i", arms, arms, ecross)
        app = 0.5 * plus / (1.0 + kn)
        apm = 0.5 * plus / (1.0 - kn)
        acp = 0.5 * cross / (1.0 + kn)
        acm = 0.5 * cross / (1.0 - kn)
        for key, (a, b, c) in triples.items():
            m, mf = tdi_spline(a, b, c, float(t), amp_spline, phase_spline, aplus, across, cos2psi, sin2psi, app, apm, acp, acm, kr, larm)
            out[key][0][n] = m
            out[key][1][n] = mf
    return out


def extract_ap(m: np.ndarray, mf: np.ndarray, reference_phase: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    amp = np.sqrt(m * m + mf * mf)
    flip = np.ones_like(amp)
    pjump = np.zeros_like(amp)
    i = 1
    while i < amp.size - 1:
        flip[i] = flip[i - 1]
        pjump[i] = pjump[i - 1]
        if amp[i] < amp[i - 1] and amp[i] < amp[i + 1]:
            d1 = amp[i + 1] + amp[i - 1] - 2.0 * amp[i]
            d2 = -amp[i + 1] + amp[i - 1] - 2.0 * amp[i]
            d3 = -amp[i + 1] + amp[i - 1] + 2.0 * amp[i]
            if abs(d1) > 0.0 and abs(d2 / d1) < 0.1:
                flip[i + 1] = -flip[i]
                pjump[i + 1] = pjump[i] + PI
                i += 1
            elif abs(d1) > 0.0 and abs(d3 / d1) < 0.1:
                flip[i] = -flip[i - 1]
                pjump[i] = pjump[i - 1] + PI
        i += 1
    flip[-1] = flip[-2]
    pjump[-1] = pjump[-2]
    signed_amp = flip * amp
    phase_offset = -np.arctan2(mf, m) + pjump - np.remainder(reference_phase, 2.0 * PI)
    return signed_amp, unwrap(phase_offset)


def unwrap(phi: np.ndarray) -> np.ndarray:
    out = np.array(phi, dtype=np.float64, copy=True)
    v = out[0]
    for i in range(out.size):
        u = out[i]
        q = round(abs(u - v) / (2.0 * PI))
        if q > 0:
            if v > u:
                u += q * 2.0 * PI
            else:
                u -= q * 2.0 * PI
        v = u
        out[i] = u
    return out


def compute_tdi_grid(source: SourceParams, intrinsic: IntrinsicGrid, l_splines, p_splines, v_splines) -> TDIGrid:
    amp_spline = make_ap_spline(intrinsic.bary_time, intrinsic.amplitude)
    phase_spline = make_ap_spline(intrinsic.bary_time, intrinsic.phase)

    # Match the C trimming loop exactly:
    #   i=0; do { i++; } while(tspace[Ns-i] > TS[Ns-1]); Ns = Ns-i;
    # This drops the first endpoint whose center-source argument is back inside
    # the intrinsic source-time spline range, giving one guard sample for later
    # TDI delays.
    i_trim = 0
    while True:
        i_trim += 1
        if not (intrinsic.detector_time[intrinsic.detector_time.size - i_trim] > intrinsic.bary_time[-1]):
            break
    last = intrinsic.detector_time.size - i_trim
    times = intrinsic.bary_time[:last]
    detector_time = intrinsic.detector_time[:last]

    phi_r = phase_spline(detector_time)
    phi_b = phase_spline(times)

    amp: dict[str, np.ndarray] = {}
    offset: dict[str, np.ndarray] = {}
    # Match the folded (2,+2)+(2,-2) carrier used by PhenomTHM_TDI.c.  The
    # orbital azimuth pi/2 supplies a minus sign to both m=+/-2 modes, while
    # sqrt(5/(64*pi)) is the spin-weighted-spherical-harmonic normalization.
    # These replace the older unnormalized textbook A+/Ax factors.
    swsh22norm = math.sqrt(5.0 / (64.0 * PI))
    cosi = source.cos_inclination
    aplus = -2.0 * swsh22norm * (1.0 + cosi * cosi)
    across = -4.0 * swsh22norm * cosi
    if NUMBA_AVAILABLE:
        u, v, kv = sky_vectors(source)
        eplus = np.ascontiguousarray(np.outer(v, v) - np.outer(u, u), dtype=np.float64)
        ecross = np.ascontiguousarray(np.outer(u, v) + np.outer(v, u), dtype=np.float64)
        psi = source.polarization
        cos2psi = math.cos(2.0 * psi)
        sin2psi = math.sin(2.0 * psi)
        raw_arrays = _fast_response_jit(
            np.ascontiguousarray(times, dtype=np.float64),
            np.ascontiguousarray(l_splines[0].x, dtype=np.float64),
            _stack_spline_coefficients(l_splines),
            _stack_spline_coefficients(p_splines),
            _stack_spline_coefficients(v_splines),
            np.ascontiguousarray(amp_spline.x, dtype=np.float64),
            np.ascontiguousarray(amp_spline.c, dtype=np.float64),
            np.ascontiguousarray(phase_spline.c, dtype=np.float64),
            np.ascontiguousarray(kv, dtype=np.float64),
            eplus,
            ecross,
            aplus,
            across,
            cos2psi,
            sin2psi,
        )
        raw = {"X": (raw_arrays[0], raw_arrays[1]), "Y": (raw_arrays[2], raw_arrays[3]), "Z": (raw_arrays[4], raw_arrays[5])}
        for key, (m, mf) in raw.items():
            amp[key], offset[key] = _extract_ap_jit(m, mf, np.ascontiguousarray(phi_r, dtype=np.float64))
    else:
        raw = fast_response(
            times,
            source,
            l_splines,
            p_splines,
            v_splines,
            amp_spline,
            phase_spline,
            aplus_override=aplus,
            across_override=across,
        )
        for key, (m, mf) in raw.items():
            amp[key], offset[key] = extract_ap(m, mf, phi_r)
    return TDIGrid(times, detector_time, phi_r, phi_b, amp, offset)


def nonuniform_phase_derivative(index: int, times: np.ndarray, values: np.ndarray) -> float:
    n = times.size
    if n < 2:
        return 0.0
    if index <= 0:
        dt0 = times[1] - times[0]
        return 0.0 if dt0 == 0.0 else float((values[1] - values[0]) / dt0)
    if index >= n - 1:
        dt0 = times[n - 1] - times[n - 2]
        return 0.0 if dt0 == 0.0 else float((values[n - 1] - values[n - 2]) / dt0)

    x0 = float(times[index - 1])
    x1 = float(times[index])
    x2 = float(times[index + 1])
    y0 = float(values[index - 1])
    y1 = float(values[index])
    y2 = float(values[index + 1])
    d0 = (x0 - x1) * (x0 - x2)
    d1 = (x1 - x0) * (x1 - x2)
    d2 = (x2 - x0) * (x2 - x1)
    if d0 == 0.0 or d1 == 0.0 or d2 == 0.0:
        return 0.0 if x2 == x0 else float((y2 - y0) / (x2 - x0))

    c0 = (x1 - x2) / d0
    c1 = (2.0 * x1 - x0 - x2) / d1
    c2 = (x1 - x0) / d2
    return c0 * y0 + c1 * y1 + c2 * y2


def nonuniform_derivative_array(times: np.ndarray, values: np.ndarray) -> np.ndarray:
    if NUMBA_AVAILABLE:
        return _nonuniform_derivative_array_jit(
            np.ascontiguousarray(times, dtype=np.float64),
            np.ascontiguousarray(values, dtype=np.float64),
        )
    out = np.empty_like(values, dtype=np.float64)
    for i in range(values.size):
        out[i] = nonuniform_phase_derivative(i, times, values)
    return out


@njit(cache=True)
def _nonuniform_derivative_array_jit(
    times: np.ndarray, values: np.ndarray
) -> np.ndarray:
    """Compiled nonuniform three-point derivative with matching arithmetic."""

    n = values.size
    out = np.empty(n, dtype=np.float64)
    if n < 2:
        if n == 1:
            out[0] = 0.0
        return out
    dt0 = times[1] - times[0]
    out[0] = 0.0 if dt0 == 0.0 else (values[1] - values[0]) / dt0
    for index in range(1, n - 1):
        x0 = times[index - 1]
        x1 = times[index]
        x2 = times[index + 1]
        y0 = values[index - 1]
        y1 = values[index]
        y2 = values[index + 1]
        d0 = (x0 - x1) * (x0 - x2)
        d1 = (x1 - x0) * (x1 - x2)
        d2 = (x2 - x0) * (x2 - x1)
        if d0 == 0.0 or d1 == 0.0 or d2 == 0.0:
            out[index] = 0.0 if x2 == x0 else (y2 - y0) / (x2 - x0)
        else:
            c0 = (x1 - x2) / d0
            c1 = (2.0 * x1 - x0 - x2) / d1
            c2 = (x1 - x0) / d2
            out[index] = c0 * y0 + c1 * y1 + c2 * y2
    dt0 = times[n - 1] - times[n - 2]
    out[n - 1] = (
        0.0 if dt0 == 0.0 else (values[n - 1] - values[n - 2]) / dt0
    )
    return out


def build_nonuniform_phase_derivatives(times: np.ndarray, phase: np.ndarray, phase_spline) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phase_grid = np.asarray(phase, dtype=np.float64).copy()
    freq_grid = nonuniform_derivative_array(times, phase_grid) / (2.0 * PI)
    bad = (~np.isfinite(freq_grid)) | (freq_grid <= 0.0)
    if np.any(bad):
        # Last-ditch compatibility fallback.  The production path should be
        # driven by the local nonuniform derivatives above; spline derivatives
        # are used only for invalid edge cases.
        freq_grid[bad] = np.asarray(phase_spline.derivative(1)(times[bad]), dtype=np.float64) / (2.0 * PI)

    fdot_grid = nonuniform_derivative_array(times, freq_grid)
    bad = (~np.isfinite(fdot_grid)) | (fdot_grid == 0.0)
    if np.any(bad):
        fdot_grid[bad] = np.asarray(phase_spline.derivative(2)(times[bad]), dtype=np.float64) / (2.0 * PI)
    return phase_grid, freq_grid, fdot_grid


def tdi_frequency_track(times: np.ndarray, phase: np.ndarray, fallback_freq: np.ndarray | None = None) -> np.ndarray:
    freq = np.abs(nonuniform_derivative_array(times, phase)) / (2.0 * PI)
    bad = (~np.isfinite(freq)) | (freq <= 0.0)
    if np.any(bad):
        if fallback_freq is not None and fallback_freq.size >= freq.size:
            freq[bad] = np.abs(fallback_freq[: freq.size][bad])
        good = np.isfinite(freq) & (freq > 0.0)
        replacement = np.nanmedian(freq[good]) if np.any(good) else 0.0
        freq[(~np.isfinite(freq)) | (freq <= 0.0)] = replacement
    return freq


def ftran(setup: np.ndarray, times: np.ndarray, amp: np.ndarray, phase: np.ndarray, shape: WDMShape) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    amp_spline = make_ap_spline(times, amp)
    phase_spline = make_ap_spline(times, phase)
    phase_grid, freq_grid, fdot_grid = build_nonuniform_phase_derivatives(times, phase, phase_spline)

    dte = float(setup[0])
    nend = int(setup[1])
    tspan = nend * dte
    tes = float(setup[2])
    rise = float(setup[3])
    jj = min(max(int(setup[4]), 1), times.size - 2)
    imx = min(int(setup[6]), nend // 2)
    fjoin = float(freq_grid[jj])
    if not math.isfinite(fjoin) or fjoin <= 0.0:
        fjoin = abs(float(phase_spline.derivative(1)(times[jj]))) / (2.0 * PI)
    ii = int(math.ceil(fjoin * tspan))
    ii = min(max(ii, 1), max(imx - 1, 1))
    nx = jj + (imx - ii) + 1

    freq = np.zeros(nx, dtype=np.float64)
    out_phase = np.zeros(nx, dtype=np.float64)
    out_amp = np.zeros(nx, dtype=np.float64)

    f0 = float(freq_grid[0])
    fdot_val = abs(float(fdot_grid[0]))
    if not math.isfinite(fdot_val) or fdot_val <= 0.0:
        fdot_val = abs(float(phase_spline.derivative(2)(times[1]))) / (2.0 * PI)
    if not math.isfinite(fdot_val) or fdot_val <= 0.0:
        fdot_val = np.finfo(np.float64).tiny
    out_phase[0] = phase_grid[0] - 2.0 * PI * f0 * (times[0] - shape.Tobs) + PI / 4.0
    out_amp[0] = math.sqrt(1.0 / fdot_val) * float(amp_spline(times[0]))
    freq[0] = f0
    for i in range(1, jj + 1):
        t = times[i]
        f = float(freq_grid[i])
        if not math.isfinite(f) or f <= 0.0:
            f = abs(float(phase_spline.derivative(1)(t))) / (2.0 * PI)
        fdot_val = abs(float(fdot_grid[i]))
        if not math.isfinite(fdot_val) or fdot_val <= 0.0:
            fdot_val = abs(float(phase_spline.derivative(2)(t))) / (2.0 * PI)
        if not math.isfinite(fdot_val) or fdot_val <= 0.0:
            fdot_val = np.finfo(np.float64).tiny
        freq[i] = f
        out_phase[i] = phase_grid[i] - 2.0 * PI * f * (t - shape.Tobs) + PI / 4.0
        out_amp[i] = math.sqrt(1.0 / fdot_val) * float(amp_spline(t))

    tt = tes + dte * np.arange(nend, dtype=np.float64)
    hend = np.zeros(nend, dtype=np.float64)
    mask = tt < times[-1]
    local_amp = amp_spline(tt[mask])
    roll = tt[mask] - tes
    taper_mask = roll < rise
    local_amp = np.asarray(local_amp, dtype=np.float64)
    local_amp[taper_mask] *= 0.5 * (1.0 - np.cos(PI * roll[taper_mask] / rise))
    hend[mask] = local_amp * np.cos(phase_spline(tt[mask]))
    short_htime = hend.copy()

    hfft = np.fft.rfft(hend) * (2.0 * dte)

    f = float(ii) / tspan
    f1 = freq[jj - 1]
    f2 = freq[jj]
    p1 = out_phase[jj - 1]
    p2 = out_phase[jj]
    if f2 != f1:
        pjoin = p1 + (p2 - p1) / (f2 - f1) * (f - f1)
    else:
        pjoin = p2
    fft_reference_time = tes + 0.5 * tspan
    fft_reference_offset = fft_reference_time - tes
    fft_phase_shift = shape.Tobs - fft_reference_time
    raw_phase = math.atan2(hfft[ii].imag, hfft[ii].real)
    theta = 2.0 * PI * f * fft_reference_offset
    pold = math.atan2(math.sin(raw_phase + theta), math.cos(raw_phase + theta))
    fft_join_phase = pold + 2.0 * PI * f * fft_phase_shift
    pw = 2.0 * PI * round((pjoin - fft_join_phase) / (2.0 * PI))

    k = jj
    for i in range(ii, imx):
        f = float(i) / tspan
        raw_phase = math.atan2(hfft[i].imag, hfft[i].real)
        theta = 2.0 * PI * f * fft_reference_offset
        pf = math.atan2(math.sin(raw_phase + theta), math.cos(raw_phase + theta))
        delta = pf - pold
        if delta > PI:
            pw -= 2.0 * PI
        if delta < -PI:
            pw += 2.0 * PI
        pold = pf
        k += 1
        out_phase[k] = (pf + pw) + 2.0 * PI * f * fft_phase_shift
        out_amp[k] = abs(hfft[i])
        freq[k] = f

    flip = np.ones(nx, dtype=np.float64)
    pjump = np.zeros(nx, dtype=np.float64)
    istart = max(jj + 1, 1)
    i = 1
    while i < nx - 1:
        flip[i] = flip[i - 1]
        pjump[i] = pjump[i - 1]
        if i >= istart and out_amp[i] < out_amp[i - 1] and out_amp[i] < out_amp[i + 1]:
            d1 = out_amp[i + 1] + out_amp[i - 1] - 2.0 * out_amp[i]
            d2 = -out_amp[i + 1] + out_amp[i - 1] - 2.0 * out_amp[i]
            d3 = -out_amp[i + 1] + out_amp[i - 1] + 2.0 * out_amp[i]
            if abs(d1) > 0.0 and abs(d2 / d1) < 0.1:
                flip[i + 1] = -flip[i]
                pjump[i + 1] = pjump[i] + PI
                i += 1
            elif abs(d1) > 0.0 and abs(d3 / d1) < 0.1:
                flip[i] = -flip[i - 1]
                pjump[i] = pjump[i - 1] + PI
        i += 1
    flip[-1] = flip[-2]
    pjump[-1] = pjump[-2]
    out_amp[istart:] *= flip[istart:]
    out_phase[istart:] += pjump[istart:]
    return freq, out_phase, out_amp, short_htime


def ftran_spa_only(
    setup: np.ndarray,
    times: np.ndarray,
    amp: np.ndarray,
    phase: np.ndarray,
    shape: WDMShape,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stationary-phase samples up to the configured handoff point.

    The production C path no longer appends the short endpoint FFT to these
    arrays and splines through the join.  The endpoint WDM coefficients are
    replaced directly from the short time-domain FFT instead, so this helper
    intentionally returns only the SPA side of the Fourier-domain input.
    """

    phase_spline = make_ap_spline(times, phase)
    phase_grid, freq_grid, fdot_grid = build_nonuniform_phase_derivatives(times, phase, phase_spline)

    jj = min(max(int(setup[4]), 1), times.size - 1)
    count = jj + 1
    local_time = np.asarray(times[:count], dtype=np.float64)
    freq = np.asarray(freq_grid[:count], dtype=np.float64).copy()
    bad = (~np.isfinite(freq)) | (freq <= 0.0)
    if np.any(bad):
        derivative = phase_spline.derivative(1)
        freq[bad] = np.abs(derivative(local_time[bad])) / (2.0 * PI)

    fdot = np.abs(np.asarray(fdot_grid[:count], dtype=np.float64))
    bad = (~np.isfinite(fdot)) | (fdot <= 0.0)
    if np.any(bad):
        derivative = phase_spline.derivative(2)
        fdot[bad] = np.abs(derivative(local_time[bad])) / (2.0 * PI)
    fdot[(~np.isfinite(fdot)) | (fdot <= 0.0)] = np.finfo(np.float64).tiny

    # The SPA samples are located at spline knots, so evaluating the splines
    # one scalar at a time only reproduces the input arrays.  Keeping this
    # operation array-valued removes tens of thousands of Python/SciPy calls
    # in the full higher-mode path without changing the construction.
    out_phase = (
        np.asarray(phase_grid[:count], dtype=np.float64)
        - 2.0 * PI * freq * (local_time - shape.Tobs)
        + PI / 4.0
    )
    out_amp = np.asarray(amp[:count], dtype=np.float64) / np.sqrt(fdot)

    return freq, out_phase, out_amp


def build_short_endpoint_waveform(
    setup: np.ndarray,
    times: np.ndarray,
    amp: np.ndarray,
    phase: np.ndarray,
) -> np.ndarray:
    """Build the tapered endpoint time segment used for direct WDM replacement."""

    amp_spline = make_ap_spline(times, amp)
    phase_spline = make_ap_spline(times, phase)
    dte = float(setup[0])
    nend = int(setup[1])
    tes = float(setup[2])
    rise = float(setup[3])

    tt = tes + dte * np.arange(nend, dtype=np.float64)
    htime = np.zeros(nend, dtype=np.float64)
    mask = (tt >= float(times[0])) & (tt <= float(times[-1]))
    if np.any(mask):
        local_amp = np.asarray(amp_spline(tt[mask]), dtype=np.float64)
        roll = tt[mask] - tes
        taper_mask = roll < rise
        if np.any(taper_mask):
            local_amp[taper_mask] *= 0.5 * (1.0 - np.cos(PI * roll[taper_mask] / rise))
        htime[mask] = local_amp * np.cos(phase_spline(tt[mask]))
    return htime


def build_direct_endpoint_waveforms(
    setup: np.ndarray,
    source: SourceParams,
    intrinsic: IntrinsicGrid,
    l_splines: list[CubicSpline],
    p_splines: list[CubicSpline],
    v_splines: list[CubicSpline],
) -> dict[str, np.ndarray]:
    """Evaluate exact TDI delays at every endpoint FFT output time.

    The early SPA still uses the sparse post-TDI amplitude/phase grid.  Near
    merger, however, interpolating that representation can introduce smooth,
    parameter-dependent errors because all delayed copies are changing on a
    few-second scale.  This short path evaluates the intrinsic AP splines at
    the actual fractional TDI delays and applies the endpoint taper only after
    the complete X/Y/Z responses have been formed.
    """

    dte = float(setup[0])
    nend = int(setup[1])
    start = float(setup[2])
    rise = float(setup[3])
    endpoint_time = start + dte * np.arange(nend, dtype=np.float64)
    amp_spline = make_ap_spline(intrinsic.bary_time, intrinsic.amplitude)
    phase_spline = make_ap_spline(intrinsic.bary_time, intrinsic.phase)

    swsh22norm = math.sqrt(5.0 / (64.0 * PI))
    cosi = source.cos_inclination
    aplus = -2.0 * swsh22norm * (1.0 + cosi * cosi)
    across = -4.0 * swsh22norm * cosi
    if NUMBA_AVAILABLE:
        u, v, kv = sky_vectors(source)
        eplus = np.ascontiguousarray(np.outer(v, v) - np.outer(u, u), dtype=np.float64)
        ecross = np.ascontiguousarray(np.outer(u, v) + np.outer(v, u), dtype=np.float64)
        raw_arrays = _fast_response_jit(
            np.ascontiguousarray(endpoint_time),
            np.ascontiguousarray(l_splines[0].x, dtype=np.float64),
            _stack_spline_coefficients(l_splines),
            _stack_spline_coefficients(p_splines),
            _stack_spline_coefficients(v_splines),
            np.ascontiguousarray(amp_spline.x, dtype=np.float64),
            np.ascontiguousarray(amp_spline.c, dtype=np.float64),
            np.ascontiguousarray(phase_spline.c, dtype=np.float64),
            np.ascontiguousarray(kv, dtype=np.float64),
            eplus,
            ecross,
            aplus,
            across,
            math.cos(2.0 * source.polarization),
            math.sin(2.0 * source.polarization),
        )
        response = {
            "X": np.asarray(raw_arrays[0]),
            "Y": np.asarray(raw_arrays[2]),
            "Z": np.asarray(raw_arrays[4]),
        }
    else:
        raw = fast_response(
            endpoint_time,
            source,
            l_splines,
            p_splines,
            v_splines,
            amp_spline,
            phase_spline,
            aplus_override=aplus,
            across_override=across,
        )
        response = {name: np.asarray(values[0]) for name, values in raw.items()}

    roll = endpoint_time - start
    taper = np.ones(nend, dtype=np.float64)
    taper_region = roll < rise
    taper[taper_region] = 0.5 * (1.0 - np.cos(PI * roll[taper_region] / rise))
    return {name: values * taper for name, values in response.items()}


def wdm_pixels(times: np.ndarray, freq: np.ndarray, shape: WDMShape) -> tuple[np.ndarray, np.ndarray]:
    nmid = np.full(shape.nf, -1, dtype=np.int64)
    nsize = np.zeros(shape.nf, dtype=np.int64)
    jmin_layer = np.full(shape.nf, shape.nt + 1, dtype=np.int64)
    jmax_layer = np.full(shape.nf, -1, dtype=np.int64)
    time_pad = max(int(WDM_TIMESCAN_PAD_PIXELS), 0)

    for i in range(times.size - 1):
        if not (
            math.isfinite(float(times[i]))
            and math.isfinite(float(times[i + 1]))
            and math.isfinite(float(freq[i]))
            and math.isfinite(float(freq[i + 1]))
        ):
            continue
        tlo = min(float(times[i]), float(times[i + 1]))
        thi = max(float(times[i]), float(times[i + 1]))
        if thi < 0.0 or tlo > shape.Tobs:
            continue
        tlo = max(tlo, 0.0)
        thi = min(thi, shape.Tobs)

        flo = abs(float(freq[i]))
        fhi = abs(float(freq[i + 1]))
        if flo > fhi:
            flo, fhi = fhi, flo
        flo -= shape.FB
        fhi += shape.FB
        if fhi <= 0.0:
            continue

        mlo = int(math.ceil(flo / shape.DF))
        mhi = int(math.floor(fhi / shape.DF))
        mlo = max(mlo, 1)
        mhi = min(mhi, shape.nf - 1)
        if mhi < mlo:
            continue

        jlo = int(math.floor(tlo / shape.DT)) - time_pad
        jhi = int(math.ceil(thi / shape.DT)) + time_pad
        jlo = max(jlo, 0)
        jhi = min(jhi, shape.nt - 1)
        if jhi < jlo:
            continue

        jmin_layer[mlo : mhi + 1] = np.minimum(jmin_layer[mlo : mhi + 1], jlo)
        jmax_layer[mlo : mhi + 1] = np.maximum(jmax_layer[mlo : mhi + 1], jhi)

    for m in range(1, shape.nf):
        if jmax_layer[m] < jmin_layer[m]:
            continue
        jlo = int(jmin_layer[m])
        jhi = int(jmax_layer[m])
        needed = max(jhi - jlo + 1, 1)

        block = 1
        while block < needed + 2:
            block *= 2
        block = max(block, 2 * shape.mult)
        block = min(block, shape.nt)

        center = (jlo + jhi + 1) // 2
        if center % 2 != 0:
            center -= 1
        if center < block // 2:
            center = block // 2
        if center + block // 2 > shape.nt:
            center = shape.nt - block // 2
        if center % 2 != 0:
            center -= 1
        while center - block // 2 > jlo:
            center -= 2
        while center + block // 2 - 1 < jhi:
            center += 2
        if center < block // 2:
            center = block // 2
        if center + block // 2 > shape.nt:
            center = shape.nt - block // 2
        if center % 2 != 0:
            center -= 1

        nmid[m] = center
        nsize[m] = block
    return nmid, nsize


def wdm_pixels_range(
    times: np.ndarray,
    freq: np.ndarray,
    tmin: float,
    tmax: float,
    shape: WDMShape,
) -> tuple[np.ndarray, np.ndarray]:
    """Time-scan WDM pixel finder clipped to a finite endpoint interval."""

    if NUMBA_AVAILABLE:
        return _wdm_pixels_range_jit(
            np.ascontiguousarray(times, dtype=np.float64),
            np.ascontiguousarray(freq, dtype=np.float64),
            float(tmin),
            float(tmax),
            int(shape.nt),
            int(shape.nf),
            float(shape.DT),
            float(shape.DF),
            float(shape.FB),
            int(shape.mult),
            max(int(WDM_TIMESCAN_PAD_PIXELS), 0),
        )

    nmid = np.full(shape.nf, -1, dtype=np.int64)
    nsize = np.zeros(shape.nf, dtype=np.int64)
    jmin_layer = np.full(shape.nf, shape.nt + 1, dtype=np.int64)
    jmax_layer = np.full(shape.nf, -1, dtype=np.int64)
    time_pad = max(int(WDM_TIMESCAN_PAD_PIXELS), 0)

    if not (math.isfinite(tmin) and math.isfinite(tmax)) or tmax <= tmin:
        return nmid, nsize
    tmin = max(float(tmin), 0.0)
    tmax = min(float(tmax), shape.Tobs)
    if tmax <= tmin:
        return nmid, nsize

    for i in range(times.size - 1):
        if not (
            math.isfinite(float(times[i]))
            and math.isfinite(float(times[i + 1]))
            and math.isfinite(float(freq[i]))
            and math.isfinite(float(freq[i + 1]))
        ):
            continue
        ta = float(times[i])
        tb = float(times[i + 1])
        fa = float(freq[i])
        fb = float(freq[i + 1])
        if tb == ta:
            continue
        if tb < ta:
            ta, tb = tb, ta
            fa, fb = fb, fa
        if tb < tmin or ta > tmax:
            continue

        seglo = max(ta, tmin)
        seghi = min(tb, tmax)
        if seghi < seglo:
            continue
        ua = (seglo - ta) / (tb - ta)
        ub = (seghi - ta) / (tb - ta)
        flo = abs(fa + ua * (fb - fa))
        fhi = abs(fa + ub * (fb - fa))
        if flo > fhi:
            flo, fhi = fhi, flo
        flo -= shape.FB
        fhi += shape.FB
        if fhi <= 0.0:
            continue

        mlo = int(math.ceil(flo / shape.DF))
        mhi = int(math.floor(fhi / shape.DF))
        mlo = max(mlo, 1)
        mhi = min(mhi, shape.nf - 1)
        if mhi < mlo:
            continue

        jlo = int(math.floor(seglo / shape.DT)) - time_pad
        jhi = int(math.ceil(seghi / shape.DT)) + time_pad
        jlo = max(jlo, 0)
        jhi = min(jhi, shape.nt - 1)
        if jhi < jlo:
            continue

        jmin_layer[mlo : mhi + 1] = np.minimum(jmin_layer[mlo : mhi + 1], jlo)
        jmax_layer[mlo : mhi + 1] = np.maximum(jmax_layer[mlo : mhi + 1], jhi)

    for m in range(1, shape.nf):
        if jmax_layer[m] < jmin_layer[m]:
            continue
        jlo = int(jmin_layer[m])
        jhi = int(jmax_layer[m])
        needed = max(jhi - jlo + 1, 1)

        block = 1
        while block < needed + 2:
            block *= 2
        block = max(block, 2 * shape.mult)
        block = min(block, shape.nt)

        center = (jlo + jhi + 1) // 2
        if center % 2 != 0:
            center -= 1
        if center < block // 2:
            center = block // 2
        if center + block // 2 > shape.nt:
            center = shape.nt - block // 2
        if center % 2 != 0:
            center -= 1
        while center - block // 2 > jlo:
            center -= 2
        while center + block // 2 - 1 < jhi:
            center += 2
        if center < block // 2:
            center = block // 2
        if center + block // 2 > shape.nt:
            center = shape.nt - block // 2
        if center % 2 != 0:
            center -= 1

        nmid[m] = center
        nsize[m] = block
    return nmid, nsize


@njit(cache=True)
def _wdm_pixels_range_jit(
    times: np.ndarray,
    freq: np.ndarray,
    tmin: float,
    tmax: float,
    nt: int,
    nf: int,
    dt_pixel: float,
    df_pixel: float,
    frequency_pad: float,
    mult: int,
    time_pad: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compiled implementation of :func:`wdm_pixels_range`."""

    nmid = np.full(nf, -1, dtype=np.int64)
    nsize = np.zeros(nf, dtype=np.int64)
    jmin_layer = np.full(nf, nt + 1, dtype=np.int64)
    jmax_layer = np.full(nf, -1, dtype=np.int64)
    if not (math.isfinite(tmin) and math.isfinite(tmax)) or tmax <= tmin:
        return nmid, nsize
    tmin = max(tmin, 0.0)
    tmax = min(tmax, dt_pixel * nt)
    if tmax <= tmin:
        return nmid, nsize

    for i in range(times.size - 1):
        ta = times[i]
        tb = times[i + 1]
        fa = freq[i]
        fb = freq[i + 1]
        if not (
            math.isfinite(ta)
            and math.isfinite(tb)
            and math.isfinite(fa)
            and math.isfinite(fb)
        ):
            continue
        if tb == ta:
            continue
        if tb < ta:
            hold = ta
            ta = tb
            tb = hold
            hold = fa
            fa = fb
            fb = hold
        if tb < tmin or ta > tmax:
            continue

        seglo = max(ta, tmin)
        seghi = min(tb, tmax)
        if seghi < seglo:
            continue
        ua = (seglo - ta) / (tb - ta)
        ub = (seghi - ta) / (tb - ta)
        flo = abs(fa + ua * (fb - fa))
        fhi = abs(fa + ub * (fb - fa))
        if flo > fhi:
            hold = flo
            flo = fhi
            fhi = hold
        flo -= frequency_pad
        fhi += frequency_pad
        if fhi <= 0.0:
            continue

        mlo = max(int(math.ceil(flo / df_pixel)), 1)
        mhi = min(int(math.floor(fhi / df_pixel)), nf - 1)
        if mhi < mlo:
            continue
        jlo = max(int(math.floor(seglo / dt_pixel)) - time_pad, 0)
        jhi = min(int(math.ceil(seghi / dt_pixel)) + time_pad, nt - 1)
        if jhi < jlo:
            continue
        for m in range(mlo, mhi + 1):
            if jlo < jmin_layer[m]:
                jmin_layer[m] = jlo
            if jhi > jmax_layer[m]:
                jmax_layer[m] = jhi

    for m in range(1, nf):
        if jmax_layer[m] < jmin_layer[m]:
            continue
        jlo = int(jmin_layer[m])
        jhi = int(jmax_layer[m])
        needed = max(jhi - jlo + 1, 1)
        block = 1
        while block < needed + 2:
            block *= 2
        block = max(block, 2 * mult)
        block = min(block, nt)

        center = (jlo + jhi + 1) // 2
        if center % 2 != 0:
            center -= 1
        if center < block // 2:
            center = block // 2
        if center + block // 2 > nt:
            center = nt - block // 2
        if center % 2 != 0:
            center -= 1
        while center - block // 2 > jlo:
            center -= 2
        while center + block // 2 - 1 < jhi:
            center += 2
        if center < block // 2:
            center = block // 2
        if center + block // 2 > nt:
            center = nt - block // 2
        if center % 2 != 0:
            center -= 1
        nmid[m] = center
        nsize[m] = block
    return nmid, nsize


def wdm_pixels_add_merger_frequency_tail(nmid: np.ndarray, nsize: np.ndarray, fmax_spectrum: float, tail_time: float, shape: WDMShape) -> None:
    if not math.isfinite(fmax_spectrum) or fmax_spectrum <= 0.0 or not math.isfinite(tail_time):
        return

    active = np.flatnonzero((nmid >= 0) & (nsize > 0))
    active_max = int(active[-1]) if active.size else 0
    mmax_tail = int(math.floor((fmax_spectrum + shape.FB) / shape.DF))
    mmax_tail = min(mmax_tail, shape.nf - 1)
    if mmax_tail <= active_max:
        return

    block = 1
    while block < 2 * shape.mult:
        block *= 2
    block = min(block, shape.nt)

    center = 2 * int(math.floor(tail_time / (2.0 * shape.DT) + 0.5))
    if center < block // 2:
        center = block // 2
    if center + block // 2 > shape.nt:
        center = shape.nt - block // 2
    if center % 2 != 0:
        center -= 1
    if center < block // 2:
        center = block // 2

    for m in range(active_max + 1, mmax_tail + 1):
        if nmid[m] < 0 or nsize[m] <= 0:
            nmid[m] = center
            nsize[m] = block


def merge_pixel_plans(
    nmid_total: np.ndarray,
    nsize_total: np.ndarray,
    nmid_add: np.ndarray,
    nsize_add: np.ndarray,
    shape: WDMShape,
) -> None:
    """In-place union of two one-block-per-layer WDM packet plans."""

    if NUMBA_AVAILABLE:
        _merge_pixel_plans_jit(
            nmid_total,
            nsize_total,
            nmid_add,
            nsize_add,
            int(shape.nt),
            int(shape.nf),
            int(shape.mult),
        )
        return

    for m in range(1, shape.nf):
        if nmid_add[m] < 0 or nsize_add[m] <= 0:
            continue
        if nmid_total[m] < 0 or nsize_total[m] <= 0:
            nmid_total[m] = nmid_add[m]
            nsize_total[m] = nsize_add[m]
            continue

        lo_total = int(nmid_total[m] - nsize_total[m] // 2)
        hi_total = int(nmid_total[m] + nsize_total[m] // 2 - 1)
        lo_add = int(nmid_add[m] - nsize_add[m] // 2)
        hi_add = int(nmid_add[m] + nsize_add[m] // 2 - 1)
        lo = max(min(lo_total, lo_add), 0)
        hi = min(max(hi_total, hi_add), shape.nt - 1)
        needed = max(hi - lo + 1, 1)

        block = 1
        while block < needed:
            block *= 2
        block = max(block, 2 * shape.mult)
        block = min(block, shape.nt)

        center = (lo + hi + 1) // 2
        if center % 2 != 0:
            center -= 1
        if center < block // 2:
            center = block // 2
        if center + block // 2 > shape.nt:
            center = shape.nt - block // 2
        if center % 2 != 0:
            center -= 1
        while center - block // 2 > lo:
            center -= 2
        while center + block // 2 - 1 < hi:
            center += 2
        if center < block // 2:
            center = block // 2
        if center + block // 2 > shape.nt:
            center = shape.nt - block // 2
        if center % 2 != 0:
            center -= 1

        nmid_total[m] = center
        nsize_total[m] = block


@njit(cache=True)
def _merge_pixel_plans_jit(
    nmid_total: np.ndarray,
    nsize_total: np.ndarray,
    nmid_add: np.ndarray,
    nsize_add: np.ndarray,
    nt: int,
    nf: int,
    mult: int,
) -> None:
    """Compiled in-place pixel-plan union."""

    for m in range(1, nf):
        if nmid_add[m] < 0 or nsize_add[m] <= 0:
            continue
        if nmid_total[m] < 0 or nsize_total[m] <= 0:
            nmid_total[m] = nmid_add[m]
            nsize_total[m] = nsize_add[m]
            continue

        lo_total = int(nmid_total[m] - nsize_total[m] // 2)
        hi_total = int(nmid_total[m] + nsize_total[m] // 2 - 1)
        lo_add = int(nmid_add[m] - nsize_add[m] // 2)
        hi_add = int(nmid_add[m] + nsize_add[m] // 2 - 1)
        lo = max(min(lo_total, lo_add), 0)
        hi = min(max(hi_total, hi_add), nt - 1)
        needed = max(hi - lo + 1, 1)

        block = 1
        while block < needed:
            block *= 2
        block = max(block, 2 * mult)
        block = min(block, nt)

        center = (lo + hi + 1) // 2
        if center % 2 != 0:
            center -= 1
        if center < block // 2:
            center = block // 2
        if center + block // 2 > nt:
            center = nt - block // 2
        if center % 2 != 0:
            center -= 1
        while center - block // 2 > lo:
            center -= 2
        while center + block // 2 - 1 < hi:
            center += 2
        if center < block // 2:
            center = block // 2
        if center + block // 2 > nt:
            center = nt - block // 2
        if center % 2 != 0:
            center -= 1

        nmid_total[m] = center
        nsize_total[m] = block


def phitilde(omega: np.ndarray, shape: WDMShape) -> np.ndarray:
    x = np.abs(omega)
    out = np.zeros_like(x, dtype=np.float64)
    low = x < shape.A
    roll = (x >= shape.A) & (x < shape.A + shape.B)
    out[low] = shape.insDOM
    if np.any(roll):
        y = betainc(shape.nx, shape.nx, (x[roll] - shape.A) / shape.B)
        out[roll] = shape.insDOM * np.cos(y * PI / 2.0)
    return out


def tukey_inplace(data: np.ndarray, alpha: float) -> None:
    n = data.size
    if n <= 1 or alpha <= 0.0:
        return
    imin = int(alpha * float(n - 1) / 2.0)
    imax = int(float(n - 1) * (1.0 - alpha / 2.0))
    nwin = n - imax
    if imin > 0:
        i = np.arange(imin, dtype=np.float64)
        data[:imin] *= 0.5 * (1.0 + np.cos(PI * (i / float(imin) - 1.0)))
    if nwin > 0 and imax + 1 < n:
        i = np.arange(imax + 1, n, dtype=np.float64)
        data[imax + 1 :] *= 0.5 * (1.0 + np.cos(PI * ((i - float(imax)) / float(nwin))))


def wdm_band_for_frequency_range(fstart: float, fend: float, shape: WDMShape) -> tuple[int, int]:
    """Choose the heterodyne layer and local bandwidth for a finite band."""

    fs = fstart
    fe = fend
    if fs < fe:
        fmin = fs * (1.0 - 1.0e-4)
        fmax = fe * (1.0 + 1.0e-4)
    else:
        fmin = fe * (1.0 - 1.0e-4)
        fmax = fs * (1.0 + 1.0e-4)

    kk = int(round(fmin / shape.DF))
    k = kk
    dfmin = fmin - float(k) * shape.DF
    while dfmin < shape.DF / 2.0 + shape.FB:
        k -= 1
        dfmin = fmin - float(k) * shape.DF
    if k % 2 != 0:
        k -= 1
    kmin = max(k, 0)
    if kmin % 2 != 0:
        kmin = max(kmin - 1, 0)

    kmax = kk
    dfmax = fmax - float(kmax + 1) * shape.DF
    while dfmax + shape.FB > 0.0:
        kmax += 1
        dfmax = fmax - float(kmax + 1) * shape.DF
    kmax = min(kmax, shape.nf - 1)
    kwidth = max(kmax - kmin, 1)
    return kmin, kwidth


def ucb_wdm_band(source: UCBSourceParams, shape: WDMShape) -> tuple[int, int]:
    """Choose the heterodyne layer and local bandwidth as in response.c."""

    fdot, fddot = source.frequency_derivatives()
    fs = source.frequency_hz
    fe = fs + fdot * shape.Tobs + 0.5 * fddot * shape.Tobs * shape.Tobs
    return wdm_band_for_frequency_range(fs, fe, shape)


def phihetf_ucb(nfx: int, shape: WDMShape) -> np.ndarray:
    dtx = shape.DT / float(nfx)
    dom = 2.0 * PI / shape.Tobs
    omega = np.arange(shape.nt // 2 + 1, dtype=np.float64) * dom
    phihf = phitilde(omega, shape)
    nrm = phihf[0] * phihf[0] + 2.0 * np.sum(phihf[1:] * phihf[1:])
    nrm = math.sqrt(float(nrm) / dtx)
    if nrm > 0.0:
        phihf /= nrm
    return phihf


def wdmtran_heterodyne_frequency(kx: int, kw: int, data: np.ndarray, shape: WDMShape) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Frequency-domain heterodyned WDM transform used by response.c."""

    nfx = kw + 1
    nx = nfx * shape.nt
    if data.size != nx:
        raise ValueError(f"heterodyned data length {data.size} does not match Nt*(kw+1)={nx}")

    work = np.asarray(data, dtype=np.float64).copy()
    tukey_inplace(work, 8.0 / float(shape.nt))
    hfft = np.fft.rfft(work)
    phihf = phihetf_ucb(nfx, shape)
    # ``wdmtranF`` starts from the one-sided FFT of a real heterodyned
    # signal.  The historical UCB implementation omitted the conversion to
    # the frequency-domain Meyer normalization now used by wd_viafreq and the
    # PhenomT/FEW sparse paths.  The missing amplitude factor is sqrt(8/15).
    fac = math.sqrt(8.0 / 15.0) / math.sqrt(float(nx) / 2.0)

    half_nt = shape.nt // 2
    j = np.arange(-half_nt, half_nt, dtype=np.int64)
    iout = np.arange(shape.nt, dtype=np.int64)
    listn: list[np.ndarray] = []
    listm: list[np.ndarray] = []
    values: list[np.ndarray] = []

    for local_m in range(1, nfx):
        global_m = kx + local_m
        if global_m <= 0 or global_m >= shape.nf:
            continue
        bins = j + local_m * half_nt
        valid = (bins > 0) & (bins < nx // 2)
        dx = np.zeros(shape.nt, dtype=np.complex128)
        if np.any(valid):
            dx[valid] = hfft[bins[valid]] * phihf[np.abs(j[valid])]
        packet = np.fft.ifft(dx) * shape.nt

        even = ((iout + global_m) % 2) == 0
        if global_m % 2 == 0:
            wdmout = fac * np.where(even, packet.real, packet.imag)
        else:
            wdmout = fac * np.where(even, packet.real, -packet.imag)
        listn.append(iout.copy())
        listm.append(np.full(shape.nt, global_m, dtype=np.int64))
        values.append(np.asarray(wdmout, dtype=np.float64))

    if not listn:
        empty_i = np.empty(0, dtype=np.int64)
        empty = np.empty(0, dtype=np.float64)
        return empty_i, empty_i, empty
    return np.concatenate(listn), np.concatenate(listm), np.concatenate(values)


def ucb_heterodyned_time_series(
    source: UCBSourceParams,
    intrinsic: IntrinsicGrid,
    tdi: TDIGrid,
    p_splines: list[CubicSpline],
    channel: str,
    kx: int,
    kw: int,
    shape: WDMShape,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    nfx = kw + 1
    nx = shape.nt * nfx
    dtx = shape.DT / float(nfx)
    tf = dtx * np.arange(nx, dtype=np.float64)
    fx = float(kx) * shape.DF

    reference_time = spacecraft0_reference_time(tf, source, p_splines)
    phase_spline = CubicSpline(intrinsic.bary_time, intrinsic.phase, bc_type="natural")
    pref = phase_spline(reference_time)

    amp_spline = CubicSpline(tdi.time, tdi.amplitude[channel], bc_type="natural")
    offset_spline = CubicSpline(tdi.time, tdi.phase_offset[channel], bc_type="natural")
    hphase = 2.0 * PI * fx * tf
    wave = amp_spline(tf) * np.cos(offset_spline(tf) + pref - hphase)
    return tf, np.asarray(wave, dtype=np.float64), pref, fx


def ucb_frequency_domain_channel(tf: np.ndarray, wave: np.ndarray, fx: float, dtx: float, shape: WDMShape) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    work = np.asarray(wave, dtype=np.float64).copy()
    tukey_inplace(work, 8.0 / float(shape.nt))
    hfft = np.fft.rfft(work) * (2.0 * dtx)
    freq = fx + np.arange(hfft.size, dtype=np.float64) / (float(wave.size) * dtx)
    return freq, np.angle(hfft), np.abs(hfft)


def _ucb_tukey_at_wdm_centers(
    shape: WDMShape,
    nfx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the heterodyned-FFT Tukey window and its first two derivatives."""

    nx = shape.nt * nfx
    dtx = shape.DT / float(nfx)
    alpha = 8.0 / float(shape.nt)
    imin = int(alpha * float(nx - 1) / 2.0)
    imax = int(float(nx - 1) * (1.0 - alpha / 2.0))
    nwin = nx - imax
    sample_index = np.arange(shape.nt, dtype=np.float64) * shape.DT / dtx
    window = np.ones(shape.nt, dtype=np.float64)
    window_dot = np.zeros(shape.nt, dtype=np.float64)
    window_ddot = np.zeros(shape.nt, dtype=np.float64)
    if imin > 0:
        left = sample_index < float(imin)
        argument = PI * sample_index[left] / float(imin)
        rate = PI / (float(imin) * dtx)
        window[left] = 0.5 * (1.0 - np.cos(argument))
        window_dot[left] = 0.5 * rate * np.sin(argument)
        window_ddot[left] = 0.5 * rate * rate * np.cos(argument)
    if nwin > 0:
        right = sample_index > float(imax)
        argument = PI * (sample_index[right] - float(imax)) / float(nwin)
        rate = PI / (float(nwin) * dtx)
        window[right] = 0.5 * (1.0 + np.cos(argument))
        window_dot[right] = -0.5 * rate * np.sin(argument)
        window_ddot[right] = -0.5 * rate * rate * np.cos(argument)
    return window, window_dot, window_ddot


def ucb_chirplet_lookup_channel(
    source: UCBSourceParams,
    tdi: TDIGrid,
    p_splines: list[CubicSpline],
    channel: str,
    kx: int,
    kw: int,
    shape: WDMShape,
    *,
    chirp_rate_max: float = 8.0,
    chirp_rate_step: float = 0.1,
    frequency_step: float = 0.01,
    amplitude_order: int = 1,
) -> tuple[WDMChannel, dict[str, float]]:
    """Evaluate a slowly evolving Galactic-binary track by chirplet lookup.

    The lookup coordinate ``v = fdot * Tfilt / DF`` is signed. Its symmetric
    table therefore treats positive radiation-reaction chirps and negative
    mass-transfer chirps on exactly the same footing.
    """

    from wdm_chirplet_lookup import build_wdm_chirplet_lookup

    if amplitude_order < 0 or amplitude_order > 2:
        raise ValueError("chirplet amplitude expansion is supported through order 2")
    times = np.arange(shape.nt, dtype=np.float64) * shape.DT
    layers = np.arange(kx + 1, kx + kw + 1, dtype=np.int64)
    layers = layers[(layers > 0) & (layers < shape.nf)]
    if layers.size == 0:
        raise RuntimeError("galactic-binary lookup band contains no WDM layers")

    # Reconstruct the same post-TDI carrier represented by the sparse AP grid.
    reference_time = spacecraft0_reference_time(times, source, p_splines)
    phase, _source_amplitude, source_frequency = source.evaluate(reference_time)
    source_fdot, source_fddot = source.frequency_derivatives()
    source_frequency_dot = source_fdot + source_fddot * reference_time

    _, _, kv = sky_vectors(source)
    position_dot = np.column_stack(
        [np.asarray(spline(times, 1), dtype=np.float64) for spline in p_splines[:3]]
    )
    position_ddot = np.column_stack(
        [np.asarray(spline(times, 2), dtype=np.float64) for spline in p_splines[:3]]
    )
    reference_dot = 1.0 - position_dot @ kv
    reference_ddot = -(position_ddot @ kv)

    amplitude_spline = CubicSpline(
        tdi.time, tdi.amplitude[channel], bc_type="natural"
    )
    offset_spline = CubicSpline(
        tdi.time, tdi.phase_offset[channel], bc_type="natural"
    )
    response_amplitude = np.asarray(amplitude_spline(times), dtype=np.float64)
    response_amplitude_dot = np.asarray(amplitude_spline(times, 1), dtype=np.float64)
    response_amplitude_ddot = np.asarray(amplitude_spline(times, 2), dtype=np.float64)
    phase += np.asarray(offset_spline(times), dtype=np.float64)
    frequency = (
        source_frequency * reference_dot
        + np.asarray(offset_spline(times, 1), dtype=np.float64) / (2.0 * PI)
    )
    fdot = (
        source_frequency_dot * reference_dot * reference_dot
        + source_frequency * reference_ddot
        + np.asarray(offset_spline(times, 2), dtype=np.float64) / (2.0 * PI)
    )

    window, window_dot, window_ddot = _ucb_tukey_at_wdm_centers(
        shape, kw + 1
    )
    amplitude = response_amplitude * window
    amplitude_dot = response_amplitude_dot * window + response_amplitude * window_dot
    amplitude_ddot = (
        response_amplitude_ddot * window
        + 2.0 * response_amplitude_dot * window_dot
        + response_amplitude * window_ddot
    )

    listn = np.tile(np.arange(shape.nt, dtype=np.int64), layers.size)
    listm = np.repeat(layers, shape.nt)
    local_amplitude = np.tile(amplitude, layers.size)
    local_amplitude_dot = np.tile(amplitude_dot, layers.size)
    local_amplitude_ddot = np.tile(amplitude_ddot, layers.size)
    local_phase = np.tile(phase, layers.size)
    local_frequency = np.tile(frequency, layers.size)
    local_fdot = np.tile(fdot, layers.size)
    frequency_offset = local_frequency / shape.DF - listm.astype(np.float64)
    local_rate = local_fdot * shape.Tfilt / shape.DF

    amplitude_scale = max(float(np.max(np.abs(amplitude))), np.finfo(float).tiny)
    relevant = np.abs(local_amplitude) > 1.0e-13 * amplitude_scale
    required_rate = (
        float(np.max(np.abs(local_rate[relevant]))) if np.any(relevant) else 0.0
    )
    if required_rate > chirp_rate_max:
        raise RuntimeError(
            "galactic-binary chirplet rate cap is too small: "
            f"required |fdot*Tfilt/DF|={required_rate:.6g}, cap={chirp_rate_max:.6g}"
        )
    adaptive_rate_max = max(
        0.5,
        chirp_rate_step
        * math.ceil((required_rate + chirp_rate_step) / chirp_rate_step),
    )
    adaptive_rate_max = min(chirp_rate_max, adaptive_rate_max)
    table_start = time.perf_counter()
    lookup = build_wdm_chirplet_lookup(
        shape,
        maximum_moment_order=0,
        chirp_rate_max=adaptive_rate_max,
        chirp_rate_step=chirp_rate_step,
        frequency_step=frequency_step,
    )
    table_seconds = time.perf_counter() - table_start

    evaluation_start = time.perf_counter()
    overlap, inside = lookup.interpolate(0, frequency_offset, local_rate)
    integral = local_amplitude * overlap
    if amplitude_order >= 1:
        overlap_u, inside_u = lookup.interpolate(
            0, frequency_offset, local_rate, frequency_derivative_order=1
        )
        integral += local_amplitude_dot * overlap_u / (1j * 2.0 * PI * shape.DF)
        inside &= inside_u
    if amplitude_order >= 2:
        overlap_uu, inside_uu = lookup.interpolate(
            0, frequency_offset, local_rate, frequency_derivative_order=2
        )
        integral += 0.5 * local_amplitude_ddot * overlap_uu / (
            1j * 2.0 * PI * shape.DF
        ) ** 2
        inside &= inside_uu
    packet = np.exp(1j * local_phase) * integral
    even = ((listn + listm) % 2) == 0
    values = np.where(even, packet.real, -packet.imag)
    values[~inside] = 0.0
    evaluation_seconds = time.perf_counter() - evaluation_start

    nmid = np.full(shape.nf, -1, dtype=np.int64)
    nsize = np.zeros(shape.nf, dtype=np.int64)
    mark_heterodyne_band_pixels(nmid, nsize, kx, kw, shape)
    channel_out = WDMChannel(
        np.empty(0), np.empty(0), np.empty(0), nmid, nsize,
        listn, listm, np.asarray(values, dtype=np.float64),
    )
    diagnostics = {
        "lookup_table": table_seconds,
        "lookup_evaluate": evaluation_seconds,
        "lookup_entries": float(lookup.entries),
        "lookup_rate_min": float(np.min(local_rate[relevant])),
        "lookup_rate_max": float(np.max(local_rate[relevant])),
        "lookup_table_rate_limit": float(lookup.chirp_rate[-1]),
    }
    return channel_out, diagnostics


def combine_sparse_wdm(
    shape: WDMShape,
    listn_parts: list[np.ndarray],
    listm_parts: list[np.ndarray],
    value_parts: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not listn_parts:
        empty_i = np.empty(0, dtype=np.int64)
        empty = np.empty(0, dtype=np.float64)
        return empty_i, empty_i, empty
    listn = np.concatenate(listn_parts).astype(np.int64, copy=False)
    listm = np.concatenate(listm_parts).astype(np.int64, copy=False)
    values = np.concatenate(value_parts).astype(np.float64, copy=False)
    linear = listn * (shape.nf + 1) + listm
    order = np.argsort(linear)
    linear = linear[order]
    values = values[order]
    unique, starts = np.unique(linear, return_index=True)
    summed = np.add.reduceat(values, starts)
    keep = summed != 0.0
    unique = unique[keep]
    summed = summed[keep]
    return unique // (shape.nf + 1), unique % (shape.nf + 1), summed


def mark_heterodyne_band_pixels(nmid: np.ndarray, nsize: np.ndarray, kx: int, kw: int, shape: WDMShape) -> None:
    for local_m in range(1, kw + 1):
        global_m = kx + local_m
        if 0 < global_m < shape.nf:
            nmid[global_m] = shape.nt // 2
            nsize[global_m] = shape.nt


def eccentric_harmonic_wdm_band(group: list[tuple[EccentricCarrierGrid, TDIGrid]]) -> tuple[int, int]:
    """Choose one heterodyne band covering all sidebands of an eccentric harmonic."""

    kmin = min(int(round(float(carrier.intrinsic.setup[0]))) for carrier, _tdi in group)
    kmax = max(int(round(float(carrier.intrinsic.setup[0] + carrier.intrinsic.setup[1]))) for carrier, _tdi in group)
    return kmin, max(kmax - kmin, 1)


def _eval_cubic_spline_array(spline: CubicSpline, x: np.ndarray) -> np.ndarray:
    knots = spline.x
    coeffs = spline.c
    indices = np.searchsorted(knots, x, side="right") - 1
    indices = np.clip(indices, 0, knots.size - 2)
    dx = x - knots[indices]
    return ((coeffs[0, indices] * dx + coeffs[1, indices]) * dx + coeffs[2, indices]) * dx + coeffs[3, indices]


def wdmtran_frequency_layer(m: int, data: np.ndarray, window: np.ndarray, scale: float) -> np.ndarray:
    ntx = data.size
    dx = np.fft.ifft(data * window) * ntx
    out = np.empty(ntx, dtype=np.float64)
    n = np.arange(ntx)
    even = ((n + m) % 2) == 0
    if m % 2 == 0:
        out[even] = scale * dx.real[even]
        out[~even] = scale * dx.imag[~even]
    else:
        out[even] = scale * dx.real[even]
        out[~even] = -scale * dx.imag[~even]
    return out


def prune_nonincreasing_frequency_samples(freq: np.ndarray, phase: np.ndarray, amp: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if freq.size < 2:
        return freq, phase, amp
    keep = np.zeros(freq.size, dtype=bool)
    keep[0] = True
    last = float(freq[0])
    for i in range(1, freq.size):
        if float(freq[i]) > last:
            keep[i] = True
            last = float(freq[i])
    return freq[keep], phase[keep], amp[keep]


def wdm_track(freq: np.ndarray, phase: np.ndarray, amp: np.ndarray, nmid: np.ndarray, nsize: np.ndarray, shape: WDMShape) -> WDMChannel:
    amp_spline = Akima1DInterpolator(freq, amp, extrapolate=True)
    phase_spline = CubicSpline(freq, phase, bc_type="natural")
    active_m = [m for m in range(1, shape.nf) if nmid[m] > 0]
    if not active_m:
        empty_i = np.empty(0, dtype=np.int64)
        empty = np.empty(0, dtype=np.float64)
        return WDMChannel(freq, phase, amp, nmid.copy(), nsize.copy(), empty_i, empty_i, empty)

    # The C code transforms one active layer at a time. In Python the per-call
    # FFT overhead is significant, so layers with the same local packet size are
    # assembled into one batched IFFT while preserving the original layer order.
    order = {m: i for i, m in enumerate(active_m)}
    groups: dict[int, list[int]] = {}
    for m in active_m:
        groups.setdefault(int(nsize[m]), []).append(m)

    cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]] = {}
    listn: list[np.ndarray | None] = [None] * len(active_m)
    listm: list[np.ndarray | None] = [None] * len(active_m)
    values: list[np.ndarray | None] = [None] * len(active_m)

    for ntx, modes in groups.items():
        if ntx not in cache:
            half = ntx // 2
            base_n = np.arange(ntx, dtype=np.int64)
            j = np.arange(-half, half, dtype=np.int64)
            jf = j.astype(np.float64)
            dfx = 1.0 / (float(ntx) * shape.DT)
            dom = 2.0 * PI / (float(ntx) * shape.DT)
            phihf = phitilde(np.arange(0, half + 1, dtype=np.float64) * dom, shape)
            window = phihf[np.abs(base_n - half)]
            window[0] = 0.0
            scale = math.sqrt(8.0 * PI / 15.0) / (float(ntx) * shape.DT)
            cache[ntx] = (base_n, j, jf, window, dfx, scale)
        base_n, j, jf, window, dfx, scale = cache[ntx]
        half = ntx // 2

        mode_array = np.asarray(modes, dtype=np.int64)
        n_array = nmid[mode_array].astype(np.float64)
        fgrid = jf[np.newaxis, :] * dfx + mode_array[:, np.newaxis].astype(np.float64) * shape.DF
        start_grid = (n_array[:, np.newaxis] - float(half)) * shape.DT
        mask = (base_n[np.newaxis, :] > 0) & (fgrid > freq[0]) & (fgrid < freq[-1])

        data = np.zeros((mode_array.size, ntx), dtype=np.complex128)
        if np.any(mask):
            fvals = fgrid[mask]
            aa = _eval_cubic_spline_array(amp_spline, fvals)
            pp = _eval_cubic_spline_array(phase_spline, fvals) + 2.0 * PI * fvals * np.broadcast_to(start_grid, fgrid.shape)[mask]
            data[mask] = aa * (np.cos(pp) + 1j * np.sin(pp))

        dx = np.fft.ifft(data * window[np.newaxis, :], axis=1) * ntx
        even = ((base_n[np.newaxis, :] + mode_array[:, np.newaxis]) % 2) == 0
        imag_sign = np.where((mode_array % 2) == 0, 1.0, -1.0)[:, np.newaxis]
        wdmout = scale * np.where(even, dx.real, imag_sign * dx.imag)

        for row, m in enumerate(modes):
            n = int(nmid[m])
            out_n = base_n + n - half
            keep = (out_n >= 0) & (out_n < shape.nt)
            slot = order[m]
            listn[slot] = out_n[keep]
            listm[slot] = np.full(np.count_nonzero(keep), m, dtype=np.int64)
            values[slot] = wdmout[row, keep]

    return WDMChannel(
        freq,
        phase,
        amp,
        nmid.copy(),
        nsize.copy(),
        np.concatenate([x for x in listn if x is not None]),
        np.concatenate([x for x in listm if x is not None]),
        np.concatenate([x for x in values if x is not None]),
    )


def wdm_channel_from_dense(
    freq: np.ndarray,
    phase: np.ndarray,
    amp: np.ndarray,
    nmid: np.ndarray,
    nsize: np.ndarray,
    dense: np.ndarray,
    shape: WDMShape,
) -> WDMChannel:
    if NUMBA_AVAILABLE:
        listn_array, listm_array, value_array = _wdm_channel_from_dense_jit(
            nmid,
            nsize,
            np.ascontiguousarray(dense, dtype=np.float64),
            int(shape.nt),
            int(shape.nf),
        )
        return WDMChannel(
            freq,
            phase,
            amp,
            nmid.copy(),
            nsize.copy(),
            listn_array,
            listm_array,
            value_array,
        )

    listn: list[np.ndarray] = []
    listm: list[np.ndarray] = []
    values: list[np.ndarray] = []
    base = np.arange(shape.nt, dtype=np.int64)

    for m in range(1, shape.nf):
        if nmid[m] < 0 or nsize[m] <= 0:
            continue
        half = int(nsize[m]) // 2
        out_n = base[: int(nsize[m])] + int(nmid[m]) - half
        keep = (out_n >= 0) & (out_n < shape.nt)
        if not np.any(keep):
            continue
        rows = out_n[keep]
        listn.append(rows)
        listm.append(np.full(rows.size, m, dtype=np.int64))
        values.append(dense[rows, m])

    if not listn:
        empty_i = np.empty(0, dtype=np.int64)
        empty = np.empty(0, dtype=np.float64)
        return WDMChannel(freq, phase, amp, nmid.copy(), nsize.copy(), empty_i, empty_i, empty)

    return WDMChannel(
        freq,
        phase,
        amp,
        nmid.copy(),
        nsize.copy(),
        np.concatenate(listn),
        np.concatenate(listm),
        np.concatenate(values),
    )


@njit(cache=True)
def _wdm_channel_from_dense_jit(
    nmid: np.ndarray,
    nsize: np.ndarray,
    dense: np.ndarray,
    nt: int,
    nf: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gather the planned packets directly into sparse storage."""

    count = 0
    for m in range(1, nf):
        if nmid[m] < 0 or nsize[m] <= 0:
            continue
        half = int(nsize[m]) // 2
        for local in range(int(nsize[m])):
            row = local + int(nmid[m]) - half
            if 0 <= row < nt:
                count += 1
    listn = np.empty(count, dtype=np.int64)
    listm = np.empty(count, dtype=np.int64)
    values = np.empty(count, dtype=np.float64)
    index = 0
    for m in range(1, nf):
        if nmid[m] < 0 or nsize[m] <= 0:
            continue
        half = int(nsize[m]) // 2
        for local in range(int(nsize[m])):
            row = local + int(nmid[m]) - half
            if 0 <= row < nt:
                listn[index] = row
                listm[index] = m
                values[index] = dense[row, m]
                index += 1
    return listn, listm, values


def _smooth_step(value: float, lower: float, upper: float) -> float:
    if value <= lower:
        return 0.0
    if value >= upper:
        return 1.0
    x = (value - lower) / (upper - lower)
    return 0.5 * (1.0 - math.cos(PI * x))


def apply_short_fft_threshold(
    dense: np.ndarray,
    nmid: np.ndarray,
    nsize: np.ndarray,
    short_htime: np.ndarray,
    setup: np.ndarray,
    f_replace_start: float,
    f_replace_stop: float,
    shape: WDMShape,
    *,
    blend_half_width: float = 0.0,
) -> tuple[int, int]:
    """Replace or blend endpoint WDM layers with native short-FFT bins.

    A positive ``blend_half_width`` selects the current coefficient-space
    blend used by the Fisher-tested C path.  No endpoint ``A(f),phi(f)`` spline
    bridge is constructed: native short-FFT bins go directly through the local
    Meyer packet transform.
    """

    dte = float(setup[0])
    nshort = int(setup[1])
    tes = float(setup[2])
    if dte <= 0.0 or nshort < 2:
        return 0, 0
    if not math.isfinite(f_replace_stop) or f_replace_stop <= 0.0:
        f_replace_stop = 0.5 / dte
    use_blend = math.isfinite(blend_half_width) and blend_half_width > 0.0
    blend_lo = max(0.0, f_replace_start - blend_half_width) if use_blend else 0.0
    blend_hi = f_replace_start + blend_half_width if use_blend else 0.0

    packet_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]] = {}
    short_fft_cache: dict[int, np.ndarray] = {}
    groups: dict[tuple[int, int], list[tuple[int, float]]] = {}
    for m in range(1, shape.nf):
        if nmid[m] < 0 or nsize[m] <= 0:
            continue

        fcenter = float(m) * shape.DF
        flower = fcenter - shape.FB
        fupper = fcenter + shape.FB
        if use_blend:
            replace_layer = fupper >= blend_lo and flower <= f_replace_stop
            blend_weight = _smooth_step(fcenter, blend_lo, blend_hi)
            if blend_weight <= 0.0:
                replace_layer = False
        else:
            replace_layer = flower >= f_replace_start and fupper <= f_replace_stop
            blend_weight = 1.0

        ntx = int(nsize[m])
        nblock = int(round(float(ntx) * shape.DT / dte))
        if nblock < 2 or (nblock & (nblock - 1)) != 0:
            replace_layer = False
        if nblock < nshort:
            replace_layer = False
        if not replace_layer:
            continue

        groups.setdefault((ntx, nblock), []).append((m, blend_weight))

    layers = sum(len(group) for group in groups.values())
    pixels = 0
    # Keep the batched complex work arrays modest even if a future plan puts
    # many long packets in one size class.
    maximum_batch_elements = 1 << 20

    for (ntx, nblock), group in groups.items():

        if nblock not in short_fft_cache:
            padded = np.zeros(nblock, dtype=np.float64)
            ncopy = min(nshort, nblock)
            padded[:ncopy] = short_htime[:ncopy]
            short_fft_cache[nblock] = np.fft.rfft(padded) * (2.0 * dte)
        hfft = short_fft_cache[nblock]

        if ntx not in packet_cache:
            half = ntx // 2
            base_n = np.arange(ntx, dtype=np.int64)
            j = np.arange(-half, half, dtype=np.int64)
            jf = j.astype(np.float64)
            dfx = 1.0 / (float(ntx) * shape.DT)
            dom = 2.0 * PI / (float(ntx) * shape.DT)
            phihf = phitilde(np.arange(0, half + 1, dtype=np.float64) * dom, shape)
            window = phihf[np.abs(base_n - half)]
            window[0] = 0.0
            scale = math.sqrt(8.0 * PI / 15.0) / (float(ntx) * shape.DT)
            packet_cache[ntx] = (base_n, j, jf, window, dfx, scale)
        base_n, _j, jf, window, dfx, scale = packet_cache[ntx]
        half = ntx // 2

        batch_rows = max(1, maximum_batch_elements // ntx)
        nyquist = 0.5 / dte
        for first in range(0, len(group), batch_rows):
            batch = group[first : first + batch_rows]
            modes = np.asarray([item[0] for item in batch], dtype=np.int64)
            weights = np.asarray([item[1] for item in batch], dtype=np.float64)
            centers = np.asarray(nmid[modes], dtype=np.int64)
            fgrid = (
                jf[np.newaxis, :] * dfx
                + modes[:, np.newaxis].astype(np.float64) * shape.DF
            )
            scaled_bins = fgrid * (float(nblock) * dte)
            bins = np.rint(scaled_bins).astype(np.int64)
            valid = (
                (base_n[np.newaxis, :] > 0)
                & (fgrid > 0.0)
                & (fgrid < nyquist)
                & (np.abs(scaled_bins - bins.astype(np.float64)) <= 1.0e-6)
                & (bins > 0)
                & (bins < hfft.size)
            )
            data = np.zeros(fgrid.shape, dtype=np.complex128)
            row_index, frequency_index = np.nonzero(valid)
            if row_index.size:
                selected_frequency = fgrid[row_index, frequency_index]
                t0 = (
                    centers[row_index].astype(np.float64) - float(half)
                ) * shape.DT
                phase_rot = 2.0 * PI * selected_frequency * (
                    shape.Tobs - tes + t0
                )
                data[row_index, frequency_index] = hfft[
                    bins[row_index, frequency_index]
                ] * (np.cos(phase_rot) + 1j * np.sin(phase_rot))

            transformed = np.fft.ifft(
                data * window[np.newaxis, :], axis=1
            ) * ntx
            even = (
                (base_n[np.newaxis, :] + modes[:, np.newaxis]) % 2
            ) == 0
            imag_sign = np.where((modes % 2) == 0, 1.0, -1.0)[:, np.newaxis]
            wdmout = scale * np.where(
                even, transformed.real, imag_sign * transformed.imag
            )

            output_time = (
                base_n[np.newaxis, :] + centers[:, np.newaxis] - half
            )
            keep = (output_time >= 0) & (output_time < shape.nt)
            packet_index, local_index = np.nonzero(keep)
            rows = output_time[packet_index, local_index]
            columns = modes[packet_index]
            values = wdmout[packet_index, local_index]
            if use_blend:
                local_weight = weights[packet_index]
                dense[rows, columns] = (
                    (1.0 - local_weight) * dense[rows, columns]
                    + local_weight * values
                )
            else:
                dense[rows, columns] = values
            pixels += int(rows.size)

    return layers, pixels


def replace_wdm_with_short_fft_threshold(
    dense: np.ndarray,
    nmid: np.ndarray,
    nsize: np.ndarray,
    short_htime: np.ndarray,
    setup: np.ndarray,
    f_replace_start: float,
    f_replace_stop: float,
    shape: WDMShape,
) -> tuple[int, int]:
    """Compatibility wrapper for the historical hard endpoint replacement."""

    return apply_short_fft_threshold(
        dense,
        nmid,
        nsize,
        short_htime,
        setup,
        f_replace_start,
        f_replace_stop,
        shape,
    )


def generate_tdi_wdm(
    source: SourceParams = SourceParams(),
    shape: WDMShape = WDMShape(),
    channels: Iterable[str] = ("X", "Y", "Z"),
    nsmax: int = 10000,
    coefficient_backend: str = "auto",
    compute_wdm: bool = True,
    blend_endpoint: bool = DEFAULT_WDM_BLEND_ENDPOINT,
    blend_half_width_layers: float = DEFAULT_WDM_BLEND_HALF_WIDTH_LAYERS,
    intrinsic_backend: str = "local",
    direct_endpoint_tdi: bool = True,
    tdi_generation: int = 1,
) -> TDIWDMResult | THMTDIWDMResult:
    if tdi_generation == 2:
        from phenomthm_tdi_wdm import generate_thm_tdi_wdm

        if intrinsic_backend != "local":
            raise ValueError("22-only TDI-2 uses the local THM backbone")
        return generate_thm_tdi_wdm(
            source, shape, channels, modes="22pair", nsmax=nsmax,
            coefficient_backend=coefficient_backend, compute_wdm=compute_wdm,
            tdi_generation=2,
        )
    if tdi_generation != 1:
        raise ValueError("tdi_generation must be 1 or 2")
    requested = tuple(ch.upper() for ch in channels)
    timings: dict[str, float] = {}

    constellation_times, l_splines, p_splines, v_splines = build_constellation_splines(shape)

    start = time.perf_counter()
    intrinsic = build_intrinsic_grid(
        source,
        shape,
        p_splines,
        float(constellation_times[0]),
        float(constellation_times[-1]),
        nsmax=nsmax,
        coefficient_backend=coefficient_backend,
        intrinsic_backend=intrinsic_backend,
    )
    timings["adaptive_ap"] = time.perf_counter() - start

    start = time.perf_counter()
    tdi = compute_tdi_grid(source, intrinsic, l_splines, p_splines, v_splines)
    timings["fast_tdi"] = time.perf_counter() - start

    # At this point the fast TDI response is available as a coarse signed
    # amplitude/phase representation for each channel:
    #   h_ch(t) = tdi.amplitude[ch] * cos(tdi.channel_phase(ch)).
    # A user who wants a time-domain or ordinary frequency-domain likelihood can
    # use this directly instead of the WDM packets. The production WDM path below
    # uses SPA-only A(f), phi(f) samples before the handoff, then overwrites the
    # endpoint WDM layers directly from the short merger/ringdown FFT.  This
    # avoids the old spline bridge through endpoint A(f), phi(f).
    start = time.perf_counter()
    times = tdi.time
    m1, m2, _chi1, _chi2 = source.masses_seconds()
    total_mass = m1 + m2
    chirp_mass = (m1 * m2) ** (3.0 / 5.0) / (total_mass ** (1.0 / 5.0))
    fring = intrinsic.model.omega_ring / (2.0 * PI * total_mass)
    fdamp = intrinsic.model.alpha1rd / total_mass
    t_transition = source.tc + total_mass * T_CUT_FREQ
    blend_half_width = float(blend_half_width_layers) * shape.DF if blend_endpoint else 0.0
    if blend_endpoint and (not math.isfinite(blend_half_width) or blend_half_width <= 0.0):
        raise ValueError("blend_half_width_layers must be positive when endpoint blending is enabled")
    first_nmid = np.full(shape.nf, -1, dtype=np.int64)
    first_nsize = np.zeros(shape.nf, dtype=np.int64)
    first_plan_recorded = False
    response_merger = float(
        np.interp(source.tc, intrinsic.detector_time[: times.size], times)
    )

    channels_out: dict[str, WDMChannel] = {}
    endpoint_cache: dict[tuple[float, ...], dict[str, np.ndarray]] = {}
    frequency_domain_time = 0.0
    wdm_packet_time = 0.0
    for ch in requested:
        phase = tdi.channel_phase(ch)
        fd_start = time.perf_counter()
        freq_track = tdi_frequency_track(times, phase, intrinsic.frequency[: times.size])
        setup = transform_plan(
            chirp_mass,
            total_mass,
            source.tc,
            times,
            2.0 * PI * freq_track,
            t_transition,
            fring,
            fdamp,
            times[-1],
            shape,
            center_source_time=intrinsic.detector_time[: times.size],
        )
        f_clean_endpoint_start = float(np.interp(float(setup[5]), times, freq_track))
        if not math.isfinite(f_clean_endpoint_start) or f_clean_endpoint_start <= 0.0:
            positive_track = freq_track[np.isfinite(freq_track) & (freq_track > 0.0)]
            f_clean_endpoint_start = float(positive_track[-1]) if positive_track.size else shape.DF
        original_flat_time = float(setup[2]) + float(setup[3])
        original_taper_guard = float(
            np.interp(original_flat_time, times, freq_track)
        )
        if math.isfinite(original_taper_guard) and original_taper_guard > 0.0:
            f_clean_endpoint_start = max(
                f_clean_endpoint_start, original_taper_guard
            )
        setup = plan_endpoint_taper_flat_time(
            setup,
            times,
            ((freq_track,),),
            (max(0.0, f_clean_endpoint_start - blend_half_width),),
            response_merger,
        )

        # Keep the SPA samples far enough into the endpoint boundary layer that
        # the last unreplaced Meyer packet has valid Fourier input across its
        # full support.  The endpoint FFT itself supplies the high-frequency
        # layers, so no short-FFT A(f),phi(f) samples are appended here.
        setup_spa = setup.copy()
        f_spa_stop = f_clean_endpoint_start + shape.FB
        if blend_endpoint:
            f_spa_stop += blend_half_width
        jstop = min(max(int(setup_spa[4]), 1), times.size - 1)
        while jstop + 1 < times.size and float(freq_track[jstop]) < f_spa_stop:
            jstop += 1
        if float(jstop) > setup_spa[4]:
            setup_spa[4] = float(jstop)
        spa_support_tmax = float(times[jstop])

        freq, fphase, famp = ftran_spa_only(setup_spa, times, tdi.amplitude[ch], phase, shape)
        freq, fphase, famp = prune_nonincreasing_frequency_samples(freq, fphase, famp)
        short_htime = np.empty(0, dtype=np.float64)
        if compute_wdm:
            if direct_endpoint_tdi:
                endpoint_key = tuple(float(value) for value in setup[:4])
                if endpoint_key not in endpoint_cache:
                    endpoint_cache[endpoint_key] = build_direct_endpoint_waveforms(
                        setup,
                        source,
                        intrinsic,
                        l_splines,
                        p_splines,
                        v_splines,
                    )
                short_htime = endpoint_cache[endpoint_key][ch]
            else:
                short_htime = build_short_endpoint_waveform(
                    setup, times, tdi.amplitude[ch], phase
                )
        frequency_domain_time += time.perf_counter() - fd_start
        if compute_wdm:
            # The SPA spectrum represents only the early single-valued branch.
            # Late post-TDI excursions through low frequencies belong to the
            # endpoint FFT and must not enlarge the SPA packet mask.
            nmid, nsize = wdm_pixels_range(times, freq_track, float(times[0]), spa_support_tmax, shape)
            direct_start_time = float(setup[2]) + float(setup[3]) + SHORTFFT_MERGER_TAPER_MARGIN_SECONDS
            if not blend_endpoint:
                direct_start_time = max(direct_start_time, float(setup[5]))
            endpoint_nmid, endpoint_nsize = wdm_pixels_range(
                times,
                freq_track,
                direct_start_time,
                float(setup[2]) + float(setup[0]) * float(setup[1]),
                shape,
            )
            tail_index = int(np.argmax(np.abs(tdi.amplitude[ch])))
            f_endpoint_max = float(setup[6]) / (float(setup[0]) * float(setup[1]))
            if not math.isfinite(f_endpoint_max) or f_endpoint_max <= 0.0:
                f_endpoint_max = 0.5 / float(setup[0])
            f_endpoint_max = min(f_endpoint_max, 0.5 / float(setup[0]))
            wdm_pixels_add_merger_frequency_tail(endpoint_nmid, endpoint_nsize, f_endpoint_max, float(times[tail_index]), shape)
            nmid_total = nmid.copy()
            nsize_total = nsize.copy()
            merge_pixel_plans(nmid_total, nsize_total, endpoint_nmid, endpoint_nsize, shape)
            if not first_plan_recorded:
                first_nmid = nmid_total.copy()
                first_nsize = nsize_total.copy()
                first_plan_recorded = True
            wdm_start = time.perf_counter()
            spa_channel = wdm_track(freq, fphase, famp, nmid, nsize, shape)
            dense = spa_channel.dense(shape)
            apply_short_fft_threshold(
                dense,
                endpoint_nmid,
                endpoint_nsize,
                short_htime,
                setup,
                f_clean_endpoint_start,
                f_endpoint_max,
                shape,
                blend_half_width=blend_half_width,
            )
            channels_out[ch] = wdm_channel_from_dense(freq, fphase, famp, nmid_total, nsize_total, dense, shape)
            wdm_packet_time += time.perf_counter() - wdm_start
        else:
            nmid = np.full(shape.nf, -1, dtype=np.int64)
            nsize = np.zeros(shape.nf, dtype=np.int64)
            if not first_plan_recorded:
                first_nmid = nmid.copy()
                first_nsize = nsize.copy()
                first_plan_recorded = True
            empty_i = np.empty(0, dtype=np.int64)
            empty = np.empty(0, dtype=np.float64)
            channels_out[ch] = WDMChannel(freq, fphase, famp, nmid, nsize, empty_i, empty_i, empty)
    timings["fast_frequency_domain"] = frequency_domain_time
    timings["fast_wdm_packets"] = wdm_packet_time
    timings["fast_wdm"] = time.perf_counter() - start if compute_wdm else 0.0

    return TDIWDMResult(source, shape, intrinsic, tdi, first_nmid, first_nsize, channels_out, timings)


def generate_ucb_tdi_wdm(
    source: UCBSourceParams = UCBSourceParams(),
    shape: WDMShape = WDMShape(),
    channels: Iterable[str] = ("X", "Y", "Z"),
    compute_wdm: bool = True,
    wdm_method: str = "fft",
    lookup_chirp_rate_max: float = 8.0,
    lookup_chirp_rate_step: float = 0.1,
    lookup_frequency_step: float = 0.01,
    lookup_amplitude_order: int = 1,
    constellation_data: tuple[
        np.ndarray,
        list[CubicSpline],
        list[CubicSpline],
        list[CubicSpline],
    ] | None = None,
    tdi_generation: int = 1,
) -> TDIWDMResult:
    """Generate a galactic-binary TDI response and heterodyned WDM packets.

    The default ``fft`` method follows ``wavelet_TDI_hetF`` in
    ``GB/response.c``. The experimental ``lookup`` method evaluates local
    linear-chirp overlaps directly on the WDM track and supports either sign
    of ``fdot``.
    """

    requested = tuple(ch.upper() for ch in channels)
    method = wdm_method.lower()
    if method not in {"fft", "lookup"}:
        raise ValueError("galactic-binary WDM method must be 'fft' or 'lookup'")
    timings: dict[str, float] = {}
    start = time.perf_counter()
    if constellation_data is None:
        constellation_data = build_constellation_splines(shape)
    constellation_times, l_splines, p_splines, v_splines = constellation_data
    _ = constellation_times
    timings["constellation_setup"] = time.perf_counter() - start

    start = time.perf_counter()
    intrinsic = build_ucb_intrinsic_grid(source, shape)
    timings["adaptive_ap"] = time.perf_counter() - start

    start = time.perf_counter()
    tdi = compute_ucb_tdi_grid(source, intrinsic, l_splines, p_splines,
                               v_splines, tdi_generation=tdi_generation)
    timings["fast_tdi"] = time.perf_counter() - start

    start = time.perf_counter()
    kx = int(round(float(intrinsic.setup[0])))
    kw = int(round(float(intrinsic.setup[1])))
    nfx = kw + 1
    dtx = shape.DT / float(nfx)

    base_nmid = np.full(shape.nf, -1, dtype=np.int64)
    base_nsize = np.zeros(shape.nf, dtype=np.int64)
    mark_heterodyne_band_pixels(base_nmid, base_nsize, kx, kw, shape)

    channels_out: dict[str, WDMChannel] = {}
    frequency_domain_time = 0.0
    wdm_packet_time = 0.0
    lookup_table_time = 0.0
    lookup_evaluation_time = 0.0
    for ch in requested:
        if method == "lookup" and compute_wdm:
            wdm_start = time.perf_counter()
            channel_out, diagnostics = ucb_chirplet_lookup_channel(
                source,
                tdi,
                p_splines,
                ch,
                kx,
                kw,
                shape,
                chirp_rate_max=lookup_chirp_rate_max,
                chirp_rate_step=lookup_chirp_rate_step,
                frequency_step=lookup_frequency_step,
                amplitude_order=lookup_amplitude_order,
            )
            wdm_packet_time += time.perf_counter() - wdm_start
            lookup_table_time += diagnostics["lookup_table"]
            lookup_evaluation_time += diagnostics["lookup_evaluate"]
            channels_out[ch] = channel_out
            continue

        fd_start = time.perf_counter()
        tf, wave, _pref, fx = ucb_heterodyned_time_series(source, intrinsic, tdi, p_splines, ch, kx, kw, shape)
        freq, fphase, famp = ucb_frequency_domain_channel(tf, wave, fx, dtx, shape)
        frequency_domain_time += time.perf_counter() - fd_start

        if compute_wdm:
            wdm_start = time.perf_counter()
            listn, listm, values = wdmtran_heterodyne_frequency(kx, kw, wave, shape)
            wdm_packet_time += time.perf_counter() - wdm_start
        else:
            listn = np.empty(0, dtype=np.int64)
            listm = np.empty(0, dtype=np.int64)
            values = np.empty(0, dtype=np.float64)
        channels_out[ch] = WDMChannel(freq, fphase, famp, base_nmid.copy(), base_nsize.copy(), listn, listm, values)

    timings["fast_frequency_domain"] = frequency_domain_time
    timings["fast_wdm_packets"] = wdm_packet_time
    if method == "lookup" and compute_wdm:
        timings["lookup_table"] = lookup_table_time
        timings["lookup_evaluate"] = lookup_evaluation_time
    timings["fast_wdm"] = time.perf_counter() - start if compute_wdm else 0.0
    return TDIWDMResult(source, shape, intrinsic, tdi, base_nmid, base_nsize, channels_out, timings)


def generate_eccentric_ucb_tdi_wdm(
    source: EccentricUCBSourceParams = EccentricUCBSourceParams(),
    shape: WDMShape = WDMShape(),
    channels: Iterable[str] = ("X", "Y", "Z"),
    compute_wdm: bool = True,
    carrier_labels: Iterable[str] | None = None,
    tdi_generation: int = 1,
) -> TDIWDMResult:
    """Generate a low-eccentricity galactic-binary TDI response and WDM packets.

    The waveform is represented as a sum of circular-like carriers: the central
    ``nM`` term and the ``nM +/- 2Phi`` sidebands for harmonics ``n <= nmax``.
    TDI is applied carrier by carrier because the inclination weights differ,
    but the WDM stage groups the sideband triplet for each harmonic onto one
    shared heterodyned grid before transforming it.
    """

    requested = tuple(ch.upper() for ch in channels)
    timings: dict[str, float] = {}
    constellation_times, l_splines, p_splines, v_splines = build_constellation_splines(shape)
    _ = constellation_times

    start = time.perf_counter()
    carriers = build_eccentric_ucb_carrier_grids(source, shape, carrier_labels=carrier_labels)
    timings["adaptive_ap"] = time.perf_counter() - start

    start = time.perf_counter()
    carrier_tdi: list[tuple[EccentricCarrierGrid, TDIGrid]] = []
    for carrier in carriers:
        tdi = compute_ucb_tdi_grid(
            source,
            carrier.intrinsic,
            l_splines,
            p_splines,
            v_splines,
            aplus_override=carrier.aplus,
            across_override=carrier.across,
            tdi_generation=tdi_generation,
        )
        carrier_tdi.append((carrier, tdi))
    timings["fast_tdi"] = time.perf_counter() - start

    start = time.perf_counter()
    base_nmid = np.full(shape.nf, -1, dtype=np.int64)
    base_nsize = np.zeros(shape.nf, dtype=np.int64)
    freq_parts: dict[str, list[np.ndarray]] = {ch: [] for ch in requested}
    phase_parts: dict[str, list[np.ndarray]] = {ch: [] for ch in requested}
    amp_parts: dict[str, list[np.ndarray]] = {ch: [] for ch in requested}
    listn_parts: dict[str, list[np.ndarray]] = {ch: [] for ch in requested}
    listm_parts: dict[str, list[np.ndarray]] = {ch: [] for ch in requested}
    value_parts: dict[str, list[np.ndarray]] = {ch: [] for ch in requested}

    frequency_domain_time = 0.0
    wdm_packet_time = 0.0
    harmonic_groups: dict[int, list[tuple[EccentricCarrierGrid, TDIGrid]]] = {}
    for carrier, tdi in carrier_tdi:
        harmonic_groups.setdefault(carrier.harmonic, []).append((carrier, tdi))

    for harmonic in sorted(harmonic_groups):
        group = harmonic_groups[harmonic]
        kx, kw = eccentric_harmonic_wdm_band(group)
        nfx = kw + 1
        dtx = shape.DT / float(nfx)
        mark_heterodyne_band_pixels(base_nmid, base_nsize, kx, kw, shape)
        for ch in requested:
            fd_start = time.perf_counter()
            tf_group: np.ndarray | None = None
            fx_group = float(kx) * shape.DF
            group_wave: np.ndarray | None = None
            for carrier, tdi in group:
                tf, wave, _pref, fx = ucb_heterodyned_time_series(source, carrier.intrinsic, tdi, p_splines, ch, kx, kw, shape)
                if group_wave is None:
                    tf_group = tf
                    fx_group = fx
                    group_wave = wave.copy()
                else:
                    group_wave += wave
            if group_wave is None or tf_group is None:
                continue
            freq, fphase, famp = ucb_frequency_domain_channel(tf_group, group_wave, fx_group, dtx, shape)
            frequency_domain_time += time.perf_counter() - fd_start
            freq_parts[ch].append(freq)
            phase_parts[ch].append(fphase)
            amp_parts[ch].append(famp)

            if compute_wdm:
                wdm_start = time.perf_counter()
                listn, listm, values = wdmtran_heterodyne_frequency(kx, kw, group_wave, shape)
                wdm_packet_time += time.perf_counter() - wdm_start
                listn_parts[ch].append(listn)
                listm_parts[ch].append(listm)
                value_parts[ch].append(values)

    channels_out: dict[str, WDMChannel] = {}
    for ch in requested:
        if freq_parts[ch]:
            freq = np.concatenate(freq_parts[ch])
            fphase = np.concatenate(phase_parts[ch])
            famp = np.concatenate(amp_parts[ch])
            order = np.argsort(freq)
            freq = freq[order]
            fphase = fphase[order]
            famp = famp[order]
        else:
            freq = np.empty(0, dtype=np.float64)
            fphase = np.empty(0, dtype=np.float64)
            famp = np.empty(0, dtype=np.float64)
        if compute_wdm:
            listn, listm, values = combine_sparse_wdm(shape, listn_parts[ch], listm_parts[ch], value_parts[ch])
        else:
            listn = np.empty(0, dtype=np.int64)
            listm = np.empty(0, dtype=np.int64)
            values = np.empty(0, dtype=np.float64)
        channels_out[ch] = WDMChannel(freq, fphase, famp, base_nmid.copy(), base_nsize.copy(), listn, listm, values)

    timings["fast_frequency_domain"] = frequency_domain_time
    timings["fast_wdm_packets"] = wdm_packet_time
    timings["fast_wdm"] = time.perf_counter() - start if compute_wdm else 0.0
    reference_carrier, reference_tdi = carrier_tdi[0]
    return TDIWDMResult(source, shape, reference_carrier.intrinsic, reference_tdi, base_nmid, base_nsize, channels_out, timings)


def write_wdm_matrix(path: str, matrix: np.ndarray) -> None:
    np.savetxt(path, matrix, fmt="%.15e")


def write_track_pixels(path: str, channel: WDMChannel) -> None:
    order = np.lexsort((channel.listm, channel.listn))
    data = np.column_stack([channel.listn[order], channel.listm[order], channel.values[order]])
    np.savetxt(path, data, fmt=["%d", "%d", "%.15e"], header="n m wdm")


def write_frequency_domain(path: str, channel: WDMChannel) -> None:
    real = channel.amplitude * np.cos(channel.phase)
    imag = channel.amplitude * np.sin(channel.phase)
    data = np.column_stack([channel.freq, channel.amplitude, channel.phase, real, imag])
    np.savetxt(path, data, fmt="%.15e", header="f amplitude phase real imag")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", choices=("phenomt", "ucb", "eccentric"), default="phenomt", help="Waveform model to generate. Defaults to phenomt.")
    parser.add_argument("--tdi-generation", type=int, choices=(1, 2), default=1,
                        help="TDI-2 uses the validated partitioned-FFT THM 22-pair engine for phenomt.")
    parser.add_argument("--channel", action="append", choices=("X", "Y", "Z"), help="TDI channel to transform; repeat for multiple channels. Defaults to X,Y,Z.")
    parser.add_argument("--m1-solar", type=float, default=2.0e5)
    parser.add_argument("--m2-solar", type=float, default=1.0e5)
    parser.add_argument("--chi1", type=float, default=0.42)
    parser.add_argument("--chi2", type=float, default=0.85)
    parser.add_argument("--tc", type=float, default=3.0e7)
    parser.add_argument("--distance-gpc", type=float, default=1.0)
    parser.add_argument("--ucb-frequency", type=float, default=DEFAULT_UCB_FREQUENCY_HZ, help="Galactic-binary carrier frequency in Hz.")
    parser.add_argument("--ucb-costh", type=float, default=-0.783326909627, help="Galactic-binary sky cos(theta), matching response.c.")
    parser.add_argument("--ucb-phi", type=float, default=3.0, help="Galactic-binary ecliptic longitude in radians.")
    parser.add_argument("--ucb-amplitude", type=float, default=None, help="Galactic-binary strain amplitude. Defaults to the response.c 0PN estimate.")
    parser.add_argument("--ucb-cosi", type=float, default=0.0707372016677, help="Galactic-binary cos(inclination).")
    parser.add_argument("--ucb-psi", type=float, default=0.8, help="Galactic-binary polarization angle in radians.")
    parser.add_argument("--ucb-phi0", type=float, default=1.2, help="Galactic-binary initial phase in radians.")
    parser.add_argument("--ucb-fdot", type=float, default=None, help="Galactic-binary first frequency derivative in Hz/s. Defaults to response.c 0PN value.")
    parser.add_argument("--ucb-fddot", type=float, default=None, help="Galactic-binary second frequency derivative in Hz/s^2. Defaults to response.c 0PN value.")
    parser.add_argument("--ucb-m1-solar", type=float, default=0.6, help="Galactic-binary primary mass used for default fdot/fddot/amplitude.")
    parser.add_argument("--ucb-m2-solar", type=float, default=0.7, help="Galactic-binary secondary mass used for default fdot/fddot/amplitude.")
    parser.add_argument("--ucb-distance-kpc", type=float, default=1.0, help="Galactic-binary distance used for default amplitude.")
    parser.add_argument("--ecc-e0", type=float, default=0.0, help="Initial eccentricity for --source-model eccentric.")
    parser.add_argument("--ecc-edot", type=float, default=0.0, help="Phenomenological eccentricity derivative de/dt in 1/s.")
    parser.add_argument("--ecc-delta-f", type=float, default=0.0, help="Initial periastron-advance frequency in Hz. Sidebands are shifted by +/-2 delta_f.")
    parser.add_argument("--ecc-delta-fdot", type=float, default=0.0, help="Phenomenological derivative of delta_f in Hz/s.")
    parser.add_argument("--ecc-periastron-phase0", type=float, default=0.0, help="Initial value of 2Phi in radians for the eccentric model.")
    parser.add_argument("--ecc-nmax", type=int, default=ECCENTRIC_DEFAULT_NMAX, help="Highest eccentric harmonic to include, 1 through 4.")
    parser.add_argument(
        "--intrinsic-backend",
        choices=("local", "phentax"),
        default="local",
        help="Intrinsic 2,2 implementation for --source-model phenomt. Defaults to the local NumPy/Numba port.",
    )
    parser.add_argument(
        "--coefficient-backend",
        choices=("auto", "reference", "native", "python"),
        default="auto",
        help="Coefficient source for --intrinsic-backend local; ignored by Phentax.",
    )
    parser.add_argument(
        "--wdm-blend-endpoint",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_WDM_BLEND_ENDPOINT,
        help="Blend SPA and short-FFT WDM coefficients across the endpoint boundary. Enabled by default.",
    )
    parser.add_argument(
        "--wdm-blend-half-width-layers",
        type=float,
        default=DEFAULT_WDM_BLEND_HALF_WIDTH_LAYERS,
        help="Endpoint blend half-width in WDM frequency layers. Defaults to 4.",
    )
    parser.add_argument(
        "--direct-endpoint-tdi",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate exact TDI delays at endpoint FFT times. Enabled by default.",
    )
    parser.add_argument("--write-prefix", default="phenomt_tdi", help="Prefix for output files. Defaults to phenomt_tdi.")
    parser.add_argument("--no-write", action="store_true", help="Do not write WDM product files.")
    parser.add_argument("--sparse-only", action="store_true", help="Write only sparse active-pixel files, not dense WDM matrices.")
    parser.add_argument("--write-frequency-domain", action="store_true", help="Write model-specific frequency-domain A(f), phi(f) samples.")
    parser.add_argument("--frequency-domain-only", action="store_true", help="Skip WDM packet assembly and produce only the model-specific frequency-domain channel products.")
    parser.add_argument("--timing", action="store_true", help="Print timing for adaptive AP, fast TDI, and fast WDM generation.")
    parser.add_argument("--timing-warmup-runs", type=int, default=1, help="No-write generations to run before measuring --timing. Defaults to 1; use 0 for cold-start timing.")
    parser.add_argument("--timing-repeat-runs", type=int, default=1, help="Warm measured generations to average for --timing. Defaults to 1.")
    args = parser.parse_args()
    if args.timing_warmup_runs < 0:
        raise SystemExit("--timing-warmup-runs must be non-negative")
    if args.timing_repeat_runs < 1:
        raise SystemExit("--timing-repeat-runs must be at least 1")
    if args.source_model == "phenomt" and args.tdi_generation == 2 and args.intrinsic_backend != "local":
        raise SystemExit("22-only TDI-2 requires --intrinsic-backend local")

    if args.source_model == "ucb":
        source = UCBSourceParams(
            frequency_hz=args.ucb_frequency,
            ecliptic_costheta=args.ucb_costh,
            ecliptic_longitude=args.ucb_phi,
            amplitude=args.ucb_amplitude,
            cos_inclination=args.ucb_cosi,
            polarization=args.ucb_psi,
            phi0=args.ucb_phi0,
            fdot=args.ucb_fdot,
            fddot=args.ucb_fddot,
            m1_solar=args.ucb_m1_solar,
            m2_solar=args.ucb_m2_solar,
            distance_kpc=args.ucb_distance_kpc,
        )
    elif args.source_model == "eccentric":
        source = EccentricUCBSourceParams(
            frequency_hz=args.ucb_frequency,
            ecliptic_costheta=args.ucb_costh,
            ecliptic_longitude=args.ucb_phi,
            amplitude=args.ucb_amplitude,
            cos_inclination=args.ucb_cosi,
            polarization=args.ucb_psi,
            phi0=args.ucb_phi0,
            fdot=args.ucb_fdot,
            fddot=args.ucb_fddot,
            m1_solar=args.ucb_m1_solar,
            m2_solar=args.ucb_m2_solar,
            distance_kpc=args.ucb_distance_kpc,
            eccentricity0=args.ecc_e0,
            edot=args.ecc_edot,
            delta_f0=args.ecc_delta_f,
            delta_fdot=args.ecc_delta_fdot,
            periastron_phase0=args.ecc_periastron_phase0,
            nmax=args.ecc_nmax,
        )
    else:
        source = SourceParams(
            m1_solar=args.m1_solar,
            m2_solar=args.m2_solar,
            chi1=args.chi1,
            chi2=args.chi2,
            tc=args.tc,
            distance_gpc=args.distance_gpc,
        )
    channels = tuple(args.channel) if args.channel else ("X", "Y", "Z")

    def run_once() -> TDIWDMResult:
        if args.source_model == "ucb":
            return generate_ucb_tdi_wdm(source=source, channels=channels,
                                        compute_wdm=not args.frequency_domain_only,
                                        tdi_generation=args.tdi_generation)
        if args.source_model == "eccentric":
            return generate_eccentric_ucb_tdi_wdm(
                source=source, channels=channels,
                compute_wdm=not args.frequency_domain_only,
                tdi_generation=args.tdi_generation,
            )
        return generate_tdi_wdm(
            source=source,
            channels=channels,
            coefficient_backend=args.coefficient_backend,
            intrinsic_backend=args.intrinsic_backend,
            compute_wdm=not args.frequency_domain_only,
            blend_endpoint=args.wdm_blend_endpoint,
            blend_half_width_layers=args.wdm_blend_half_width_layers,
            direct_endpoint_tdi=args.direct_endpoint_tdi,
            tdi_generation=args.tdi_generation,
        )

    warmup_runs = args.timing_warmup_runs if args.timing else 0
    for _ in range(warmup_runs):
        run_once()

    repeat_runs = args.timing_repeat_runs if args.timing else 1
    timing_samples: list[dict[str, float]] = []
    elapsed_samples: list[float] = []
    result: TDIWDMResult | None = None
    for _ in range(repeat_runs):
        start = time.perf_counter()
        result = run_once()
        elapsed_samples.append(time.perf_counter() - start)
        timing_samples.append(result.timings)
    if result is None:
        raise RuntimeError("no waveform generation was run")

    timing_keys = ("adaptive_ap", "fast_tdi", "fast_frequency_domain", "fast_wdm_packets", "fast_wdm")
    timing_mean = {
        key: sum(sample[key] for sample in timing_samples) / float(repeat_runs)
        for key in timing_keys
    }
    elapsed_mean = sum(elapsed_samples) / float(repeat_runs)

    print(f"samples {result.tdi.time.size}")
    if args.source_model == "phenomt" and args.tdi_generation == 2:
        print("intrinsic_backend local-THM modes 22pair wdm_method partitioned-fft")
    elif args.source_model == "phenomt" and result.intrinsic.model_time is not None:
        model_backend = getattr(result.intrinsic.model, "backend_name", "local")
        coefficient_source = getattr(result.intrinsic.model, "coefficient_backend", "jax")
        print(f"intrinsic_backend {model_backend} coefficient_backend {coefficient_source}")
        print(
            f"intrinsic_samples {result.intrinsic.model_time.size} "
            f"exact_response_samples {result.intrinsic.exact_response_samples} "
            f"response_switch_time {result.intrinsic.response_switch_detector_time:.15e}"
        )
        print(
            f"wdm_endpoint {'blend' if args.wdm_blend_endpoint else 'hard'} "
            f"half_width_layers {args.wdm_blend_half_width_layers:.6g} "
            f"tdi {'exact_delay' if args.direct_endpoint_tdi else 'sparse_ap'}"
        )
    print(f"wdm_grid nt {result.shape.nt} nf {result.shape.nf} dt {result.shape.dt:.15e} Tobs {result.shape.Tobs:.15e}")
    written_files: list[str] = []
    write_frequency = args.write_frequency_domain or args.frequency_domain_only
    for ch, channel in result.channels.items():
        if args.frequency_domain_only:
            print(f"channel {ch} frequency_samples {channel.freq.size} active_pixels skipped")
        else:
            print(f"channel {ch} frequency_samples {channel.freq.size} active_pixels {channel.values.size}")
        if not args.no_write:
            if write_frequency:
                freq_path = f"{args.write_prefix}_{ch}_freq_ap.dat"
                write_frequency_domain(freq_path, channel)
                written_files.append(freq_path)
            if not args.frequency_domain_only:
                track_path = f"{args.write_prefix}_{ch}_track_pixels.dat"
                write_track_pixels(track_path, channel)
                written_files.append(track_path)
                if not args.sparse_only:
                    dense_path = f"{args.write_prefix}_{ch}_wtranfast.dat"
                    write_wdm_matrix(dense_path, channel.dense(result.shape))
                    written_files.append(dense_path)
    if written_files:
        print("output_files")
        for path in written_files:
            print(path)
    else:
        print("output_files none")
    if args.timing:
        timing_std = {
            key: math.sqrt(sum((sample[key] - timing_mean[key]) ** 2 for sample in timing_samples) / float(repeat_runs))
            for key in timing_keys
        }
        elapsed_std = math.sqrt(sum((value - elapsed_mean) ** 2 for value in elapsed_samples) / float(repeat_runs))
        print("timing_seconds")
        print(f"warmup_runs {warmup_runs}")
        print(f"measured_runs {repeat_runs}")
        print(f"adaptive_ap {timing_mean['adaptive_ap']:.6f}")
        print(f"fast_tdi {timing_mean['fast_tdi']:.6f}")
        print(f"fast_frequency_domain {timing_mean['fast_frequency_domain']:.6f}")
        print(f"fast_wdm_packets {timing_mean['fast_wdm_packets']:.6f}")
        print(f"fast_wdm {timing_mean['fast_wdm']:.6f}")
        if repeat_runs > 1:
            print("timing_std_seconds")
            print(f"adaptive_ap {timing_std['adaptive_ap']:.6f}")
            print(f"fast_tdi {timing_std['fast_tdi']:.6f}")
            print(f"fast_frequency_domain {timing_std['fast_frequency_domain']:.6f}")
            print(f"fast_wdm_packets {timing_std['fast_wdm_packets']:.6f}")
            print(f"fast_wdm {timing_std['fast_wdm']:.6f}")
            print(f"elapsed_std_seconds {elapsed_std:.6f}")
    print(f"elapsed_seconds {elapsed_mean:.6f}")


if __name__ == "__main__":
    main()
