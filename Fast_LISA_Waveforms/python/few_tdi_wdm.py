#!/usr/bin/env python3
# FEW-driven LISA response and WDM implementation by Neil Cornish.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Sparse single- and multi-mode FEW TDI/WDM generator with direct checks.

The production path never asks FEW for a uniformly sampled waveform.  It
integrates the orbit adaptively, evaluates complex mode amplitudes on the
sparse trajectory, and queries FEW's DOP853 continuous solution only at the
times selected by the TDI planner.  A selected ``(ell,m,k,n)`` mode and its
``(-m,-k,-n)`` partner are folded into one carrier track.

The preserved first validation stage uses one selectable folded mode pair and
a two-sided observation taper.  The multi-mode path asks FEW to select a
frozen carrier catalog, plans one union SSB-output grid, and evaluates all
mode phases from one shared trajectory.  Entries with exactly the same
canonical ``(m,k,n)`` phase track are folded by summing their projected
polarization coefficients before TDI; nearby tracks remain separate to avoid
beating zeros and phase jumps.  Monotone early branches use the
numerical SPA.  A compact heterodyned FFT replaces only those WDM pixels whose
Meyer support straddles a resolvable frequency turnover.  A smooth partition
of unity sends the physical endpoint of
every selected mode through one summed, full-band plunge FFT.  By default,
radial harmonics are first grouped by physical ``(ell,|m|)`` mode and the FEW
hard endpoint is completed by a C1 bridge to Schwarzschild or signed-m Kerr
QNM decay.  The
endpoint partition is anchored in SSB output time after accounting for every
TDI delay.  The driver
checks

* interpolation of the sparse FEW complex amplitudes;
* sparse TDI-on-the-fly against direct full-cadence TDI;
* sparse WDM packets against a full frequency-domain Meyer transform; and
* the common all-mode plunge packet separately from the early transform.

The whole-track narrow-band FFT remains available as an independent check of
the SPA and as the accurate default for the preserved one-mode validator,
which intentionally has no separate physical endpoint packet.
The optional ``block-fft`` reference instead replaces the early SPA with an
ungrouped, bandwidth-limited partition of heterodyned FFTs, one plan per exact
carrier, while retaining the common summed endpoint FFT.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

from few.amplitude.ampinterp2d import AmpInterpKerrEccEq
from few.amplitude.romannet import RomanAmplitude
from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode.flux import KerrEccEqFlux, SchwarzEccFlux
from few.utils.constants import Gpc, MRSUN_SI, MTSUN_SI, YRSID_SI
from few.utils.modeselector import ModeSelector
from few.utils.ylm import GetYlms

from WDG import tw_freq
from phenomt_tdi_wdm import (
    AP_DTM_MAX,
    AP_FREQ_REL_CURVATURE_TOL,
    AP_FREQ_STEP_REL_TOL,
    AP_PHASE_CURVATURE_TOL,
    AP_TRANSFER_PHASE_STEP,
    FSTAR,
    PI,
    TDIGrid,
    WDMChannel,
    WDMShape,
    build_constellation_splines,
    combine_sparse_wdm,
    extract_ap,
    apply_short_fft_threshold,
    make_ap_spline,
    merge_pixel_plans,
    nonuniform_derivative_array,
    phitilde,
    sky_vectors,
    spacecraft0_reference_time,
    wdm_pixels,
    wdm_channel_from_dense,
    wdm_track,
)
from wdm_chirplet_lookup import (
    CHIRPLET_LOOKUP_FREQUENCY_STEP,
    CHIRPLET_LOOKUP_RATE_MAX,
    CHIRPLET_LOOKUP_RATE_STEP,
    build_wdm_chirplet_lookup,
)
from tdi2_response import complex_tdi2


Mode = tuple[int, int, int, int]
CHANNEL_TRIPLES = {"X": (0, 1, 2), "Y": (1, 2, 0), "Z": (2, 0, 1)}
TDI_ENDPOINT_DELAY_MARGIN_SECONDS = 600.0
EARLY_BLOCK_FFT_ROLL_SECONDS = 1.0e5
EARLY_BLOCK_FFT_MARGIN_CYCLES = 3.0
EARLY_BLOCK_FFT_NYQUIST_GUARD_LAYERS = 2

# Fundamental n=0 gravitational Schwarzschild QNMs, in units M omega.  The
# selected RomanAmplitude catalog for the reference source stops at ell=5.
# The higher-ell entries keep --modes all self contained; beyond the table the
# eikonal limit is more accurate than the phenomenological endpoint itself.
SCHWARZSCHILD_QNM_FUNDAMENTAL: dict[int, complex] = {
    2: complex(0.3736716844, -0.0889623157),
    3: complex(0.5994432884, -0.0927030479),
    4: complex(0.8091783775, -0.0941639610),
    5: complex(1.0122953121, -0.0948705161),
    6: complex(1.2120130000, -0.0952659000),
    7: complex(1.4097350000, -0.0955106000),
    8: complex(1.6061940000, -0.0956717000),
    9: complex(1.8017950000, -0.0957830000),
    10: complex(1.9967880000, -0.0958640000),
}


def schwarzschild_qnm_fundamental(ell: int) -> complex:
    """Return the fundamental gravitational Schwarzschild QNM in M omega."""

    if ell < 2:
        raise ValueError("gravitational QNM multipoles require ell >= 2")
    if ell in SCHWARZSCHILD_QNM_FUNDAMENTAL:
        return SCHWARZSCHILD_QNM_FUNDAMENTAL[ell]
    photon_orbit_scale = 1.0 / (3.0 * math.sqrt(3.0))
    return complex((ell + 0.5) * photon_orbit_scale, -0.5 * photon_orbit_scale)


def kerr_qnm_fundamental(ell: int, m: int, spin: float) -> complex:
    """Return the fundamental gravitational Kerr QNM in M omega.

    FEW represents retrograde equatorial motion with signed spin and xI=+1.
    The qnm package instead uses non-negative spin and signed m, so reversing
    the spin is equivalent to reversing the QNM azimuthal index.
    """

    if abs(spin) < 1.0e-14:
        return schwarzschild_qnm_fundamental(ell)
    if abs(spin) >= 1.0:
        raise ValueError("Kerr QNMs require |spin| < 1")
    try:
        import qnm
    except ImportError as exc:
        raise RuntimeError(
            "nonzero-spin endpoints require the qnm package; install it with "
            "python -m pip install qnm"
        ) from exc
    effective_m = m if spin > 0.0 else -m
    sequence = qnm.modes_cache(s=-2, l=ell, m=effective_m, n=0)
    omega, _, _ = sequence(a=abs(spin))
    return complex(omega)


@dataclass(frozen=True)
class FEWRingdownGroup:
    """One physical (ell,m) endpoint after summing FEW radial harmonics."""

    ell: int
    m: int
    right_value: complex
    right_derivative: complex
    left_value: complex
    left_derivative: complex
    right_omega: float
    right_damping: float
    left_omega: float
    left_damping: float
    transition: float
    fade_start: float
    fade_duration: float

    @property
    def stop(self) -> float:
        return self.fade_start + self.fade_duration

    @property
    def maximum_omega(self) -> float:
        return max(self.right_omega, self.left_omega)

    @property
    def maximum_damping(self) -> float:
        return max(self.right_damping, self.left_damping)

    @staticmethod
    def _branch(
        elapsed: np.ndarray,
        value: complex,
        derivative: complex,
        target_rate: complex,
        transition: float,
    ) -> np.ndarray:
        """C1 bridge from the FEW endpoint derivative to one QNM rate."""

        if transition <= 0.0:
            return value * np.exp(target_rate * elapsed)
        correction = derivative - target_rate * value
        envelope = value + correction * transition * (
            1.0 - np.exp(-elapsed / transition)
        )
        return np.exp(target_rate * elapsed) * envelope

    def value(self, elapsed: np.ndarray) -> complex | np.ndarray:
        values = np.asarray(elapsed, dtype=np.float64)
        right = self._branch(
            values,
            self.right_value,
            self.right_derivative,
            complex(-self.right_damping, -self.right_omega),
            self.transition,
        )
        left = self._branch(
            values,
            self.left_value,
            self.left_derivative,
            complex(-self.left_damping, self.left_omega),
            self.transition,
        )
        out = right + left
        if self.fade_duration > 0.0:
            fade = np.ones_like(values)
            active = values > self.fade_start
            fade[values >= self.stop] = 0.0
            interior = active & (values < self.stop)
            x = (values[interior] - self.fade_start) / self.fade_duration
            fade[interior] = 0.5 * (1.0 + np.cos(PI * x))
            out *= fade
        return out


@dataclass(frozen=True)
class FEWTurnoverRegion:
    """Compact SSB-output interval assigned to one local turnover FFT."""

    start: float
    stop: float
    center: float
    frequency: float
    prominence: float


@dataclass(frozen=True)
class FEWEarlyBlockFFTDiagnostic:
    """One bandwidth-limited block in the ungrouped early FFT reference."""

    carrier_index: int
    block_index: int
    tile_lo: int
    tile_hi: int
    support_lo: int
    support_hi: int
    nonzero_start: float
    nonzero_stop: float
    frequency_min: float
    frequency_max: float
    frequency_pad: float
    heterodyne_layer: int
    sample_dt: float
    sample_count: int
    packet_time_pixels: int


def parse_mode(value: str) -> Mode:
    fields = value.replace("(", "").replace(")", "").split(",")
    if len(fields) != 4:
        raise argparse.ArgumentTypeError("mode must have the form ell,m,k,n")
    try:
        result = tuple(int(field.strip()) for field in fields)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("mode indices must be integers") from exc
    if result[1] < 0:
        raise argparse.ArgumentTypeError("select the non-negative-m member of a folded pair")
    return result  # type: ignore[return-value]


@dataclass(frozen=True)
class FEWSourceParams:
    mass_solar: float = 1.0e6
    mu_solar: float = 50.0
    spin: float = 0.0
    p0: float = 12.510272236947417
    eccentricity0: float = 0.4
    xI0: float = 1.0
    distance_gpc: float = 1.0
    viewing_theta: float = math.pi / 3.0
    viewing_phi: float = math.pi / 4.0
    ecliptic_costheta: float = -0.3
    ecliptic_longitude: float = 1.1
    polarization: float = 0.4
    Phi_phi0: float = 0.0
    Phi_theta0: float = 0.0
    Phi_r0: float = 0.0


@dataclass
class FEWReusableModel:
    """Heavy FEW backend objects shared by sequential source evaluations."""

    background: str
    trajectory: EMRIInspiral
    amplitude_model: RomanAmplitude | AmpInterpKerrEccEq
    selector: ModeSelector
    source_evaluations: int = 0


_FEW_MODEL_CACHE: dict[str, FEWReusableModel] = {}


def get_few_reusable_model(
    source: FEWSourceParams,
    diagnostics: dict[str, float] | None = None,
) -> FEWReusableModel:
    """Return one process-local FEW backend, constructing it only once."""

    is_kerr = abs(source.spin) > 1.0e-14
    key = "kerr" if is_kerr else "schwarzschild"
    start = time.perf_counter()
    cache_hit = key in _FEW_MODEL_CACHE
    if not cache_hit:
        trajectory = EMRIInspiral(
            func=KerrEccEqFlux if is_kerr else SchwarzEccFlux
        )
        amplitude_model = (
            AmpInterpKerrEccEq()
            if is_kerr
            else RomanAmplitude(buffer_length=4096)
        )
        _FEW_MODEL_CACHE[key] = FEWReusableModel(
            key,
            trajectory,
            amplitude_model,
            ModeSelector(amplitude_model),
        )
    runtime = _FEW_MODEL_CACHE[key]
    elapsed = time.perf_counter() - start
    if diagnostics is not None:
        diagnostics.update(
            {
                "cache_hit": float(cache_hit),
                "source_call_warm": float(runtime.source_evaluations > 0),
                "initialization_seconds": elapsed,
            }
        )
    return runtime


def clear_few_model_cache() -> None:
    """Drop reusable FEW objects, primarily for cold-start benchmarks."""

    _FEW_MODEL_CACHE.clear()


@dataclass
class FEWModeCarrier:
    source: FEWSourceParams
    mode: Mode
    trajectory: EMRIInspiral
    amplitude_model: RomanAmplitude | AmpInterpKerrEccEq
    sparse_time: np.ndarray
    sparse_p: np.ndarray
    sparse_e: np.ndarray
    sparse_x: np.ndarray
    plus_coeff_sparse: np.ndarray
    cross_coeff_sparse: np.ndarray
    plus_coeff_spline: CubicSpline
    cross_coeff_spline: CubicSpline
    segment_start: float
    segment_stop: float
    phase_sign: float = 1.0

    @property
    def duration(self) -> float:
        return self.segment_stop - self.segment_start

    def _global_times(self, local_times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(local_times, dtype=np.float64)
        global_times = values + self.segment_start
        valid = (global_times >= self.sparse_time[0]) & (global_times <= self.sparse_time[-1])
        return global_times, valid

    def phase(self, local_times: np.ndarray) -> np.ndarray:
        global_times, _ = self._global_times(local_times)
        clipped = np.clip(global_times, self.sparse_time[0], self.sparse_time[-1])
        state = self.trajectory.inspiral_generator.eval_integrator_spline(clipped)
        _, m, k, n = self.mode
        # Delayed samples outside the selected observation have zero
        # coefficient, so a clipped endpoint phase is sufficient there.
        return self.phase_sign * (
            m * state[:, 3] + k * state[:, 4] + n * state[:, 5]
        )

    def frequency(self, local_times: np.ndarray) -> np.ndarray:
        global_times, valid = self._global_times(local_times)
        clipped = np.clip(global_times, self.sparse_time[0], self.sparse_time[-1])
        first = self.trajectory.inspiral_generator.eval_integrator_derivative_spline(
            clipped, order=1
        )
        _, m, k, n = self.mode
        out = self.phase_sign * (
            m * first[:, 3] + k * first[:, 4] + n * first[:, 5]
        ) / (2.0 * PI)
        out[~valid] = 0.0
        return out

    def polarization_coefficients(
        self, local_times: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        global_times, valid = self._global_times(local_times)
        clipped = np.clip(global_times, self.sparse_time[0], self.sparse_time[-1])
        plus = np.asarray(self.plus_coeff_spline(clipped), dtype=np.complex128)
        cross = np.asarray(self.cross_coeff_spline(clipped), dtype=np.complex128)
        plus[~valid] = 0.0
        cross[~valid] = 0.0
        return plus, cross

    def polarizations(
        self, local_times: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        phase = self.phase(local_times)
        plus_coeff, cross_coeff = self.polarization_coefficients(local_times)
        c = np.cos(phase)
        s = np.sin(phase)
        hp = plus_coeff.real * c - plus_coeff.imag * s
        hc = cross_coeff.real * c - cross_coeff.imag * s
        hpf = -plus_coeff.real * s - plus_coeff.imag * c
        hcf = -cross_coeff.real * s - cross_coeff.imag * c
        return hp, hc, hpf, hcf


@dataclass
class FEWModeCollection:
    """Selected FEW carriers sharing one trajectory and coefficient spline."""

    source: FEWSourceParams
    # ``modes`` are the exact canonical phase carriers used by TDI/WDM.
    # ``selected_modes`` preserves the uncombined FEW catalog needed by the
    # direct algebra check and the physical (ell,|m|) QNM grouping.
    modes: tuple[Mode, ...]
    selected_modes: tuple[Mode, ...]
    carrier_members: tuple[tuple[Mode, ...], ...]
    carriers: tuple[FEWModeCarrier, ...]
    trajectory: EMRIInspiral
    amplitude_model: RomanAmplitude | AmpInterpKerrEccEq
    sparse_time: np.ndarray
    sparse_p: np.ndarray
    sparse_e: np.ndarray
    sparse_x: np.ndarray
    plus_coeff_spline: CubicSpline
    cross_coeff_spline: CubicSpline
    mode_vectors: np.ndarray
    raw_mode_vectors: np.ndarray
    right_coeff_spline: CubicSpline
    left_coeff_spline: CubicSpline
    segment_start: float
    segment_stop: float
    plunge_time: float
    endpoint_model: str
    ringdown_groups: tuple[FEWRingdownGroup, ...]
    ringdown_stop: float

    @property
    def duration(self) -> float:
        return self.segment_stop - self.segment_start

    @property
    def size(self) -> int:
        return len(self.modes)

    @property
    def mass_seconds(self) -> float:
        return self.source.mass_solar * MTSUN_SI

    @property
    def maximum_qnm_frequency(self) -> float:
        if not self.ringdown_groups:
            return 0.0
        return max(group.maximum_omega for group in self.ringdown_groups) / (
            2.0 * PI
        )

    def _global_times(self, local_times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(local_times, dtype=np.float64)
        global_times = values + self.segment_start
        valid = (global_times >= self.sparse_time[0]) & (global_times <= self.sparse_time[-1])
        return global_times, valid

    def phase_matrix(self, local_times: np.ndarray) -> np.ndarray:
        global_times, _ = self._global_times(local_times)
        clipped = np.clip(global_times, self.sparse_time[0], self.sparse_time[-1])
        state = self.trajectory.inspiral_generator.eval_integrator_spline(clipped)
        return np.asarray(state[:, 3:6] @ self.mode_vectors.T, dtype=np.float64)

    def frequency_matrix(self, local_times: np.ndarray) -> np.ndarray:
        global_times, valid = self._global_times(local_times)
        clipped = np.clip(global_times, self.sparse_time[0], self.sparse_time[-1])
        first = self.trajectory.inspiral_generator.eval_integrator_derivative_spline(
            clipped, order=1
        )
        out = np.asarray(first[:, 3:6] @ self.mode_vectors.T / (2.0 * PI), dtype=np.float64)
        out[~valid, :] = 0.0
        return out

    def polarization_components(
        self, local_times: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        global_times, valid = self._global_times(local_times)
        clipped = np.clip(global_times, self.sparse_time[0], self.sparse_time[-1])
        plus = np.asarray(self.plus_coeff_spline(clipped), dtype=np.complex128)
        cross = np.asarray(self.cross_coeff_spline(clipped), dtype=np.complex128)
        plus[~valid, :] = 0.0
        cross[~valid, :] = 0.0
        phase = self.phase_matrix(local_times)
        c = np.cos(phase)
        s = np.sin(phase)
        hp = plus.real * c - plus.imag * s
        hc = cross.real * c - cross.imag * s
        hpf = -plus.real * s - plus.imag * c
        hcf = -cross.real * s - cross.imag * c
        return hp, hc, hpf, hcf

    def polarization_values(
        self, local_times: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return per-carrier hplus and hcross without quadrature arrays."""

        global_times, valid = self._global_times(local_times)
        clipped = np.clip(global_times, self.sparse_time[0], self.sparse_time[-1])
        plus = np.asarray(self.plus_coeff_spline(clipped), dtype=np.complex128)
        cross = np.asarray(self.cross_coeff_spline(clipped), dtype=np.complex128)
        plus[~valid, :] = 0.0
        cross[~valid, :] = 0.0
        phase = self.phase_matrix(local_times)
        c = np.cos(phase)
        s = np.sin(phase)
        return (
            plus.real * c - plus.imag * s,
            cross.real * c - cross.imag * s,
        )

    def summed_complex_strain(self, local_times: np.ndarray) -> np.ndarray:
        """Return hplus - i hcross, including the grouped QNM continuation."""

        values = np.asarray(local_times, dtype=np.float64)
        hp, hc = self.polarization_values(values)
        strain = np.sum(hp, axis=1) - 1j * np.sum(hc, axis=1)
        if self.endpoint_model != "qnm" or not self.ringdown_groups:
            return strain

        elapsed = values - self.plunge_time
        late = (elapsed > 0.0) & (values < self.ringdown_stop)
        if np.any(late):
            continuation = np.zeros(np.count_nonzero(late), dtype=np.complex128)
            for group in self.ringdown_groups:
                continuation += group.value(elapsed[late])
            strain[late] = continuation * np.exp(
                2j * self.source.polarization
            )
        strain[values >= self.ringdown_stop] = 0.0
        return strain

    def summed_polarizations(
        self, local_times: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        strain = self.summed_complex_strain(local_times)
        return strain.real, -strain.imag


@dataclass
class FEWValidationResult:
    source: FEWSourceParams
    mode: Mode
    shape: WDMShape
    carrier: FEWModeCarrier
    tdi: TDIGrid
    fast_wdm: WDMChannel
    direct_wdm: np.ndarray
    direct_time: np.ndarray
    direct_strain: np.ndarray
    reconstructed_strain: np.ndarray
    fast_method: str
    metrics: dict[str, float]
    timings: dict[str, float]


@dataclass
class FEWMultiModeValidationResult:
    source: FEWSourceParams
    modes: tuple[Mode, ...]
    shape: WDMShape
    collection: FEWModeCollection
    tdi: tuple[TDIGrid, ...]
    fast_wdm: WDMChannel
    direct_wdm: np.ndarray
    direct_time: np.ndarray
    direct_strain: np.ndarray
    reconstructed_strain: np.ndarray
    mode_methods: tuple[str, ...]
    metrics: dict[str, float]
    timings: dict[str, float]
    early_block_fft_diagnostics: tuple[FEWEarlyBlockFFTDiagnostic, ...] = ()
    global_fft_wdm: WDMChannel | None = None


def build_few_mode_carrier(
    source: FEWSourceParams,
    mode: Mode,
    shape: WDMShape,
    endpoint_margin: float = 1000.0,
    runtime_diagnostics: dict[str, float] | None = None,
) -> FEWModeCarrier:
    if source.xI0 != 1.0:
        raise ValueError(
            "use xI0=+1 and the sign of spin to select prograde or retrograde "
            "equatorial Kerr motion"
        )
    runtime = get_few_reusable_model(source, runtime_diagnostics)
    trajectory = runtime.trajectory
    years = max(1.0, (shape.Tobs + 2.0 * endpoint_margin) / YRSID_SI)
    values = trajectory(
        source.mass_solar,
        source.mu_solar,
        source.spin,
        source.p0,
        source.eccentricity0,
        source.xI0,
        Phi_phi0=source.Phi_phi0,
        Phi_theta0=source.Phi_theta0,
        Phi_r0=source.Phi_r0,
        T=years,
        dt=10.0,
        upsample=False,
    )
    sparse_time, p, e, x = (np.asarray(item) for item in values[:4])
    if sparse_time[-1] < shape.Tobs + 2.0 * endpoint_margin:
        endpoint_margin = max(0.0, 0.25 * (sparse_time[-1] - shape.Tobs))
    segment_stop = float(sparse_time[-1] - endpoint_margin)
    segment_start = segment_stop - shape.Tobs
    if segment_start < endpoint_margin:
        raise ValueError("FEW trajectory is too short for the requested validation segment and delay margins")

    amplitude_model = runtime.amplitude_model
    amplitudes = amplitude_model(
        source.spin, p, e, x, specific_modes=[mode]
    )[mode]
    ell, m, _, _ = mode
    ylms = GetYlms(include_minus_m=True)(
        np.array([ell], dtype=int),
        np.array([m], dtype=int),
        source.viewing_theta,
        source.viewing_phi,
    )
    distance_scale = source.mu_solar * MRSUN_SI / (source.distance_gpc * Gpc)
    right = ylms[0] * amplitudes * distance_scale
    if m > 0:
        left = ((-1) ** ell) * ylms[1] * np.conj(amplitudes) * distance_scale
    else:
        left = np.zeros_like(right)

    # If w = right*exp(-i Psi) + left*exp(+i Psi) = hplus-i hcross,
    # these coefficients give hplus=Re(P exp(i Psi)) and
    # hcross=Re(C exp(i Psi)).  The +/- pair is therefore one carrier track.
    plus_coeff = np.conj(right) + left
    cross_coeff = 1j * (left - np.conj(right))
    c2 = math.cos(2.0 * source.polarization)
    s2 = math.sin(2.0 * source.polarization)
    plus_rotated = c2 * plus_coeff + s2 * cross_coeff
    cross_rotated = c2 * cross_coeff - s2 * plus_coeff

    result = FEWModeCarrier(
        source,
        mode,
        trajectory,
        amplitude_model,
        sparse_time,
        p,
        e,
        x,
        plus_rotated,
        cross_rotated,
        CubicSpline(sparse_time, plus_rotated),
        CubicSpline(sparse_time, cross_rotated),
        segment_start,
        segment_stop,
    )
    runtime.source_evaluations += 1
    return result


def build_few_mode_collection(
    source: FEWSourceParams,
    shape: WDMShape,
    *,
    mode_selection: str | Sequence[Mode] = "threshold",
    mode_selection_threshold: float = 1.0e-3,
    endpoint_padding: float = 1000.0,
    endpoint_model: str = "qnm",
    qnm_transition_m: float = 10.0,
    qnm_efolds: float = 8.0,
    qnm_fade_efolds: float = 2.0,
    runtime_diagnostics: dict[str, float] | None = None,
) -> FEWModeCollection:
    """Build selected FEW carriers with one trajectory and amplitude call.

    The default endpoint groups radial harmonics by ``(ell,m)`` and continues
    each physical mode into the fundamental Schwarzschild or signed-m Kerr
    QNM.  ``hard`` keeps FEW's zero-padded endpoint as a diagnostic reference.
    """

    if source.xI0 != 1.0:
        raise ValueError(
            "use xI0=+1 and the sign of spin to select prograde or retrograde "
            "equatorial Kerr motion"
        )
    if endpoint_model not in {"hard", "qnm"}:
        raise ValueError("endpoint_model must be 'hard' or 'qnm'")
    if qnm_transition_m < 0.0 or qnm_efolds < 0.0 or qnm_fade_efolds < 0.0:
        raise ValueError("QNM transition and decay controls must be non-negative")
    runtime = get_few_reusable_model(source, runtime_diagnostics)
    trajectory = runtime.trajectory
    years = max(1.0, (shape.Tobs + endpoint_padding) / YRSID_SI)
    values = trajectory(
        source.mass_solar,
        source.mu_solar,
        source.spin,
        source.p0,
        source.eccentricity0,
        source.xI0,
        Phi_phi0=source.Phi_phi0,
        Phi_theta0=source.Phi_theta0,
        Phi_r0=source.Phi_r0,
        T=years,
        dt=10.0,
        upsample=False,
    )
    sparse_time, p, e, x = (np.asarray(item) for item in values[:4])
    segment_stop = float(sparse_time[-1] + endpoint_padding)
    segment_start = segment_stop - shape.Tobs
    if segment_start <= sparse_time[0]:
        raise ValueError(
            "FEW trajectory is too short for the requested observation and endpoint padding"
        )

    amplitude_model = runtime.amplitude_model
    selector = runtime.selector
    requested: str | list[Mode]
    if isinstance(mode_selection, str):
        requested = mode_selection
    else:
        requested = list(mode_selection)
    selected = selector(
        sparse_time,
        source.spin,
        p,
        e,
        x,
        source.viewing_theta,
        source.viewing_phi,
        mode_selection=requested,
        include_minus_mkn=True,
        mode_selection_threshold=mode_selection_threshold,
    )
    teuk_modes, ylms, l_arr, m_arr, k_arr, n_arr = (
        np.asarray(item) for item in selected
    )
    selected_modes = tuple(
        (int(ell), int(m), int(k), int(n))
        for ell, m, k, n in zip(l_arr, m_arr, k_arr, n_arr)
    )
    if not selected_modes:
        raise RuntimeError("FEW mode selection returned no carriers")

    distance_scale = source.mu_solar * MRSUN_SI / (source.distance_gpc * Gpc)
    raw_mode_vectors = np.column_stack((m_arr, k_arr, n_arr)).astype(np.float64)
    mode_vectors = raw_mode_vectors.copy()
    midpoint = np.array([0.5 * (segment_start + min(sparse_time[-1], segment_stop))])
    first = trajectory.inspiral_generator.eval_integrator_derivative_spline(
        midpoint, order=1
    )[0, 3:6]
    signed_frequency = mode_vectors @ first
    phase_signs = np.where(signed_frequency < 0.0, -1.0, 1.0)
    mode_vectors *= phase_signs[:, None]

    plus_columns: list[np.ndarray] = []
    cross_columns: list[np.ndarray] = []
    right_columns: list[np.ndarray] = []
    left_columns: list[np.ndarray] = []
    carriers: list[FEWModeCarrier] = []
    c2 = math.cos(2.0 * source.polarization)
    s2 = math.sin(2.0 * source.polarization)
    for index, mode in enumerate(selected_modes):
        ell, m, _, _ = mode
        right = ylms[index] * teuk_modes[:, index] * distance_scale
        if m > 0:
            left = (
                ((-1) ** ell)
                * ylms[len(selected_modes) + index]
                * np.conj(teuk_modes[:, index])
                * distance_scale
            )
        else:
            left = np.zeros_like(right)
        right_columns.append(right)
        left_columns.append(left)
        plus_unrotated = np.conj(right) + left
        cross_unrotated = 1j * (left - np.conj(right))
        plus = c2 * plus_unrotated + s2 * cross_unrotated
        cross = c2 * cross_unrotated - s2 * plus_unrotated
        # Canonicalize every real carrier to increasing positive phase.  The
        # conjugated coefficients leave hplus and hcross exactly unchanged.
        if phase_signs[index] < 0.0:
            plus = np.conj(plus)
            cross = np.conj(cross)
        plus_columns.append(plus)
        cross_columns.append(cross)

    raw_plus_matrix = np.column_stack(plus_columns)
    raw_cross_matrix = np.column_stack(cross_columns)
    right_matrix = np.column_stack(right_columns)
    left_matrix = np.column_stack(left_columns)

    # The orbital phase depends only on the canonicalized (m,k,n) vector.
    # Different ell values, and canonical positive/negative-frequency
    # partners, can therefore be combined exactly after angular projection
    # and before TDI.  This does not create a beating amplitude or phase.
    exact_groups: dict[tuple[int, int, int], list[int]] = {}
    for index, vector in enumerate(mode_vectors):
        key = tuple(int(round(value)) for value in vector)
        exact_groups.setdefault(key, []).append(index)
    group_indices = list(exact_groups.values())
    modes = tuple(selected_modes[indices[0]] for indices in group_indices)
    carrier_members = tuple(
        tuple(selected_modes[index] for index in indices)
        for indices in group_indices
    )
    grouped_mode_vectors = np.asarray(
        [mode_vectors[indices[0]] for indices in group_indices],
        dtype=np.float64,
    )
    plus_matrix = np.column_stack(
        [np.sum(raw_plus_matrix[:, indices], axis=1) for indices in group_indices]
    )
    cross_matrix = np.column_stack(
        [np.sum(raw_cross_matrix[:, indices], axis=1) for indices in group_indices]
    )
    for group_index, (mode, indices) in enumerate(zip(modes, group_indices)):
        representative = indices[0]
        carriers.append(
            FEWModeCarrier(
                source,
                mode,
                trajectory,
                amplitude_model,
                sparse_time,
                p,
                e,
                x,
                plus_matrix[:, group_index],
                cross_matrix[:, group_index],
                CubicSpline(sparse_time, plus_matrix[:, group_index]),
                CubicSpline(sparse_time, cross_matrix[:, group_index]),
                segment_start,
                segment_stop,
                float(phase_signs[representative]),
            )
        )

    right_spline = CubicSpline(sparse_time, right_matrix, axis=0)
    left_spline = CubicSpline(sparse_time, left_matrix, axis=0)
    ringdown_groups: list[FEWRingdownGroup] = []
    plunge_global = float(sparse_time[-1])
    mass_seconds = source.mass_solar * MTSUN_SI
    if endpoint_model == "qnm":
        state = trajectory.inspiral_generator.eval_integrator_spline(
            np.array([plunge_global])
        )[0]
        first = trajectory.inspiral_generator.eval_integrator_derivative_spline(
            np.array([plunge_global]), order=1
        )[0]
        phase = raw_mode_vectors @ state[3:6]
        phase_dot = raw_mode_vectors @ first[3:6]
        right_value = right_spline(plunge_global)
        right_derivative = right_spline(plunge_global, 1)
        left_value = left_spline(plunge_global)
        left_derivative = left_spline(plunge_global, 1)
        right_branch = right_value * np.exp(-1j * phase)
        right_branch_dot = (
            right_derivative - 1j * phase_dot * right_value
        ) * np.exp(-1j * phase)
        left_branch = left_value * np.exp(1j * phase)
        left_branch_dot = (
            left_derivative + 1j * phase_dot * left_value
        ) * np.exp(1j * phase)

        grouped_indices: dict[tuple[int, int], list[int]] = {}
        for index, (ell, m, _, _) in enumerate(selected_modes):
            grouped_indices.setdefault((ell, abs(m)), []).append(index)
        for (ell, m), indices in sorted(grouped_indices.items()):
            right_qnm = kerr_qnm_fundamental(ell, m, source.spin)
            left_qnm = kerr_qnm_fundamental(ell, -m, source.spin)
            right_omega = right_qnm.real / mass_seconds
            right_damping = abs(right_qnm.imag) / mass_seconds
            left_omega = left_qnm.real / mass_seconds
            left_damping = abs(left_qnm.imag) / mass_seconds
            transition = qnm_transition_m * mass_seconds
            active_damping = [right_damping]
            if np.any(np.abs(left_branch[indices]) > 0.0):
                active_damping.append(left_damping)
            slowest_damping = min(active_damping)
            fade_start = max(transition, qnm_efolds / slowest_damping)
            fade_duration = qnm_fade_efolds / slowest_damping
            ringdown_groups.append(
                FEWRingdownGroup(
                    ell,
                    m,
                    complex(np.sum(right_branch[indices])),
                    complex(np.sum(right_branch_dot[indices])),
                    complex(np.sum(left_branch[indices])),
                    complex(np.sum(left_branch_dot[indices])),
                    right_omega,
                    right_damping,
                    left_omega,
                    left_damping,
                    transition,
                    fade_start,
                    fade_duration,
                )
            )

    plunge_local = float(plunge_global - segment_start)
    ringdown_stop = plunge_local
    if ringdown_groups:
        ringdown_stop += max(group.stop for group in ringdown_groups)

    result = FEWModeCollection(
        source,
        modes,
        selected_modes,
        carrier_members,
        tuple(carriers),
        trajectory,
        amplitude_model,
        sparse_time,
        p,
        e,
        x,
        CubicSpline(sparse_time, plus_matrix, axis=0),
        CubicSpline(sparse_time, cross_matrix, axis=0),
        grouped_mode_vectors,
        raw_mode_vectors,
        right_spline,
        left_spline,
        segment_start,
        segment_stop,
        plunge_local,
        endpoint_model,
        tuple(ringdown_groups),
        ringdown_stop,
    )
    runtime.source_evaluations += 1
    return result


def _mode_pair_algebra_error(carrier: FEWModeCarrier) -> float:
    times = np.linspace(0.0, carrier.duration, 1001)
    global_times = times + carrier.segment_start
    state = carrier.trajectory.inspiral_generator.eval_integrator_spline(global_times)
    _, m, k, n = carrier.mode
    phase = m * state[:, 3] + k * state[:, 4] + n * state[:, 5]
    direct_amp = carrier.amplitude_model(
        carrier.source.spin,
        state[:, 0],
        state[:, 1],
        state[:, 2],
        specific_modes=[carrier.mode],
    )[carrier.mode]
    ell, m, _, _ = carrier.mode
    ylms = GetYlms(include_minus_m=True)(
        np.array([ell], dtype=int),
        np.array([m], dtype=int),
        carrier.source.viewing_theta,
        carrier.source.viewing_phi,
    )
    distance_scale = carrier.source.mu_solar * MRSUN_SI / (
        carrier.source.distance_gpc * Gpc
    )
    right = ylms[0] * direct_amp * distance_scale
    left = (
        ((-1) ** ell) * ylms[1] * np.conj(direct_amp) * distance_scale
        if m > 0
        else np.zeros_like(right)
    )
    waveform = right * np.exp(-1j * phase) + left * np.exp(1j * phase)
    waveform *= np.exp(2j * carrier.source.polarization)

    plus_unrotated = np.conj(right) + left
    cross_unrotated = 1j * (left - np.conj(right))
    c2 = math.cos(2.0 * carrier.source.polarization)
    s2 = math.sin(2.0 * carrier.source.polarization)
    plus = c2 * plus_unrotated + s2 * cross_unrotated
    cross = c2 * cross_unrotated - s2 * plus_unrotated
    reconstructed = (
        plus.real * np.cos(phase)
        - plus.imag * np.sin(phase)
        - 1j * (cross.real * np.cos(phase) - cross.imag * np.sin(phase))
    )
    scale = max(float(np.max(np.abs(waveform))), np.finfo(float).tiny)
    return float(np.max(np.abs(waveform - reconstructed)) / scale)


def _mode_set_algebra_error(
    collection: FEWModeCollection,
) -> float:
    """Compare the folded collection directly with FEW's mode-sum formula."""

    keep = (collection.sparse_time >= collection.segment_start) & (
        collection.sparse_time <= collection.sparse_time[-1]
    )
    global_times = collection.sparse_time[keep]
    times = global_times - collection.segment_start
    state = collection.trajectory.inspiral_generator.eval_integrator_spline(global_times)
    amplitudes = collection.amplitude_model(
        collection.source.spin,
        collection.sparse_p[keep],
        collection.sparse_e[keep],
        collection.sparse_x[keep],
        specific_modes=list(collection.selected_modes),
    )
    ell = np.array([mode[0] for mode in collection.selected_modes], dtype=int)
    m = np.array([mode[1] for mode in collection.selected_modes], dtype=int)
    ylms = GetYlms(include_minus_m=True)(
        ell,
        m,
        collection.source.viewing_theta,
        collection.source.viewing_phi,
    )
    scale = collection.source.mu_solar * MRSUN_SI / (
        collection.source.distance_gpc * Gpc
    )
    direct = np.zeros(times.size, dtype=np.complex128)
    for index, mode in enumerate(collection.selected_modes):
        ell_i, m_i, k_i, n_i = mode
        phase = m_i * state[:, 3] + k_i * state[:, 4] + n_i * state[:, 5]
        amplitude = amplitudes[mode]
        right = ylms[index] * amplitude * scale
        direct += right * np.exp(-1j * phase)
        if m_i > 0:
            left = (
                ((-1) ** ell_i)
                * ylms[len(collection.selected_modes) + index]
                * np.conj(amplitude)
                * scale
            )
            direct += left * np.exp(1j * phase)
    direct *= np.exp(2j * collection.source.polarization)
    hp, hc, _, _ = collection.polarization_components(times)
    reconstructed = np.sum(hp, axis=1) - 1j * np.sum(hc, axis=1)
    denominator = max(float(np.max(np.abs(direct))), np.finfo(float).tiny)
    return float(np.max(np.abs(direct - reconstructed)) / denominator)


def native_amplitude_dense_reevaluation_difference(
    carrier: FEWModeCarrier, samples: int = 2001
) -> tuple[float, float]:
    global_times = np.linspace(carrier.segment_start, carrier.segment_stop, samples)
    state = carrier.trajectory.inspiral_generator.eval_integrator_spline(global_times)
    direct_amp = carrier.amplitude_model(
        carrier.source.spin,
        state[:, 0],
        state[:, 1],
        state[:, 2],
        specific_modes=[carrier.mode],
    )[carrier.mode]
    ell, m, _, _ = carrier.mode
    ylms = GetYlms(include_minus_m=True)(
        np.array([ell], dtype=int),
        np.array([m], dtype=int),
        carrier.source.viewing_theta,
        carrier.source.viewing_phi,
    )
    scale = carrier.source.mu_solar * MRSUN_SI / (
        carrier.source.distance_gpc * Gpc
    )
    right = ylms[0] * direct_amp * scale
    left = (
        ((-1) ** ell) * ylms[1] * np.conj(direct_amp) * scale
        if m > 0
        else np.zeros_like(right)
    )
    plus = np.conj(right) + left
    cross = 1j * (left - np.conj(right))
    c2 = math.cos(2.0 * carrier.source.polarization)
    s2 = math.sin(2.0 * carrier.source.polarization)
    plus_rotated = c2 * plus + s2 * cross
    cross_rotated = c2 * cross - s2 * plus
    plus_interp = carrier.plus_coeff_spline(global_times)
    cross_interp = carrier.cross_coeff_spline(global_times)
    numerator = np.abs(plus_interp - plus_rotated) ** 2
    numerator += np.abs(cross_interp - cross_rotated) ** 2
    denominator = np.abs(plus_rotated) ** 2 + np.abs(cross_rotated) ** 2
    floor = 1.0e-14 * max(float(np.max(denominator)), np.finfo(float).tiny)
    valid = denominator > floor
    relative = np.sqrt(numerator[valid] / denominator[valid])
    return float(np.sqrt(np.mean(relative**2))), float(np.max(relative))


def _geometry_at_times(
    times: np.ndarray,
    source: FEWSourceParams,
    l_splines: list[CubicSpline],
    p_splines: list[CubicSpline],
    v_splines: list[CubicSpline],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    _, _, kv = sky_vectors(source)
    u, v, _ = sky_vectors(source)
    eplus = np.outer(v, v) - np.outer(u, u)
    ecross = np.outer(u, v) + np.outer(v, u)
    larm = np.stack([spline(times) for spline in l_splines], axis=1)
    pos = np.stack([spline(times) for spline in p_splines], axis=1).reshape(-1, 3, 3)
    arms = np.stack([spline(times) for spline in v_splines], axis=1).reshape(-1, 3, 3)
    kr = np.einsum("nij,j->ni", pos, kv)
    kn = np.einsum("nij,j->ni", arms, kv)
    plus = np.einsum("nij,jk,nik->ni", arms, eplus, arms)
    cross = np.einsum("nij,jk,nik->ni", arms, ecross, arms)
    app = 0.5 * plus / (1.0 + kn)
    apm = 0.5 * plus / (1.0 - kn)
    acp = 0.5 * cross / (1.0 + kn)
    acm = 0.5 * cross / (1.0 - kn)
    return larm, kr, app, apm, acp, acm


def _channel_terms(
    channel: str,
    times: np.ndarray,
    larm: np.ndarray,
    kr: np.ndarray,
    app: np.ndarray,
    apm: np.ndarray,
    acp: np.ndarray,
    acm: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    a, b, c = CHANNEL_TRIPLES[channel]
    delayed = _channel_delayed_source_times(channel, times, larm, kr)
    return [
        (delayed[:, 0], app[:, c] - apm[:, b], acp[:, c] - acm[:, b]),
        (delayed[:, 1], -app[:, c] + apm[:, c], -acp[:, c] + acm[:, c]),
        (delayed[:, 2], apm[:, b] - app[:, b], acm[:, b] - acp[:, b]),
        (delayed[:, 3], -apm[:, c] + apm[:, b], -acm[:, c] + acm[:, b]),
        (delayed[:, 4], app[:, b] - app[:, c], acp[:, b] - acp[:, c]),
        (delayed[:, 5], -apm[:, b] + app[:, b], -acm[:, b] + acp[:, b]),
        (delayed[:, 6], app[:, c] - apm[:, c], acp[:, c] - acm[:, c]),
        (delayed[:, 7], -app[:, b] + apm[:, c], -acp[:, b] + acm[:, c]),
    ]


def _channel_delayed_source_times(
    channel: str,
    output_time: np.ndarray,
    larm: np.ndarray,
    kr: np.ndarray,
) -> np.ndarray:
    """Return the eight source-waveform arguments for one TDI-1 channel."""

    a, b, c = CHANNEL_TRIPLES[channel]
    return np.column_stack(
        (
            output_time - kr[:, a] - 2.0 * larm[:, c] - 2.0 * larm[:, b],
            output_time - kr[:, b] - larm[:, c] - 2.0 * larm[:, b],
            output_time - kr[:, c] - larm[:, b] - 2.0 * larm[:, c],
            output_time - kr[:, a] - 2.0 * larm[:, b],
            output_time - kr[:, a] - 2.0 * larm[:, c],
            output_time - kr[:, c] - larm[:, b],
            output_time - kr[:, b] - larm[:, c],
            output_time - kr[:, a],
        )
    )


def tdi_source_event_output_bounds(
    source_time: float,
    source: FEWSourceParams,
    channel: str,
    l_splines: list[CubicSpline],
    p_splines: list[CubicSpline],
    v_splines: list[CubicSpline],
    *,
    tdi_generation: int = 1,
) -> tuple[float, float]:
    """SSB-output bounds at which delayed TDI terms see a source event.

    Each of the eight roots satisfies ``delayed_source_time(t)=source_time``.
    The delay varies only on the annual orbital timescale, so fixed-point
    correction converges rapidly while retaining the full sky projection and
    unequal-arm delays.
    """

    del v_splines  # Delayed times need positions and arm lengths, not projections.
    _, _, kv = sky_vectors(source)
    roots = np.full(8, float(source_time), dtype=np.float64)
    for _ in range(6):
        larm = np.stack([spline(roots) for spline in l_splines], axis=1)
        pos = np.stack([spline(roots) for spline in p_splines], axis=1).reshape(-1, 3, 3)
        kr = np.einsum("nij,j->ni", pos, kv)
        delayed = _channel_delayed_source_times(channel, roots, larm, kr)
        residual = delayed[np.arange(8), np.arange(8)] - source_time
        roots -= residual
    upper = float(np.max(roots))
    if tdi_generation == 2:
        upper += 4.0/(2.0*PI*FSTAR)
    return float(np.min(roots)), upper


def mode_pair_tdi_response(
    carrier: FEWModeCarrier,
    times: np.ndarray,
    channels: Iterable[str],
    l_splines: list[CubicSpline],
    p_splines: list[CubicSpline],
    v_splines: list[CubicSpline],
    chunk_size: int = 100000,
    tdi_generation: int = 1,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    times = np.asarray(times, dtype=np.float64)
    requested = tuple(channels)
    if tdi_generation == 2:
        latitude = math.asin(float(carrier.source.ecliptic_costheta))

        def polarizations(source_time: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            hp, hc, hpf, hcf = carrier.polarizations(source_time)
            return hp-1j*hpf, hc-1j*hcf

        analytic = complex_tdi2(times, latitude,
                                carrier.source.ecliptic_longitude,
                                p_splines, polarizations)
        return {channel: (analytic["XYZ".index(channel), 0].real,
                          -analytic["XYZ".index(channel), 0].imag)
                for channel in requested}
    if tdi_generation != 1:
        raise ValueError("tdi_generation must be 1 or 2")
    output = {
        channel: (np.empty(times.size), np.empty(times.size))
        for channel in requested
    }
    for first in range(0, times.size, chunk_size):
        last = min(first + chunk_size, times.size)
        local = times[first:last]
        larm, kr, app, apm, acp, acm = _geometry_at_times(
            local, carrier.source, l_splines, p_splines, v_splines
        )
        for channel in requested:
            value = np.zeros(local.size)
            quadrature = np.zeros(local.size)
            for delayed, coef_plus, coef_cross in _channel_terms(
                channel, local, larm, kr, app, apm, acp, acm
            ):
                hp, hc, hpf, hcf = carrier.polarizations(delayed)
                value += coef_plus * hp + coef_cross * hc
                quadrature += coef_plus * hpf + coef_cross * hcf
            output[channel][0][first:last] = value
            output[channel][1][first:last] = quadrature
    return output


def mode_set_tdi_response(
    collection: FEWModeCollection,
    times: np.ndarray,
    channels: Iterable[str],
    l_splines: list[CubicSpline],
    p_splines: list[CubicSpline],
    v_splines: list[CubicSpline],
    *,
    per_mode: bool,
    chunk_size: int = 100000,
    tdi_generation: int = 1,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Evaluate all selected modes with shared geometry and orbital phases.

    With ``per_mode=False`` the polarizations are summed at every delayed
    source time before the TDI terms are accumulated.  This is the direct
    all-mode reference.  ``per_mode=True`` retains one carrier quadrature per
    row for sparse AP extraction and the mode-local early transforms.
    """

    times = np.asarray(times, dtype=np.float64)
    requested = tuple(channels)
    if tdi_generation == 2:
        latitude = math.asin(float(collection.source.ecliptic_costheta))

        def polarizations(source_time: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            if per_mode:
                hp, hc, hpf, hcf = collection.polarization_components(source_time)
                return (hp-1j*hpf).T, (hc-1j*hcf).T
            return collection.summed_polarizations(source_time)

        analytic = complex_tdi2(times, latitude,
                                collection.source.ecliptic_longitude,
                                p_splines, polarizations)
        if per_mode:
            return {channel: (analytic["XYZ".index(channel)].real,
                              -analytic["XYZ".index(channel)].imag)
                    for channel in requested}
        return {channel: (analytic["XYZ".index(channel), 0].real,
                          -analytic["XYZ".index(channel), 0].imag)
                for channel in requested}
    if tdi_generation != 1:
        raise ValueError("tdi_generation must be 1 or 2")
    out_shape = (collection.size, times.size) if per_mode else (times.size,)
    output = {
        channel: (np.empty(out_shape), np.empty(out_shape))
        for channel in requested
    }
    for first in range(0, times.size, chunk_size):
        last = min(first + chunk_size, times.size)
        local = times[first:last]
        larm, kr, app, apm, acp, acm = _geometry_at_times(
            local, collection.source, l_splines, p_splines, v_splines
        )
        for channel in requested:
            if per_mode:
                value = np.zeros((collection.size, local.size))
                quadrature = np.zeros_like(value)
            else:
                value = np.zeros(local.size)
                quadrature = np.zeros(local.size)
            for delayed, coef_plus, coef_cross in _channel_terms(
                channel, local, larm, kr, app, apm, acp, acm
            ):
                if per_mode:
                    hp, hc, hpf, hcf = collection.polarization_components(delayed)
                    contribution = coef_plus[:, None] * hp + coef_cross[:, None] * hc
                    contribution_f = coef_plus[:, None] * hpf + coef_cross[:, None] * hcf
                    value += contribution.T
                    quadrature += contribution_f.T
                else:
                    hp, hc = collection.summed_polarizations(delayed)
                    value += coef_plus * hp + coef_cross * hc
            if per_mode:
                output[channel][0][:, first:last] = value
                output[channel][1][:, first:last] = quadrature
            else:
                output[channel][0][first:last] = value
                output[channel][1][first:last] = quadrature
    return output


def sampled_sum_tdi_response(
    collection: FEWModeCollection,
    times: np.ndarray,
    channel: str,
    sample_dt: float,
    l_splines: list[CubicSpline],
    p_splines: list[CubicSpline],
    v_splines: list[CubicSpline],
    chunk_size: int = 100000,
    tdi_generation: int = 1,
) -> np.ndarray:
    """Apply TDI to one sampled spline of the complete mode sum.

    This is used only by the common plunge packet.  It evaluates all selected
    carriers once on a fine source-time grid, sums their polarizations, and
    then delays the two real summed series.  The direct validation reference
    continues to query FEW independently at every delayed time.
    """

    times = np.asarray(times, dtype=np.float64)
    delay_margin = 800.0 if tdi_generation == 2 else 600.0
    source_start = math.floor((times[0] - delay_margin) / sample_dt) * sample_dt
    source_stop = min(
        math.ceil((times[-1] + delay_margin) / sample_dt) * sample_dt,
        collection.ringdown_stop,
    )
    source_times = np.arange(
        source_start, source_stop, sample_dt, dtype=np.float64
    )
    source_times = np.unique(
        np.concatenate((source_times, np.array([collection.ringdown_stop])))
    )
    hp_sum = np.empty(source_times.size)
    hc_sum = np.empty(source_times.size)
    for first in range(0, source_times.size, chunk_size):
        last = min(first + chunk_size, source_times.size)
        hp, hc = collection.summed_polarizations(source_times[first:last])
        hp_sum[first:last] = hp
        hc_sum[first:last] = hc
    hp_spline = CubicSpline(source_times, hp_sum, bc_type="natural")
    hc_spline = CubicSpline(source_times, hc_sum, bc_type="natural")

    def delayed_polarizations(delayed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        valid = (delayed >= source_times[0]) & (delayed <= collection.ringdown_stop)
        hp = np.zeros_like(delayed)
        hc = np.zeros_like(delayed)
        hp[valid] = hp_spline(delayed[valid])
        hc[valid] = hc_spline(delayed[valid])
        return hp, hc

    if tdi_generation == 2:
        latitude = math.asin(float(collection.source.ecliptic_costheta))
        analytic = complex_tdi2(times, latitude,
                                collection.source.ecliptic_longitude,
                                p_splines, delayed_polarizations)
        return analytic["XYZ".index(channel), 0].real
    if tdi_generation != 1:
        raise ValueError("tdi_generation must be 1 or 2")

    output = np.empty(times.size)
    for first in range(0, times.size, chunk_size):
        last = min(first + chunk_size, times.size)
        local = times[first:last]
        larm, kr, app, apm, acp, acm = _geometry_at_times(
            local, collection.source, l_splines, p_splines, v_splines
        )
        value = np.zeros(local.size)
        for delayed, coef_plus, coef_cross in _channel_terms(
            channel, local, larm, kr, app, apm, acp, acm
        ):
            hp, hc = delayed_polarizations(delayed)
            value += coef_plus * hp + coef_cross * hc
        output[first:last] = value
    return output


def _transfer_interval_fails(f0: float, fm: float, f1: float, width: float,
                             tdi_generation: int = 1) -> bool:
    values = np.abs([f0, fm, f1])
    fmin = float(np.min(values))
    fmax = float(np.max(values))
    spacing = PI * FSTAR / (2.0 if tdi_generation == 2 else 1.0)
    if fmax < spacing or width <= 1.0:
        return False
    first = max(1, int(math.ceil(fmin / spacing)))
    last = int(math.floor(fmax / spacing))
    if first > last:
        return False
    max_width = AP_TRANSFER_PHASE_STEP / (2.0 * PI * max(fmax, spacing))
    return width > max(1.0, max_width)


def _frequency_interval_ok(f0: float, fm: float, f1: float, width: float,
                           tdi_generation: int = 1) -> bool:
    values = np.abs([f0, fm, f1])
    fscale = max(float(np.max(values)), 1.0e-8)
    ferr = abs(fm - 0.5 * (f0 + f1))
    if abs(f1 - f0) / fscale > AP_FREQ_STEP_REL_TOL:
        return False
    if ferr / fscale > AP_FREQ_REL_CURVATURE_TOL:
        return False
    if 0.5 * PI * ferr * width > AP_PHASE_CURVATURE_TOL:
        return False
    return not _transfer_interval_fails(f0, fm, f1, width, tdi_generation)


def _frequency_intervals_ok(
    f0: np.ndarray, fm: np.ndarray, f1: np.ndarray, width: np.ndarray,
    tdi_generation: int = 1,
) -> np.ndarray:
    """Vectorized equivalent of ``_frequency_interval_ok``."""

    values = np.abs(np.stack((f0, fm, f1), axis=1))
    fmin = np.min(values, axis=1)
    fmax = np.max(values, axis=1)
    fscale = np.maximum(fmax, 1.0e-8)
    ferr = np.abs(fm - 0.5 * (f0 + f1))
    okay = np.abs(f1 - f0) / fscale <= AP_FREQ_STEP_REL_TOL
    okay &= ferr / fscale <= AP_FREQ_REL_CURVATURE_TOL
    okay &= 0.5 * PI * ferr * width <= AP_PHASE_CURVATURE_TOL

    spacing = PI * FSTAR / (2.0 if tdi_generation == 2 else 1.0)
    first = np.maximum(1, np.ceil(fmin / spacing).astype(np.int64))
    last = np.floor(fmax / spacing).astype(np.int64)
    crosses_transfer = (fmax >= spacing) & (width > 1.0) & (first <= last)
    max_width = AP_TRANSFER_PHASE_STEP / (
        2.0 * PI * np.maximum(fmax, spacing)
    )
    transfer_fails = crosses_transfer & (width > np.maximum(1.0, max_width))
    return okay & ~transfer_fails


def build_tdi_time_grid(
    carrier: FEWModeCarrier,
    shape: WDMShape,
    p_splines: list[CubicSpline],
    maximum_step: float = AP_DTM_MAX,
    taper_width: float | None = None,
    diagnostics: dict[str, float] | None = None,
    *,
    l_splines: list[CubicSpline] | None = None,
    channel: str = "X",
    tdi_generation: int = 1,
) -> np.ndarray:
    total_start = time.perf_counter()
    maximum_step = float(maximum_step)
    reference_seconds = 0.0
    frequency_seconds = 0.0
    decision_seconds = 0.0
    frequency_calls = 0
    frequency_values = 0
    rejected_intervals = 0
    _, _, kv = sky_vectors(carrier.source)
    channel = channel.upper()
    if channel not in CHANNEL_TRIPLES:
        raise ValueError("channel must be X, Y, or Z")

    def detector_frequency(values: np.ndarray) -> np.ndarray:
        nonlocal reference_seconds, frequency_seconds, frequency_calls, frequency_values
        start = time.perf_counter()
        positions = np.stack([spline(values) for spline in p_splines], axis=1)
        reference = values - positions[:, :3] @ kv
        arguments = [reference]
        if l_splines is not None:
            position = positions.reshape(-1, 3, 3)
            kr = np.einsum("nij,j->ni", position, kv)
            larm = np.stack([spline(values) for spline in l_splines], axis=1)
            arguments.extend(
                np.moveaxis(
                    _channel_delayed_source_times(
                        channel, values, larm, kr
                    ),
                    1,
                    0,
                )
            )
            if tdi_generation == 2:
                a, b, c = CHANNEL_TRIPLES[channel]
                arguments.extend(
                    values-kr[:, spacecraft]-8.0/(2.0*PI*FSTAR)
                    for spacecraft in (a, b, c)
                )
        source_arguments = np.column_stack(arguments)
        reference_seconds += time.perf_counter() - start
        start = time.perf_counter()
        result = carrier.frequency(source_arguments.ravel()).reshape(
            values.size, source_arguments.shape[1]
        )
        frequency_seconds += time.perf_counter() - start
        frequency_calls += 1
        frequency_values += source_arguments.size
        return result

    # Most of a long EMRI observation is accepted at the requested orbital cadence.
    # Probe all such intervals in one backend call and avoid hundreds of tiny
    # three-sample FEW and barycentric-map calls when no refinement is needed.
    coarse = np.arange(0.0, shape.Tobs, maximum_step, dtype=np.float64)
    if coarse.size == 0 or coarse[0] != 0.0:
        coarse = np.insert(coarse, 0, 0.0)
    if coarse[-1] != shape.Tobs:
        coarse = np.append(coarse, shape.Tobs)
    widths = np.diff(coarse)
    probe = np.column_stack(
        (coarse[:-1], 0.5 * (coarse[:-1] + coarse[1:]), coarse[1:])
    )
    frequency = detector_frequency(probe.ravel())
    argument_count = frequency.shape[1]
    frequency = frequency.reshape(-1, 3, argument_count)
    start = time.perf_counter()
    coarse_ok = _frequency_intervals_ok(
        frequency[:, 0, :].ravel(),
        frequency[:, 1, :].ravel(),
        frequency[:, 2, :].ravel(),
        np.repeat(widths, argument_count),
        tdi_generation,
    ).reshape(-1, argument_count)
    coarse_ok = np.all(coarse_ok, axis=1)
    decision_seconds += time.perf_counter() - start

    if np.all(coarse_ok):
        times = coarse.tolist()
    else:
        # Preserve the original sequential planner for the dynamically refined
        # merger/transfer-crossing region.  Every coarse interval before the
        # first failure has already passed exactly the test that the sequential
        # path would apply, so retain that prefix instead of querying it again.
        first_bad = int(np.flatnonzero(~coarse_ok)[0])
        times = coarse[: first_bad + 1].tolist()
        current = float(coarse[first_bad])
        step = maximum_step
        while current < shape.Tobs:
            step = min(step, maximum_step, shape.Tobs - current)
            while True:
                trial = current + step
                midpoint = 0.5 * (current + trial)
                frequency = detector_frequency(
                    np.array([current, midpoint, trial])
                )
                start = time.perf_counter()
                interval_ok = _frequency_interval_ok(
                    float(frequency[0, 0]),
                    float(frequency[1, 0]),
                    float(frequency[2, 0]),
                    step,
                    tdi_generation,
                )
                if interval_ok and argument_count > 1:
                    interval_ok = bool(
                        np.all(
                            _frequency_intervals_ok(
                                frequency[0],
                                frequency[1],
                                frequency[2],
                                np.full(argument_count, step),
                                tdi_generation,
                            )
                        )
                    )
                decision_seconds += time.perf_counter() - start
                if interval_ok or step <= 1.0:
                    break
                rejected_intervals += 1
                step *= 0.5
            current = min(current + step, shape.Tobs)
            times.append(current)
            step = min(maximum_step, 1.5 * step)

    edge_start = time.perf_counter()
    output = np.asarray(times, dtype=np.float64)
    if taper_width is not None and taper_width > 0.0:
        edge_step = min(maximum_step, taper_width / 32.0)
        left = np.arange(0.0, min(taper_width, shape.Tobs) + 0.5 * edge_step, edge_step)
        right = shape.Tobs - left
        output = np.unique(np.concatenate((output, left, right)))
        output = output[(output >= 0.0) & (output <= shape.Tobs)]
    edge_seconds = time.perf_counter() - edge_start
    total_seconds = time.perf_counter() - total_start
    if diagnostics is not None:
        diagnostics.update(
            {
                "total_seconds": total_seconds,
                "reference_time_seconds": reference_seconds,
                "few_frequency_seconds": frequency_seconds,
                "interval_tests_seconds": decision_seconds,
                "edge_merge_seconds": edge_seconds,
                "loop_overhead_seconds": max(
                    0.0,
                    total_seconds
                    - reference_seconds
                    - frequency_seconds
                    - decision_seconds
                    - edge_seconds,
                ),
                "accepted_intervals": float(len(times) - 1),
                "rejected_intervals": float(rejected_intervals),
                "frequency_calls": float(frequency_calls),
                "frequency_values": float(frequency_values),
                "adaptive_points_before_edges": float(len(times)),
                "final_grid_points": float(output.size),
                "minimum_step_seconds": float(np.min(np.diff(output))),
                "maximum_step_seconds": float(np.max(np.diff(output))),
            }
        )
    return output


def build_shared_tdi_time_grid(
    collection: FEWModeCollection,
    shape: WDMShape,
    p_splines: list[CubicSpline],
    maximum_step: float = AP_DTM_MAX,
    taper_width: float | None = None,
    planning_stop: float | None = None,
    diagnostics: dict[str, float] | None = None,
    *,
    l_splines: list[CubicSpline] | None = None,
    channels: Iterable[str] = ("X", "Y", "Z"),
    tdi_generation: int = 1,
) -> np.ndarray:
    """Plan one SSB-output grid satisfying every retained FEW carrier.

    With arm splines supplied, every actual delayed source-time argument used
    by the requested TDI channels is tested.  The guiding-center reference is
    included separately because it is used to remove the carrier phase during
    amplitude/phase extraction.
    """

    total_start = time.perf_counter()
    maximum_step = float(maximum_step)
    reference_seconds = 0.0
    frequency_seconds = 0.0
    decision_seconds = 0.0
    frequency_calls = 0
    frequency_values = 0
    rejected_intervals = 0
    _, _, kv = sky_vectors(collection.source)
    requested_channels = tuple(channel.upper() for channel in channels)
    if any(channel not in CHANNEL_TRIPLES for channel in requested_channels):
        raise ValueError("channels must be selected from X, Y, Z")
    active_stop = shape.Tobs if planning_stop is None else min(
        max(float(planning_stop), 0.0), shape.Tobs
    )

    def detector_frequencies(values: np.ndarray) -> np.ndarray:
        nonlocal reference_seconds, frequency_seconds, frequency_calls, frequency_values
        start = time.perf_counter()
        positions = np.stack([spline(values) for spline in p_splines], axis=1)
        position0 = positions[:, :3]
        reference = values - position0 @ kv
        arguments = [reference]
        if l_splines is not None:
            position = positions.reshape(-1, 3, 3)
            kr = np.einsum("nij,j->ni", position, kv)
            larm = np.stack([spline(values) for spline in l_splines], axis=1)
            for channel in requested_channels:
                arguments.extend(
                    np.moveaxis(
                        _channel_delayed_source_times(
                            channel, values, larm, kr
                        ),
                        1,
                        0,
                    )
                )
                if tdi_generation == 2:
                    a, b, c = CHANNEL_TRIPLES[channel]
                    arguments.extend(
                        values-kr[:, spacecraft]-8.0/(2.0*PI*FSTAR)
                        for spacecraft in (a, b, c)
                    )
        source_arguments = np.column_stack(arguments)
        reference_seconds += time.perf_counter() - start
        start = time.perf_counter()
        result = collection.frequency_matrix(source_arguments.ravel()).reshape(
            values.size, source_arguments.shape[1], collection.size
        )
        frequency_seconds += time.perf_counter() - start
        frequency_calls += 1
        frequency_values += source_arguments.size * collection.size
        return result

    coarse = np.arange(0.0, active_stop, maximum_step, dtype=np.float64)
    if coarse.size == 0 or coarse[0] != 0.0:
        coarse = np.insert(coarse, 0, 0.0)
    if coarse[-1] != active_stop:
        coarse = np.append(coarse, active_stop)
    widths = np.diff(coarse)
    probe = np.column_stack(
        (coarse[:-1], 0.5 * (coarse[:-1] + coarse[1:]), coarse[1:])
    )
    frequency = detector_frequencies(probe.ravel())
    argument_count = frequency.shape[1]
    frequency = frequency.reshape(-1, 3, argument_count, collection.size)
    start = time.perf_counter()
    all_ok = _frequency_intervals_ok(
        frequency[:, 0, :, :].ravel(),
        frequency[:, 1, :, :].ravel(),
        frequency[:, 2, :, :].ravel(),
        np.repeat(widths, argument_count * collection.size),
        tdi_generation,
    ).reshape(-1, argument_count, collection.size)
    coarse_ok = np.all(all_ok, axis=(1, 2))
    decision_seconds += time.perf_counter() - start

    if np.all(coarse_ok):
        times = coarse.tolist()
    else:
        first_bad = int(np.flatnonzero(~coarse_ok)[0])
        times = coarse[: first_bad + 1].tolist()
        current = float(coarse[first_bad])
        step = maximum_step
        while current < active_stop:
            step = min(step, maximum_step, active_stop - current)
            while True:
                trial = current + step
                midpoint = 0.5 * (current + trial)
                frequency = detector_frequencies(
                    np.array([current, midpoint, trial])
                )
                start = time.perf_counter()
                interval_ok = bool(
                    np.all(
                        _frequency_intervals_ok(
                            frequency[0].ravel(),
                            frequency[1].ravel(),
                            frequency[2].ravel(),
                            np.full(argument_count * collection.size, step),
                            tdi_generation,
                        )
                    )
                )
                decision_seconds += time.perf_counter() - start
                if interval_ok or step <= 1.0:
                    break
                rejected_intervals += 1
                step *= 0.5
            current = min(current + step, active_stop)
            times.append(current)
            step = min(maximum_step, 1.5 * step)

    edge_start = time.perf_counter()
    output = np.asarray(times, dtype=np.float64)
    if active_stop < shape.Tobs:
        output = np.unique(np.concatenate((output, [active_stop, shape.Tobs])))
    if taper_width is not None and taper_width > 0.0:
        edge_step = min(maximum_step, taper_width / 32.0)
        left = np.arange(0.0, min(taper_width, shape.Tobs) + 0.5 * edge_step, edge_step)
        right = shape.Tobs - left
        output = np.unique(np.concatenate((output, left, right)))
        output = output[(output >= 0.0) & (output <= shape.Tobs)]
    edge_seconds = time.perf_counter() - edge_start
    total_seconds = time.perf_counter() - total_start
    if diagnostics is not None:
        diagnostics.update(
            {
                "total_seconds": total_seconds,
                "reference_time_seconds": reference_seconds,
                "few_frequency_seconds": frequency_seconds,
                "interval_tests_seconds": decision_seconds,
                "edge_merge_seconds": edge_seconds,
                "loop_overhead_seconds": max(
                    0.0,
                    total_seconds
                    - reference_seconds
                    - frequency_seconds
                    - decision_seconds
                    - edge_seconds,
                ),
                "accepted_intervals": float(len(times) - 1),
                "rejected_intervals": float(rejected_intervals),
                "frequency_calls": float(frequency_calls),
                "frequency_values": float(frequency_values),
                "adaptive_points_before_edges": float(len(times)),
                "final_grid_points": float(output.size),
                "minimum_step_seconds": float(np.min(np.diff(output))),
                "maximum_step_seconds": float(np.max(np.diff(output))),
                "planning_stop_seconds": active_stop,
            }
        )
    return output


def extract_mode_set_tdi_ap(
    collection: FEWModeCollection,
    times: np.ndarray,
    channels: Iterable[str],
    l_splines: list[CubicSpline],
    p_splines: list[CubicSpline],
    v_splines: list[CubicSpline],
    *,
    tdi_generation: int = 1,
) -> tuple[TDIGrid, ...]:
    requested = tuple(channels)
    raw = mode_set_tdi_response(
        collection,
        times,
        requested,
        l_splines,
        p_splines,
        v_splines,
        per_mode=True,
        tdi_generation=tdi_generation,
    )
    reference_time = spacecraft0_reference_time(times, collection.source, p_splines)
    reference_phases = collection.phase_matrix(reference_time)
    bary_phases = collection.phase_matrix(times)
    output: list[TDIGrid] = []
    for index in range(collection.size):
        amplitude: dict[str, np.ndarray] = {}
        offset: dict[str, np.ndarray] = {}
        for channel in requested:
            value, quadrature = raw[channel]
            signed_amplitude, phase_offset = extract_ap(
                value[index], quadrature[index], reference_phases[:, index]
            )
            amplitude[channel] = signed_amplitude
            offset[channel] = phase_offset
        output.append(
            TDIGrid(
                times,
                reference_time,
                reference_phases[:, index],
                bary_phases[:, index],
                amplitude,
                offset,
            )
        )
    return tuple(output)


def extract_mode_pair_tdi_ap(
    carrier: FEWModeCarrier,
    times: np.ndarray,
    channels: Iterable[str],
    l_splines: list[CubicSpline],
    p_splines: list[CubicSpline],
    v_splines: list[CubicSpline],
    *,
    tdi_generation: int = 1,
) -> TDIGrid:
    raw = mode_pair_tdi_response(
        carrier, times, channels, l_splines, p_splines, v_splines,
        tdi_generation=tdi_generation,
    )
    reference_time = spacecraft0_reference_time(times, carrier.source, p_splines)
    reference_phase = carrier.phase(reference_time)
    bary_phase = carrier.phase(times)
    amplitude: dict[str, np.ndarray] = {}
    offset: dict[str, np.ndarray] = {}
    for channel, (value, quadrature) in raw.items():
        signed_amplitude, phase_offset = extract_ap(
            value, quadrature, reference_phase
        )
        amplitude[channel] = signed_amplitude
        offset[channel] = phase_offset
    return TDIGrid(times, reference_time, reference_phase, bary_phase, amplitude, offset)


def edge_window(times: np.ndarray, duration: float, width: float) -> np.ndarray:
    out = np.ones_like(times, dtype=np.float64)
    if width <= 0.0:
        return out
    width = min(width, 0.49 * duration)
    left = times < width
    right = times > duration - width
    out[left] = 0.5 * (1.0 - np.cos(PI * times[left] / width))
    out[right] = 0.5 * (1.0 - np.cos(PI * (duration - times[right]) / width))
    return out


def reconstruct_sparse_tdi(
    carrier: FEWModeCarrier,
    tdi: TDIGrid,
    channel: str,
    times: np.ndarray,
    p_splines: list[CubicSpline],
) -> np.ndarray:
    amp_spline = make_ap_spline(tdi.time, tdi.amplitude[channel])
    offset_spline = make_ap_spline(tdi.time, tdi.phase_offset[channel])
    reference_time = spacecraft0_reference_time(times, carrier.source, p_splines)
    phase = carrier.phase(reference_time) + offset_spline(times)
    return np.asarray(amp_spline(times) * np.cos(phase), dtype=np.float64)


def numerical_spa_wdm(
    carrier: FEWModeCarrier,
    tdi: TDIGrid,
    channel: str,
    shape: WDMShape,
    taper_width: float,
    time_weight: Callable[[np.ndarray], np.ndarray] | None = None,
    time_bounds: tuple[float, float] | None = None,
    fdot_sign: float = 1.0,
) -> tuple[WDMChannel, np.ndarray, np.ndarray]:
    times = tdi.time
    # FEW's accumulated orbital phases are continuous and may advance by many
    # radians between sparse response samples.  Only the atan2-derived TDI
    # offset is unwrapped by extract_ap; unwrapping this sum would alias the
    # physical carrier whenever a sparse phase step exceeds pi.
    phase = tdi.channel_phase(channel)
    offset_derivative = nonuniform_derivative_array(times, tdi.phase_offset[channel])
    reference_time_rate = nonuniform_derivative_array(times, tdi.detector_time)
    frequency = carrier.frequency(tdi.detector_time) * reference_time_rate
    frequency += offset_derivative / (2.0 * PI)
    fdot = nonuniform_derivative_array(times, frequency)
    window = edge_window(times, shape.Tobs, taper_width)
    if time_weight is not None:
        window *= time_weight(times)
    amplitude = tdi.amplitude[channel] * window
    in_interval = np.ones(times.size, dtype=bool)
    if time_bounds is not None:
        in_interval &= times >= time_bounds[0]
        in_interval &= times <= time_bounds[1]
    amplitude_floor = 1.0e-13 * max(
        float(np.max(np.abs(amplitude))), np.finfo(float).tiny
    )
    valid = (
        np.isfinite(frequency)
        & np.isfinite(fdot)
        & in_interval
        & (frequency > 0.0)
        & (fdot * fdot_sign > 0.0)
        & np.isfinite(amplitude)
        & (np.abs(amplitude) > amplitude_floor)
    )
    if np.count_nonzero(valid) < 4:
        raise RuntimeError("selected FEW mode has insufficient monotone SPA support")
    freq = frequency[valid]
    fd_phase = phase[valid] - 2.0 * PI * freq * (times[valid] - shape.Tobs)
    fd_phase += np.sign(fdot[valid]) * PI / 4.0
    fd_amp = amplitude[valid] / np.sqrt(np.abs(fdot[valid]))
    order = np.argsort(freq, kind="stable")
    freq = freq[order]
    fd_phase = fd_phase[order]
    fd_amp = fd_amp[order]
    keep = np.ones(freq.size, dtype=bool)
    keep[1:] = np.diff(freq) > np.finfo(float).eps * np.maximum(
        np.abs(freq[1:]), 1.0
    )
    freq = freq[keep]
    fd_phase = fd_phase[keep]
    fd_amp = fd_amp[keep]
    if freq.size < 4:
        raise RuntimeError("selected FEW SPA branch has insufficient unique frequencies")
    support = (
        np.isfinite(frequency)
        & in_interval
        & (np.abs(amplitude) > amplitude_floor)
    )
    nmid, nsize = wdm_pixels(times[support], np.abs(frequency[support]), shape)
    return wdm_track(freq, fd_phase, fd_amp, nmid, nsize, shape), frequency, fdot


def next_power_of_two(value: int) -> int:
    return 1 << max(0, int(value - 1).bit_length())


def tdi_track_frequency(
    carrier: FEWModeCarrier, tdi: TDIGrid, channel: str
) -> tuple[np.ndarray, np.ndarray]:
    """Stable post-TDI frequency and derivative on the sparse response grid."""

    times = tdi.time
    offset_derivative = nonuniform_derivative_array(
        times, tdi.phase_offset[channel]
    )
    reference_time_rate = nonuniform_derivative_array(times, tdi.detector_time)
    frequency = carrier.frequency(tdi.detector_time) * reference_time_rate
    frequency += offset_derivative / (2.0 * PI)
    return frequency, nonuniform_derivative_array(times, frequency)


def turnover_packet_envelope(
    times: np.ndarray, start: float, stop: float, taper: float
) -> np.ndarray:
    """Raised-cosine compact window used by a local turnover FFT."""

    values = np.asarray(times, dtype=np.float64)
    out = np.zeros_like(values)
    duration = stop - start
    if duration <= 0.0:
        return out
    taper = min(max(taper, 0.0), 0.5 * duration)
    inside = (values >= start) & (values <= stop)
    out[inside] = 1.0
    if taper <= 0.0:
        return out
    left = inside & (values < start + taper)
    right = inside & (values > stop - taper)
    out[left] = 0.5 * (
        1.0 - np.cos(PI * (values[left] - start) / taper)
    )
    out[right] = 0.5 * (
        1.0 - np.cos(PI * (stop - values[right]) / taper)
    )
    return out


def find_turnover_regions(
    carrier: FEWModeCarrier,
    tdi: TDIGrid,
    channel: str,
    shape: WDMShape,
    time_weight: Callable[[np.ndarray], np.ndarray],
    *,
    packet_pixels: float = 48.0,
    taper_pixels: float = 4.0,
    minimum_prominence_layers: float = 1.0e-2,
) -> tuple[list[FEWTurnoverRegion], np.ndarray, np.ndarray]:
    """Locate only WDM-resolvable frequency turnovers.

    A sign change in a noisy numerical ``fdot`` is not sufficient.  The
    frequency must turn by at least a small fraction of one WDM layer over the
    Meyer half-support.  Nearby candidate packets are merged before any FFT is
    built.
    """

    times = tdi.time
    frequency, fdot = tdi_track_frequency(carrier, tdi, channel)
    weight = np.asarray(time_weight(times), dtype=np.float64)
    amplitude = np.abs(tdi.amplitude[channel] * weight)
    amplitude_floor = 1.0e-10 * max(
        float(np.max(amplitude)), np.finfo(float).tiny
    )
    active = (
        np.isfinite(frequency)
        & np.isfinite(fdot)
        & (frequency > 0.0)
        & (weight > 1.0e-8)
        & (amplitude > amplitude_floor)
    )
    active_index = np.flatnonzero(active)
    if active_index.size < 4:
        return [], frequency, fdot

    active_times = times[active]
    frequency_spline = PchipInterpolator(
        active_times, frequency[active], extrapolate=False
    )
    probe_half_width = float(shape.mult) * shape.DT
    minimum_prominence = minimum_prominence_layers * shape.DF
    centers: list[tuple[float, float, float]] = []
    for left, right in zip(active_index[:-1], active_index[1:]):
        if right != left + 1:
            continue
        d0 = float(fdot[left])
        d1 = float(fdot[right])
        if d0 == 0.0:
            root = float(times[left])
        elif d1 == 0.0:
            root = float(times[right])
        elif d0 * d1 < 0.0:
            root = float(
                times[left]
                - d0 * (times[right] - times[left]) / (d1 - d0)
            )
        else:
            continue
        tleft = root - probe_half_width
        tright = root + probe_half_width
        if tleft < active_times[0] or tright > active_times[-1]:
            continue
        froot = float(frequency_spline(root))
        prominence = min(
            abs(float(frequency_spline(tleft)) - froot),
            abs(float(frequency_spline(tright)) - froot),
        )
        if prominence >= minimum_prominence:
            centers.append((root, froot, prominence))

    if not centers:
        return [], frequency, fdot

    duration = packet_pixels * shape.DT
    taper = taper_pixels * shape.DT
    candidates = [
        FEWTurnoverRegion(
            max(float(active_times[0]), center - 0.5 * duration),
            min(float(active_times[-1]), center + 0.5 * duration),
            center,
            fturn,
            prominence,
        )
        for center, fturn, prominence in centers
    ]
    merged: list[FEWTurnoverRegion] = []
    for region in candidates:
        if region.stop - region.start <= 2.0 * taper:
            continue
        if not merged or region.start >= merged[-1].stop:
            merged.append(region)
            continue
        previous = merged[-1]
        start = previous.start
        stop = max(previous.stop, region.stop)
        center = 0.5 * (start + stop)
        merged[-1] = FEWTurnoverRegion(
            start,
            stop,
            center,
            float(frequency_spline(center)),
            max(previous.prominence, region.prominence),
        )
    return merged, frequency, fdot


def local_turnover_fft_wdm(
    carrier: FEWModeCarrier,
    tdi: TDIGrid,
    channel: str,
    shape: WDMShape,
    taper_width: float,
    region: FEWTurnoverRegion,
    time_weight: Callable[[np.ndarray], np.ndarray],
    *,
    packet_taper_pixels: float = 4.0,
) -> tuple[WDMChannel, int]:
    """Transform one compact turnover partition with a heterodyned FFT."""

    live_pixels = int(math.ceil((region.stop - region.start) / shape.DT))
    block_pixels = min(
        next_power_of_two(max(live_pixels + 2, 2 * shape.mult)), shape.nt
    )
    center = int(round(region.center / shape.DT))
    if center % 2:
        center -= 1
    center = min(max(center, block_pixels // 2), shape.nt - block_pixels // 2)
    if center % 2:
        center -= 1
    block_start = (center - block_pixels // 2) * shape.DT
    block_stop = block_start + block_pixels * shape.DT

    frequency, _ = tdi_track_frequency(carrier, tdi, channel)
    finite = np.isfinite(frequency) & (frequency > 0.0)
    frequency_spline = PchipInterpolator(
        tdi.time[finite], frequency[finite], extrapolate=True
    )
    probe_start = max(region.start, block_start)
    probe_stop = min(region.stop, block_stop)
    probe_times = np.linspace(probe_start, probe_stop, 129)
    probe_frequency = np.asarray(frequency_spline(probe_times), dtype=np.float64)

    heterodyne_layer = 2 * int(round(region.frequency / (2.0 * shape.DF)))
    heterodyne_frequency = heterodyne_layer * shape.DF
    residual_band = float(
        np.max(np.abs(probe_frequency - heterodyne_frequency))
    ) + 2.0 * shape.DF
    samples_per_pixel = next_power_of_two(
        max(2, int(math.ceil(2.25 * shape.DT * residual_band)))
    )
    sample_dt = shape.DT / float(samples_per_pixel)
    sample_count = block_pixels * samples_per_pixel
    sample_times = block_start + sample_dt * np.arange(sample_count)

    amp_spline = make_ap_spline(tdi.time, tdi.amplitude[channel])
    offset_spline = make_ap_spline(tdi.time, tdi.phase_offset[channel])
    reference_time_spline = make_ap_spline(tdi.time, tdi.detector_time)
    reference_time = np.asarray(reference_time_spline(sample_times), dtype=np.float64)
    phase = carrier.phase(reference_time) + offset_spline(sample_times)
    amplitude = amp_spline(sample_times)
    packet_taper = packet_taper_pixels * shape.DT
    amplitude *= turnover_packet_envelope(
        sample_times, region.start, region.stop, packet_taper
    )
    amplitude *= time_weight(sample_times)
    amplitude *= edge_window(sample_times, shape.Tobs, taper_width)
    residual = amplitude * np.exp(
        1j * (phase - 2.0 * PI * heterodyne_frequency * sample_times)
    )
    spectrum = np.fft.fft(residual)

    frequency_min = max(
        float(np.min(probe_frequency)) - 2.0 * shape.DF, shape.DF
    )
    frequency_max = float(np.max(probe_frequency)) + 2.0 * shape.DF
    layer_min = max(1, int(math.floor(frequency_min / shape.DF)))
    layer_max = min(
        shape.nf - 1, int(math.ceil(frequency_max / shape.DF))
    )
    layers = np.arange(layer_min, layer_max + 1, dtype=np.int64)
    nmid = np.full(shape.nf, -1, dtype=np.int64)
    nsize = np.zeros(shape.nf, dtype=np.int64)
    nmid[layers] = center
    nsize[layers] = block_pixels

    half = block_pixels // 2
    base_n = np.arange(block_pixels, dtype=np.int64)
    offsets = np.arange(-half, half, dtype=np.float64)
    packet_df = 1.0 / (float(block_pixels) * shape.DT)
    fgrid = (
        layers[:, None].astype(np.float64) * shape.DF
        + offsets[None, :] * packet_df
    )
    residual_frequency = fgrid - heterodyne_frequency
    valid = (
        (base_n[None, :] > 0)
        & (fgrid > 0.0)
        & (np.abs(residual_frequency) < 0.5 / sample_dt)
    )
    data = np.zeros(fgrid.shape, dtype=np.complex128)
    if np.any(valid):
        bins_float = residual_frequency[valid] * sample_count * sample_dt
        bins_integer = np.rint(bins_float).astype(np.int64)
        aligned = np.abs(bins_float - bins_integer) < 1.0e-7
        target = np.flatnonzero(valid)[aligned]
        bins_integer = bins_integer[aligned] % sample_count
        flat_fgrid = fgrid.ravel()
        flat_residual_frequency = residual_frequency.ravel()
        transform = spectrum[bins_integer] * sample_dt
        transform *= np.exp(
            -2j * PI * flat_residual_frequency[target] * block_start
        )
        transform *= np.exp(
            2j * PI * flat_fgrid[target] * (shape.Tobs + block_start)
        )
        data.ravel()[target] = transform

    dom = 2.0 * PI / (float(block_pixels) * shape.DT)
    phihf = phitilde(
        np.arange(0, half + 1, dtype=np.float64) * dom, shape
    )
    window = phihf[np.abs(base_n - half)]
    window[0] = 0.0
    scale = math.sqrt(8.0 * PI / 15.0) / (
        float(block_pixels) * shape.DT
    )
    transformed = np.fft.ifft(data * window[None, :], axis=1) * block_pixels
    even = ((base_n[None, :] + layers[:, None]) % 2) == 0
    imag_sign = np.where((layers % 2) == 0, 1.0, -1.0)[:, None]
    coefficients = scale * np.where(
        even, transformed.real, imag_sign * transformed.imag
    )
    output_time = base_n + center - half
    keep_time = (output_time >= 0) & (output_time < shape.nt)
    listn = np.tile(output_time[keep_time], layers.size)
    listm = np.repeat(layers, np.count_nonzero(keep_time))
    values = coefficients[:, keep_time].reshape(-1)
    return (
        WDMChannel(
            np.empty(0),
            np.empty(0),
            np.empty(0),
            nmid,
            nsize,
            listn,
            listm,
            values,
        ),
        sample_count,
    )


def narrow_band_fft_wdm(
    carrier: FEWModeCarrier,
    tdi: TDIGrid,
    channel: str,
    shape: WDMShape,
    taper_width: float,
    *,
    samples_per_time_pixel: int = 2,
    time_weight: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[WDMChannel, np.ndarray, np.ndarray, int]:
    """Reference-quality fast transform of one folded carrier track.

    The analytic TDI carrier is heterodyned near the middle of its frequency
    range and sampled only fast enough for the residual bandwidth.  Requiring
    at least two samples per WDM time pixel also resolves the compact Meyer
    support.  This path handles a non-monotone track without an inverse t(f)
    map and is the numerical reference for the cheaper SPA plus local-packet
    construction.
    """

    times = tdi.time
    phase = tdi.channel_phase(channel)
    frequency = nonuniform_derivative_array(times, phase) / (2.0 * PI)
    fdot = nonuniform_derivative_array(times, frequency)
    sparse_weight = (
        np.ones_like(times) if time_weight is None else time_weight(times)
    )
    finite = np.isfinite(frequency) & (sparse_weight > 1.0e-12)
    if np.count_nonzero(finite) < 2:
        raise RuntimeError("selected FEW mode has no finite TDI frequency track")
    fmin = max(float(np.min(frequency[finite])) - 2.0 * shape.FB, 1.0 / shape.Tobs)
    fmax = float(np.max(frequency[finite])) + 2.0 * shape.FB
    fhet = round(0.5 * (fmin + fmax) * shape.Tobs) / shape.Tobs
    residual_band = max(fhet - fmin, fmax - fhet)
    bandwidth_samples = int(math.ceil(4.0 * shape.Tobs * residual_band))
    temporal_samples = samples_per_time_pixel * shape.nt
    sample_count = next_power_of_two(max(bandwidth_samples, temporal_samples, 16))
    sample_dt = shape.Tobs / float(sample_count)
    sample_times = sample_dt * np.arange(sample_count, dtype=np.float64)

    amp_spline = make_ap_spline(times, tdi.amplitude[channel])
    offset_spline = make_ap_spline(times, tdi.phase_offset[channel])
    reference_time_spline = make_ap_spline(times, tdi.detector_time)
    reference_time = np.asarray(reference_time_spline(sample_times), dtype=np.float64)
    sample_phase = carrier.phase(reference_time) + offset_spline(sample_times)
    sample_amplitude = amp_spline(sample_times)
    sample_amplitude *= edge_window(sample_times, shape.Tobs, taper_width)
    if time_weight is not None:
        sample_amplitude *= time_weight(sample_times)
    residual = sample_amplitude * np.exp(
        1j * (sample_phase - 2.0 * PI * fhet * sample_times)
    )
    spectrum = np.fft.fft(residual) * sample_dt
    frequencies = fhet + np.fft.fftfreq(sample_count, sample_dt)
    order = np.argsort(frequencies)
    frequencies = frequencies[order]
    spectrum = spectrum[order]
    keep = (frequencies >= fmin) & (frequencies <= fmax)
    frequencies = frequencies[keep]
    spectrum = spectrum[keep]

    # wdm_track stores Fourier phases relative to the observation endpoint.
    spectrum *= np.exp(2j * PI * frequencies * shape.Tobs)
    spectral_amplitude = np.abs(spectrum)
    spectral_phase = np.unwrap(np.angle(spectrum))
    nmid, nsize = wdm_pixels(times[finite], np.abs(frequency[finite]), shape)
    result = wdm_track(
        frequencies, spectral_phase, spectral_amplitude, nmid, nsize, shape
    )
    return result, frequency, fdot, sample_count


@dataclass
class _FEWEarlyBlockSpectrum:
    """Runtime state for one member of an early FFT partition."""

    tile_lo: int
    tile_hi: int
    support_lo: int
    support_hi: int
    block_start: float
    nonzero_start: float
    nonzero_stop: float
    frequency_min: float
    frequency_max: float
    frequency_pad: float
    heterodyne_layer: int
    sample_dt: float
    packet_time_pixels: int
    spectrum: np.ndarray


def _track_frequency_range(
    times: np.ndarray,
    frequency: np.ndarray,
    spline: PchipInterpolator,
    lower: float,
    upper: float,
) -> tuple[float, float]:
    """Conservative extrema of a shape-preserving track over one interval."""

    lower = max(float(lower), float(times[0]))
    upper = min(float(upper), float(times[-1]))
    if upper <= lower:
        value = float(spline(lower))
        return value, value
    inside = (times > lower) & (times < upper) & np.isfinite(frequency)
    probes = np.concatenate(
        (np.array([lower]), times[inside], np.array([upper]))
    )
    values = np.asarray(spline(probes), dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise RuntimeError("no finite frequency samples in early FFT block")
    return float(np.min(values)), float(np.max(values))


def _centered_even_heterodyne_layer(layer_lo: int, layer_hi: int) -> int:
    """Center a complex baseband while preserving WDM layer parity."""

    center = 0.5 * float(layer_lo + layer_hi)
    return max(0, 2 * int(math.floor(0.5 * center + 0.5)))


def chirplet_lookup_wdm(
    carrier: FEWModeCarrier,
    tdi: TDIGrid,
    channel: str,
    shape: WDMShape,
    taper_width: float,
    time_weight: Callable[[np.ndarray], np.ndarray],
    *,
    track_frequency: np.ndarray | None = None,
    maximum_moment_order: int = 1,
    chirp_rate_max: float = CHIRPLET_LOOKUP_RATE_MAX,
    chirp_rate_step: float = CHIRPLET_LOOKUP_RATE_STEP,
    frequency_step: float = CHIRPLET_LOOKUP_FREQUENCY_STEP,
    amplitude_order: int = 2,
    timing_diagnostics: dict[str, float] | None = None,
    quadrature_output: list[WDMChannel] | None = None,
    frequency_moment_output: dict[int, list[WDMChannel]] | None = None,
) -> tuple[WDMChannel, int, list[FEWEarlyBlockFFTDiagnostic]]:
    """Approximate one slowly evolving carrier directly on WDM pixels.

    The local phase is retained through ``fdot``.  The common endpoint remains
    an exact summed-mode FFT; sharply curved turnover packets should be checked
    against the partitioned-FFT path before this approximation is promoted to
    a production default.
    """

    total_start = time.perf_counter()
    if amplitude_order < 0 or amplitude_order > 2:
        raise ValueError("chirplet amplitude expansion is supported through order 2")
    moment_orders = tuple(
        sorted(int(order) for order in (frequency_moment_output or {}))
    )
    required_order = max(moment_orders, default=0)
    maximum_moment_order = max(maximum_moment_order, required_order)
    if not math.isfinite(chirp_rate_max) or chirp_rate_max <= 0.0:
        raise ValueError("chirplet lookup rate cap must be positive")

    times = np.asarray(tdi.time, dtype=np.float64)
    if track_frequency is None:
        frequency, fdot = tdi_track_frequency(carrier, tdi, channel)
    else:
        frequency = np.asarray(track_frequency, dtype=np.float64)
        if frequency.shape != times.shape:
            raise ValueError("precomputed TDI frequency has the wrong shape")
        fdot = nonuniform_derivative_array(times, frequency)
    sparse_weight = np.asarray(time_weight(times), dtype=np.float64)
    weighted_amplitude = tdi.amplitude[channel] * sparse_weight
    weighted_amplitude *= edge_window(times, shape.Tobs, taper_width)
    amplitude_scale = max(
        float(np.max(np.abs(weighted_amplitude))), np.finfo(float).tiny
    )
    active = (
        np.isfinite(frequency)
        & np.isfinite(fdot)
        & (frequency > 0.0)
        & (sparse_weight > 1.0e-10)
        & (np.abs(weighted_amplitude) > 1.0e-13 * amplitude_scale)
    )
    if np.count_nonzero(active) < 4:
        raise RuntimeError("selected FEW carrier has insufficient lookup support")
    nmid, nsize = wdm_pixels(times[active], np.abs(frequency[active]), shape)

    listn_parts: list[np.ndarray] = []
    listm_parts: list[np.ndarray] = []
    for layer in np.flatnonzero(nsize > 0):
        packet_size = int(nsize[layer])
        output_time = (
            np.arange(packet_size, dtype=np.int64)
            + int(nmid[layer])
            - packet_size // 2
        )
        keep = (output_time >= 0) & (output_time < shape.nt)
        listn_parts.append(output_time[keep])
        listm_parts.append(
            np.full(np.count_nonzero(keep), int(layer), dtype=np.int64)
        )
    if not listn_parts:
        raise RuntimeError("lookup planner produced no active WDM packets")
    listn = np.concatenate(listn_parts)
    listm = np.concatenate(listm_parts)
    unique_n, inverse_n = np.unique(listn, return_inverse=True)
    evaluation_times = unique_n.astype(np.float64) * shape.DT

    amplitude_spline = make_ap_spline(times, weighted_amplitude)
    amplitude_dot_spline = None
    amplitude_ddot_spline = None
    if amplitude_order >= 1:
        amplitude_dot = nonuniform_derivative_array(times, weighted_amplitude)
        amplitude_dot_spline = PchipInterpolator(
            times, amplitude_dot, extrapolate=False
        )
    if amplitude_order >= 2:
        amplitude_ddot = nonuniform_derivative_array(times, amplitude_dot)
        amplitude_ddot_spline = PchipInterpolator(
            times, amplitude_ddot, extrapolate=False
        )
    offset_spline = make_ap_spline(times, tdi.phase_offset[channel])
    frequency_spline = PchipInterpolator(times, frequency, extrapolate=False)
    fdot_spline = PchipInterpolator(times, fdot, extrapolate=False)
    valid_time = (
        (evaluation_times >= times[0])
        & (evaluation_times <= times[-1])
        & (evaluation_times >= 0.0)
        & (evaluation_times <= shape.Tobs)
    )
    clipped_time = np.clip(evaluation_times, times[0], times[-1])
    amplitude = np.asarray(amplitude_spline(clipped_time), dtype=np.float64)
    amplitude[~valid_time] = 0.0
    amplitude_dot_at_center = np.zeros_like(amplitude)
    amplitude_ddot_at_center = np.zeros_like(amplitude)
    if amplitude_dot_spline is not None:
        amplitude_dot_at_center = np.asarray(
            amplitude_dot_spline(clipped_time), dtype=np.float64
        )
        amplitude_dot_at_center[~valid_time] = 0.0
    if amplitude_ddot_spline is not None:
        amplitude_ddot_at_center = np.asarray(
            amplitude_ddot_spline(clipped_time), dtype=np.float64
        )
        amplitude_ddot_at_center[~valid_time] = 0.0
    phase = carrier.phase(clipped_time) + np.asarray(
        offset_spline(clipped_time), dtype=np.float64
    )
    center_frequency = np.asarray(
        frequency_spline(clipped_time), dtype=np.float64
    )
    center_fdot = np.asarray(fdot_spline(clipped_time), dtype=np.float64)

    local_amplitude = amplitude[inverse_n]
    local_amplitude_dot = amplitude_dot_at_center[inverse_n]
    local_amplitude_ddot = amplitude_ddot_at_center[inverse_n]
    local_phase = phase[inverse_n]
    local_frequency = center_frequency[inverse_n]
    local_rate = center_fdot[inverse_n] * shape.Tfilt / shape.DF
    frequency_offset = local_frequency / shape.DF - listm.astype(np.float64)
    phase_phasor = np.exp(1j * local_phase)
    even = ((listn + listm) % 2) == 0
    relevant_rate = (
        np.abs(local_amplitude) > 1.0e-13 * amplitude_scale
    ) & np.isfinite(local_rate)
    required_rate = (
        float(np.max(np.abs(local_rate[relevant_rate])))
        if np.any(relevant_rate)
        else 0.0
    )
    if required_rate > chirp_rate_max:
        raise RuntimeError(
            "chirplet lookup rate cap is too small: "
            f"required |fdot*Tfilt/DF|={required_rate:.6g}, cap="
            f"{chirp_rate_max:.6g}"
        )
    # Slowly evolving tracks generally occupy a tiny fraction of the broad
    # universal safety range.  Retain a symmetric table and one rate-cell of
    # interpolation margin without paying to build unused high-rate rows.
    adaptive_rate_max = max(
        0.5,
        chirp_rate_step
        * math.ceil((required_rate + chirp_rate_step) / chirp_rate_step),
    )
    adaptive_rate_max = min(chirp_rate_max, adaptive_rate_max)
    table_start = time.perf_counter()
    lookup = build_wdm_chirplet_lookup(
        shape,
        maximum_moment_order=maximum_moment_order,
        chirp_rate_max=adaptive_rate_max,
        chirp_rate_step=chirp_rate_step,
        frequency_step=frequency_step,
    )
    table_seconds = time.perf_counter() - table_start
    evaluation_start = time.perf_counter()

    def project(order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        overlap, inside = lookup.interpolate(order, frequency_offset, local_rate)
        local_integral = local_amplitude * overlap
        if amplitude_order >= 1:
            overlap_u, inside_u = lookup.interpolate(
                order,
                frequency_offset,
                local_rate,
                frequency_derivative_order=1,
            )
            local_integral += local_amplitude_dot * overlap_u / (
                1j * 2.0 * PI * shape.DF
            )
            inside &= inside_u
        if amplitude_order >= 2:
            overlap_uu, inside_uu = lookup.interpolate(
                order,
                frequency_offset,
                local_rate,
                frequency_derivative_order=2,
            )
            local_integral += 0.5 * local_amplitude_ddot * overlap_uu / (
                1j * 2.0 * PI * shape.DF
            ) ** 2
            inside &= inside_uu
        packet = phase_phasor * local_integral
        value = np.where(even, packet.real, -packet.imag)
        quadrature = np.where(even, -packet.imag, -packet.real)
        return (
            np.asarray(value, dtype=np.float64),
            np.asarray(quadrature, dtype=np.float64),
            inside,
        )

    values, quadrature_values, inside = project(0)
    relevant_rate &= np.abs(frequency_offset) <= lookup.frequency_offset[-1]
    missed_rate = relevant_rate & ~(
        (local_rate >= lookup.chirp_rate[0])
        & (local_rate <= lookup.chirp_rate[-1])
    )
    if np.any(missed_rate):
        extreme = float(np.max(np.abs(local_rate[missed_rate])))
        raise RuntimeError(
            "chirplet lookup rate range is too small: "
            f"max |fdot*Tfilt/DF|={extreme:.6g}, table limit="
            f"{lookup.chirp_rate[-1]:.6g}"
        )
    values[~inside] = 0.0
    quadrature_values[~inside] = 0.0
    channel_out = WDMChannel(
        np.empty(0), np.empty(0), np.empty(0), nmid.copy(), nsize.copy(),
        listn, listm, values,
    )
    if quadrature_output is not None:
        quadrature_output.append(
            WDMChannel(
                np.empty(0), np.empty(0), np.empty(0),
                nmid.copy(), nsize.copy(), listn.copy(), listm.copy(),
                quadrature_values,
            )
        )
    if frequency_moment_output is not None:
        for order in moment_orders:
            moment, moment_quadrature, moment_inside = project(order)
            moment[~moment_inside] = 0.0
            moment_quadrature[~moment_inside] = 0.0
            frequency_moment_output[order].extend(
                (
                    WDMChannel(
                        np.empty(0), np.empty(0), np.empty(0),
                        nmid.copy(), nsize.copy(), listn.copy(), listm.copy(),
                        moment,
                    ),
                    WDMChannel(
                        np.empty(0), np.empty(0), np.empty(0),
                        nmid.copy(), nsize.copy(), listn.copy(), listm.copy(),
                        moment_quadrature,
                    ),
                )
            )
    if timing_diagnostics is not None:
        timing_diagnostics.update(
            {
                "table_access": table_seconds,
                "table_build": (
                    lookup.build_seconds
                    if table_seconds > 0.5 * lookup.build_seconds
                    else 0.0
                ),
                "evaluate": time.perf_counter() - evaluation_start,
                "total": time.perf_counter() - total_start,
                "table_entries": float(lookup.entries),
                "table_transform_size": float(lookup.transform_size),
                "table_chirp_rate_limit": float(lookup.chirp_rate[-1]),
                "pixel_evaluations": float(listn.size),
                "time_evaluations": float(unique_n.size),
                "chirp_rate_min": (
                    float(np.nanmin(local_rate[relevant_rate]))
                    if np.any(relevant_rate)
                    else 0.0
                ),
                "chirp_rate_max": (
                    float(np.nanmax(local_rate[relevant_rate]))
                    if np.any(relevant_rate)
                    else 0.0
                ),
            }
        )
    return channel_out, 0, []


def partitioned_band_fft_wdm(
    carrier: FEWModeCarrier,
    tdi: TDIGrid,
    channel: str,
    shape: WDMShape,
    taper_width: float,
    time_weight: Callable[[np.ndarray], np.ndarray],
    *,
    bandwidth_hz: float = 1.0e-3,
    roll_seconds: float = EARLY_BLOCK_FFT_ROLL_SECONDS,
    tile_strategy: str = "dyadic",
    endpoint_start: float | None = None,
    endpoint_rise: float = 0.0,
    carrier_index: int = 0,
    track_frequency: np.ndarray | None = None,
    timing_diagnostics: dict[str, float] | None = None,
    quadrature_output: list[WDMChannel] | None = None,
    frequency_moment_output: dict[int, list[WDMChannel]] | None = None,
) -> tuple[WDMChannel, int, list[FEWEarlyBlockFFTDiagnostic]]:
    """Numerical early WDM transform from ungrouped heterodyned FFT blocks.

    Each exact FEW phase carrier receives its own bandwidth-driven time
    partition and even-``DF`` heterodyne. Adjacent raised-cosine windows add
    to unity; their sum is then multiplied by ``time_weight``, which is the
    complement of the common all-mode endpoint window in multimode runs.
    Native bins from every overlapping block are added before the local Meyer
    packet IFFT, so this is an additive partition rather than a handoff.  When
    ``quadrature_output`` is supplied, the same analytic packet also emits the
    WDM coefficients of the minus-sine quadrature.  This avoids a second block
    plan and FFT in polarization-resolved TDI Tapestry calculations.  The
    optional ``frequency_moment_output`` maps positive moment orders to empty
    output lists.  Each list receives the Wilson value and minus-sine
    quadrature for the dimensionless packet moment
    ``((f - f_layer) / DF)**order``.  These moments reuse the block spectra and
    support a local frequency expansion of a detector response.
    """

    if not math.isfinite(bandwidth_hz) or bandwidth_hz <= 2.0 * shape.FB:
        raise ValueError("early FFT bandwidth must exceed the Meyer support width")
    if not math.isfinite(roll_seconds) or roll_seconds <= 0.0:
        raise ValueError("early FFT roll time must be positive")
    if tile_strategy not in {"dyadic", "maximal"}:
        raise ValueError("early FFT tile strategy must be 'dyadic' or 'maximal'")
    moment_orders = tuple(
        sorted(int(order) for order in (frequency_moment_output or {}))
    )
    if any(order < 1 for order in moment_orders):
        raise ValueError("frequency moment orders must be positive")
    if len(set(moment_orders)) != len(moment_orders):
        raise ValueError("frequency moment orders must be unique")

    total_start = time.perf_counter()
    times = np.asarray(tdi.time, dtype=np.float64)
    if track_frequency is None:
        frequency, _ = tdi_track_frequency(carrier, tdi, channel)
    else:
        frequency = np.asarray(track_frequency, dtype=np.float64)
        if frequency.shape != times.shape:
            raise ValueError("precomputed TDI frequency has the wrong shape")
    sparse_weight = np.asarray(time_weight(times), dtype=np.float64)
    amplitude_scale = max(
        float(np.max(np.abs(tdi.amplitude[channel] * sparse_weight))),
        np.finfo(float).tiny,
    )
    active = (
        np.isfinite(frequency)
        & (frequency > 0.0)
        & (sparse_weight > 1.0e-10)
        & (np.abs(tdi.amplitude[channel] * sparse_weight) > 1.0e-13 * amplitude_scale)
    )
    if np.count_nonzero(active) < 4:
        raise RuntimeError("selected FEW carrier has insufficient early FFT support")
    nmid_eval, nsize_eval = wdm_pixels(
        times[active], np.abs(frequency[active]), shape
    )

    finite_frequency = np.isfinite(frequency)
    frequency_spline = PchipInterpolator(
        times[finite_frequency], frequency[finite_frequency], extrapolate=False
    )
    plan_stop = (
        min(float(endpoint_start), float(times[-1]))
        if endpoint_start is not None
        else min(shape.Tobs, float(times[-1]))
    )
    active_hi = min(int(math.floor(plan_stop / shape.DT)), shape.nt)
    if active_hi < 1:
        raise RuntimeError("early FFT partition ends before the first WDM time pixel")
    endpoint_support_stop = (
        min(float(endpoint_start) + max(float(endpoint_rise), 0.0), float(times[-1]))
        if endpoint_start is not None
        else min(shape.Tobs, float(times[-1]))
    )
    endpoint_support_hi = min(
        int(math.ceil(endpoint_support_stop / shape.DT)), shape.nt
    )
    roll_pixels = max(1, int(math.ceil(roll_seconds / shape.DT)))

    plan_start = time.perf_counter()
    tiles: list[tuple[int, int]] = []
    tile_lo = 0
    while tile_lo < active_hi:
        remaining = active_hi - tile_lo
        def width_fits(width: int) -> bool:
            support_lo = max(tile_lo - roll_pixels, 0)
            support_hi = min(tile_lo + width + roll_pixels, endpoint_support_hi)
            lower = float(support_lo) * shape.DT
            upper = float(support_hi) * shape.DT
            fmin, fmax = _track_frequency_range(
                times, frequency, frequency_spline, lower, upper
            )
            return (fmax - fmin) + 2.0 * shape.FB <= bandwidth_hz

        if tile_strategy == "dyadic":
            # Dyadic tiles keep each local FFT comfortably inside cache.  A
            # bandwidth sweep showed that this can beat much fewer, larger
            # blocks even when the dyadic plan evaluates more total samples.
            width = 1 << (remaining.bit_length() - 1)
            selected_width = 0
            while width >= 1:
                if width_fits(width):
                    selected_width = width
                    break
                width //= 2
            selected_width = max(selected_width, 1)
        else:
            # Diagnostic alternative: tile boundaries themselves need not be
            # powers of two because the sampled FFT extent is rounded below.
            # Binary search the longest integer tile satisfying the requested
            # residual bandwidth.
            if width_fits(remaining):
                selected_width = remaining
            else:
                accepted = 0
                lower_width = 1
                upper_width = remaining - 1
                while lower_width <= upper_width:
                    width = (lower_width + upper_width) // 2
                    if width_fits(width):
                        accepted = width
                        lower_width = width + 1
                    else:
                        upper_width = width - 1
                selected_width = max(accepted, 1)
        tile_hi = min(tile_lo + selected_width, active_hi)
        tiles.append((tile_lo, tile_hi))
        tile_lo = tile_hi

    boundaries = np.asarray(
        [float(tile_hi) * shape.DT for _, tile_hi in tiles], dtype=np.float64
    )
    boundary_rolls = np.full(len(tiles), roll_seconds, dtype=np.float64)
    for index in range(max(len(tiles) - 1, 0)):
        next_width = boundaries[index + 1] - boundaries[index]
        endpoint_gap = plan_stop - boundaries[index]
        boundary_rolls[index] = max(
            min(roll_seconds, next_width, endpoint_gap), shape.DT
        )
    plan_seconds = time.perf_counter() - plan_start

    amp_spline = make_ap_spline(times, tdi.amplitude[channel])
    offset_spline = make_ap_spline(times, tdi.phase_offset[channel])
    reference_time_spline = make_ap_spline(times, tdi.detector_time)
    blocks: list[_FEWEarlyBlockSpectrum] = []
    diagnostics: list[FEWEarlyBlockFFTDiagnostic] = []
    total_samples = 0
    build_start = time.perf_counter()
    for index, (tile_lo, tile_hi) in enumerate(tiles):
        nonzero_start = float(times[0]) if index == 0 else boundaries[index - 1]
        if index < len(tiles) - 1:
            nonzero_stop = boundaries[index] + boundary_rolls[index]
        else:
            nonzero_stop = endpoint_support_stop
        nonzero_stop = min(nonzero_stop, float(times[-1]), shape.Tobs)
        if nonzero_stop <= nonzero_start:
            continue
        support_lo = max(int(math.floor(nonzero_start / shape.DT)), 0)
        support_hi = min(int(math.ceil(nonzero_stop / shape.DT)), shape.nt)
        if support_hi <= support_lo:
            support_hi = min(support_lo + 1, shape.nt)
        support_pixels = support_hi - support_lo
        fmin, fmax = _track_frequency_range(
            times, frequency, frequency_spline, nonzero_start, nonzero_stop
        )

        frequency_pad = shape.FB
        taper_scales: list[float] = []
        if index > 0:
            taper_scales.append(float(boundary_rolls[index - 1]))
        if index < len(tiles) - 1:
            taper_scales.append(float(boundary_rolls[index]))
        elif endpoint_start is not None and endpoint_rise > 0.0:
            taper_scales.append(float(endpoint_rise))
        if index == 0 and taper_width > 0.0:
            taper_scales.append(float(taper_width))
        for scale_time in taper_scales:
            frequency_pad = max(
                frequency_pad,
                shape.FB + EARLY_BLOCK_FFT_MARGIN_CYCLES / scale_time,
            )

        layer_lo = max(1, int(math.floor((fmin - frequency_pad) / shape.DF)))
        layer_hi = min(
            shape.nf - 1,
            int(math.ceil((fmax + frequency_pad) / shape.DF)),
        )
        heterodyne_layer = _centered_even_heterodyne_layer(layer_lo, layer_hi)
        shifted_extent = max(
            abs(layer_lo - heterodyne_layer),
            abs(layer_hi - heterodyne_layer),
            1,
        )
        local_bins = next_power_of_two(
            shifted_extent + EARLY_BLOCK_FFT_NYQUIST_GUARD_LAYERS + 1
        )
        local_bins = min(local_bins, shape.nf)
        if index == len(tiles) - 1 and local_bins < shape.nf:
            local_bins *= 2

        maximum_packet_size = 1
        for layer in np.flatnonzero(nsize_eval > 0):
            packet_start = (
                float(nmid_eval[layer]) - 0.5 * float(nsize_eval[layer])
            ) * shape.DT
            packet_stop = packet_start + float(nsize_eval[layer]) * shape.DT
            layer_lower = float(layer) * shape.DF - shape.FB
            layer_upper = float(layer) * shape.DF + shape.FB
            if layer_upper < fmin - frequency_pad or layer_lower > fmax + frequency_pad:
                continue
            if packet_stop <= nonzero_start or packet_start >= nonzero_stop:
                continue
            maximum_packet_size = max(
                maximum_packet_size, int(nsize_eval[layer])
            )
        packet_time_pixels = max(
            next_power_of_two(max(support_pixels, 1)), maximum_packet_size
        )
        sample_dt = shape.DT / float(local_bins)
        sample_count = packet_time_pixels * local_bins
        block_start = float(support_lo) * shape.DT
        sample_times = block_start + sample_dt * np.arange(sample_count)
        live = (
            (sample_times >= nonzero_start)
            & (sample_times <= nonzero_stop)
            & (sample_times >= times[0])
            & (sample_times <= times[-1])
            & (sample_times <= shape.Tobs)
        )
        residual = np.zeros(sample_count, dtype=np.complex128)
        if np.any(live):
            live_times = sample_times[live]
            weight = np.ones(live_times.size, dtype=np.float64)
            if index > 0:
                weight *= plunge_partition(
                    live_times, boundaries[index - 1], boundary_rolls[index - 1]
                )
            if index < len(tiles) - 1:
                weight *= 1.0 - plunge_partition(
                    live_times, boundaries[index], boundary_rolls[index]
                )
            weight *= np.asarray(time_weight(live_times), dtype=np.float64)
            weight *= edge_window(live_times, shape.Tobs, taper_width)
            reference_time = np.asarray(
                reference_time_spline(live_times), dtype=np.float64
            )
            phase = carrier.phase(reference_time) + offset_spline(live_times)
            amplitude = amp_spline(live_times) * weight
            heterodyne_frequency = float(heterodyne_layer) * shape.DF
            residual[live] = amplitude * np.exp(
                1j
                * (
                    phase
                    - 2.0
                    * PI
                    * heterodyne_frequency
                    * (live_times - block_start)
                )
            )
        spectrum = np.fft.fft(residual) * sample_dt
        blocks.append(
            _FEWEarlyBlockSpectrum(
                tile_lo=tile_lo,
                tile_hi=tile_hi,
                support_lo=support_lo,
                support_hi=support_hi,
                block_start=block_start,
                nonzero_start=nonzero_start,
                nonzero_stop=nonzero_stop,
                frequency_min=fmin,
                frequency_max=fmax,
                frequency_pad=frequency_pad,
                heterodyne_layer=heterodyne_layer,
                sample_dt=sample_dt,
                packet_time_pixels=packet_time_pixels,
                spectrum=spectrum,
            )
        )
        diagnostics.append(
            FEWEarlyBlockFFTDiagnostic(
                carrier_index,
                index,
                tile_lo,
                tile_hi,
                support_lo,
                support_hi,
                nonzero_start,
                nonzero_stop,
                fmin,
                fmax,
                frequency_pad,
                heterodyne_layer,
                sample_dt,
                sample_count,
                packet_time_pixels,
            )
        )
        total_samples += sample_count
    build_seconds = time.perf_counter() - build_start
    if not blocks:
        raise RuntimeError("early FFT planner produced no transform blocks")

    packet_start_time = time.perf_counter()
    active_layers = [int(layer) for layer in np.flatnonzero(nsize_eval > 0)]
    groups: dict[int, list[int]] = {}
    for layer in active_layers:
        groups.setdefault(int(nsize_eval[layer]), []).append(layer)
    output_nmid = np.full(shape.nf, -1, dtype=np.int64)
    output_nsize = np.zeros(shape.nf, dtype=np.int64)
    listn: list[np.ndarray] = []
    listm: list[np.ndarray] = []
    values: list[np.ndarray] = []
    quadrature_values: list[np.ndarray] = []
    moment_values: dict[int, list[np.ndarray]] = {
        order: [] for order in moment_orders
    }
    moment_quadrature_values: dict[int, list[np.ndarray]] = {
        order: [] for order in moment_orders
    }
    for packet_size, layers_in_group in groups.items():
        half = packet_size // 2
        base_n = np.arange(packet_size, dtype=np.int64)
        offsets = np.arange(-half, half, dtype=np.float64)
        packet_df = 1.0 / (float(packet_size) * shape.DT)
        dom = 2.0 * PI * packet_df
        window = phitilde(
            np.arange(0, half + 1, dtype=np.float64) * dom, shape
        )[np.abs(base_n - half)]
        window[0] = 0.0
        scale = math.sqrt(8.0 * PI / 15.0) / (
            float(packet_size) * shape.DT
        )
        for layer in layers_in_group:
            center = int(nmid_eval[layer])
            packet_start = (float(center) - float(half)) * shape.DT
            packet_stop = packet_start + float(packet_size) * shape.DT
            fgrid = float(layer) * shape.DF + offsets * packet_df
            data = np.zeros(packet_size, dtype=np.complex128)
            used_block = False
            for block in blocks:
                layer_lower = float(layer) * shape.DF - shape.FB
                layer_upper = float(layer) * shape.DF + shape.FB
                if layer_upper < block.frequency_min - block.frequency_pad:
                    continue
                if layer_lower > block.frequency_max + block.frequency_pad:
                    continue
                if packet_stop <= block.nonzero_start or packet_start >= block.nonzero_stop:
                    continue
                heterodyne_frequency = float(block.heterodyne_layer) * shape.DF
                residual_frequency = fgrid - heterodyne_frequency
                bin_float = residual_frequency * block.spectrum.size * block.sample_dt
                bin_integer = np.rint(bin_float).astype(np.int64)
                valid = (
                    (base_n > 0)
                    & (fgrid > 0.0)
                    & (np.abs(residual_frequency) < 0.5 / block.sample_dt)
                    & (np.abs(bin_float - bin_integer.astype(np.float64)) < 1.0e-7)
                )
                if not np.any(valid):
                    continue
                phase_rotation = 2.0 * PI * fgrid[valid] * (
                    shape.Tobs + packet_start - block.block_start
                )
                data[valid] += block.spectrum[
                    bin_integer[valid] % block.spectrum.size
                ] * np.exp(1j * phase_rotation)
                used_block = True
            if not used_block:
                continue
            packet = np.fft.ifft(data * window) * packet_size
            moment_packets: dict[int, np.ndarray] = {}
            if moment_orders:
                normalized_frequency = offsets * packet_df / shape.DF
                weighted = np.asarray(
                    [
                        data * window * normalized_frequency**order
                        for order in moment_orders
                    ]
                )
                transformed = np.fft.ifft(weighted, axis=1) * packet_size
                moment_packets = {
                    order: transformed[row]
                    for row, order in enumerate(moment_orders)
                }
            even = ((base_n + layer) % 2) == 0
            imag_sign = 1.0 if layer % 2 == 0 else -1.0
            coefficients = scale * np.where(
                even, packet.real, imag_sign * packet.imag
            )
            if quadrature_output is not None:
                quadrature_coefficients = scale * np.where(
                    even, -packet.imag, imag_sign * packet.real
                )
            moment_coefficients: dict[int, np.ndarray] = {}
            moment_quadrature_coefficients: dict[int, np.ndarray] = {}
            for order, moment_packet in moment_packets.items():
                moment_coefficients[order] = scale * np.where(
                    even, moment_packet.real, imag_sign * moment_packet.imag
                )
                moment_quadrature_coefficients[order] = scale * np.where(
                    even, -moment_packet.imag, imag_sign * moment_packet.real
                )
            output_time = base_n + center - half
            keep = (output_time >= 0) & (output_time < shape.nt)
            listn.append(output_time[keep])
            listm.append(np.full(np.count_nonzero(keep), layer, dtype=np.int64))
            values.append(np.asarray(coefficients[keep], dtype=np.float64))
            if quadrature_output is not None:
                quadrature_values.append(
                    np.asarray(quadrature_coefficients[keep], dtype=np.float64)
                )
            for order in moment_orders:
                moment_values[order].append(
                    np.asarray(moment_coefficients[order][keep], dtype=np.float64)
                )
                moment_quadrature_values[order].append(
                    np.asarray(
                        moment_quadrature_coefficients[order][keep],
                        dtype=np.float64,
                    )
                )
            output_nmid[layer] = center
            output_nsize[layer] = packet_size
    packet_seconds = time.perf_counter() - packet_start_time
    if not values:
        raise RuntimeError("early FFT blocks did not overlap any active WDM packets")
    total_seconds = time.perf_counter() - total_start
    if timing_diagnostics is not None:
        timing_diagnostics.update(
            {
                "plan": plan_seconds,
                "fft_build": build_seconds,
                "packet_wdm": packet_seconds,
                # Track/support construction and spline setup sit outside the
                # three explicitly timed stages.  This residual makes the
                # fixed cost of using many small blocks visible in planner
                # sweeps without perturbing the hot loops with more timers.
                "setup": max(
                    0.0,
                    total_seconds - plan_seconds - build_seconds - packet_seconds,
                ),
                "total": total_seconds,
                "blocks": float(len(blocks)),
                "samples": float(total_samples),
            }
        )
    channel_out = WDMChannel(
        np.empty(0),
        np.empty(0),
        np.empty(0),
        output_nmid,
        output_nsize,
        np.concatenate(listn),
        np.concatenate(listm),
        np.concatenate(values),
    )
    if quadrature_output is not None:
        quadrature_output.append(
            WDMChannel(
                np.empty(0),
                np.empty(0),
                np.empty(0),
                output_nmid.copy(),
                output_nsize.copy(),
                channel_out.listn.copy(),
                channel_out.listm.copy(),
                np.concatenate(quadrature_values),
            )
        )
    if frequency_moment_output is not None:
        for order in moment_orders:
            value_channel = WDMChannel(
                np.empty(0),
                np.empty(0),
                np.empty(0),
                output_nmid.copy(),
                output_nsize.copy(),
                channel_out.listn.copy(),
                channel_out.listm.copy(),
                np.concatenate(moment_values[order]),
            )
            quadrature_channel = WDMChannel(
                np.empty(0),
                np.empty(0),
                np.empty(0),
                output_nmid.copy(),
                output_nsize.copy(),
                channel_out.listn.copy(),
                channel_out.listm.copy(),
                np.concatenate(moment_quadrature_values[order]),
            )
            frequency_moment_output[order].extend(
                (value_channel, quadrature_channel)
            )
    return (
        channel_out,
        total_samples,
        diagnostics,
    )


def turnover_partition_wdm(
    carrier: FEWModeCarrier,
    tdi: TDIGrid,
    channel: str,
    shape: WDMShape,
    taper_width: float,
    time_weight: Callable[[np.ndarray], np.ndarray],
    *,
    packet_pixels: float = 48.0,
    packet_taper_pixels: float = 4.0,
    diagnostic_parts: dict[str, list[WDMChannel]] | None = None,
    timing_diagnostics: dict[str, float] | None = None,
) -> tuple[WDMChannel, list[FEWTurnoverRegion], int]:
    """SPA monotone branches with local FFT replacement at each turnover.

    The packet is not added as a time-domain partition.  Such a partition
    gives the SPA wings artificial taper edges whose Fourier ripples are not
    represented by leading-order SPA.  Instead, each monotone branch is
    transformed as usual and the small set of WDM pixels whose compact Meyer
    support straddles the turnover is replaced by the local FFT result.
    """

    total_start = time.perf_counter()
    stage_start = time.perf_counter()
    regions, frequency, fdot = find_turnover_regions(
        carrier,
        tdi,
        channel,
        shape,
        time_weight,
        packet_pixels=packet_pixels,
        taper_pixels=packet_taper_pixels,
    )
    search_seconds = time.perf_counter() - stage_start
    if not regions:
        active = (
            np.asarray(time_weight(tdi.time), dtype=np.float64) > 1.0e-10
        ) & np.isfinite(fdot)
        branch_sign = float(np.sign(np.median(fdot[active]))) if np.any(active) else 1.0
        if branch_sign == 0.0:
            branch_sign = 1.0
        stage_start = time.perf_counter()
        result, _, _ = numerical_spa_wdm(
            carrier,
            tdi,
            channel,
            shape,
            taper_width,
            time_weight=time_weight,
            fdot_sign=branch_sign,
        )
        if timing_diagnostics is not None:
            timing_diagnostics.update(
                {
                    "turnover_search": search_seconds,
                    "pure_spa": time.perf_counter() - stage_start,
                    "turnover_fft": 0.0,
                    "turnover_spa": 0.0,
                    "combine": 0.0,
                    "total": time.perf_counter() - total_start,
                }
            )
        return result, [], 0

    packet_parts: list[WDMChannel] = []
    spa_parts: list[WDMChannel] = []
    fft_samples = 0
    stage_start = time.perf_counter()
    for region in regions:
        packet, samples = local_turnover_fft_wdm(
            carrier,
            tdi,
            channel,
            shape,
            taper_width,
            region,
            time_weight,
            packet_taper_pixels=packet_taper_pixels,
        )
        # The packet has a flat center surrounded by a raised-cosine taper.
        # Retain only coefficients whose truncated Meyer time support lies
        # wholly inside that flat center; there the packet coefficient is the
        # coefficient of the unwindowed signal to numerical precision.
        safe_margin = (packet_taper_pixels + float(shape.mult)) * shape.DT
        safe_start = region.start + safe_margin
        safe_stop = region.stop - safe_margin
        keep = (
            packet.listn.astype(np.float64) * shape.DT >= safe_start
        ) & (
            packet.listn.astype(np.float64) * shape.DT <= safe_stop
        )
        packet = WDMChannel(
            packet.freq,
            packet.phase,
            packet.amplitude,
            packet.nmid,
            packet.nsize,
            packet.listn[keep],
            packet.listm[keep],
            packet.values[keep],
        )
        packet_parts.append(packet)
        if diagnostic_parts is not None:
            diagnostic_parts.setdefault("turnover", []).append(packet)
        fft_samples += samples
    turnover_fft_seconds = time.perf_counter() - stage_start

    boundaries = [float(tdi.time[0])]
    boundaries.extend(region.center for region in regions)
    boundaries.append(float(tdi.time[-1]))
    stage_start = time.perf_counter()
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        if upper <= lower:
            continue

        def branch_weight(
            values: np.ndarray, lo: float = lower, hi: float = upper
        ) -> np.ndarray:
            values = np.asarray(values, dtype=np.float64)
            out = np.asarray(time_weight(values), dtype=np.float64)
            out[(values < lo) | (values > hi)] = 0.0
            return out

        node_weight = branch_weight(tdi.time)
        use = (
            (node_weight > 1.0e-10)
            & np.isfinite(fdot)
            & np.isfinite(frequency)
        )
        if np.count_nonzero(use) < 4:
            continue
        branch_sign = float(np.sign(np.median(fdot[use])))
        if branch_sign == 0.0:
            indices = np.flatnonzero(use)
            branch_sign = float(
                np.sign(frequency[indices[-1]] - frequency[indices[0]])
            )
        if branch_sign == 0.0:
            continue
        try:
            branch, _, _ = numerical_spa_wdm(
                carrier,
                tdi,
                channel,
                shape,
                taper_width,
                time_weight=branch_weight,
                time_bounds=(lower, upper),
                fdot_sign=branch_sign,
            )
        except RuntimeError:
            continue
        spa_parts.append(branch)
        if diagnostic_parts is not None:
            diagnostic_parts.setdefault("spa", []).append(branch)
    turnover_spa_seconds = time.perf_counter() - stage_start

    if not spa_parts and not packet_parts:
        raise RuntimeError("turnover partition produced no WDM packets")

    # Add the stationary points from all monotone branches first.  Where a
    # local FFT is valid, overwrite that coefficient rather than adding it:
    # the FFT evaluates the full carrier through the turnover and already
    # contains both stationary contributions when both exist.
    stage_start = time.perf_counter()
    listn, listm, values = combine_sparse_wdm(
        shape,
        [part.listn for part in spa_parts],
        [part.listm for part in spa_parts],
        [part.values for part in spa_parts],
    )
    packet_n, packet_m, packet_values = combine_sparse_wdm(
        shape,
        [part.listn for part in packet_parts],
        [part.listm for part in packet_parts],
        [part.values for part in packet_parts],
    )
    if packet_values.size:
        linear = listn * (shape.nf + 1) + listm
        packet_linear = packet_n * (shape.nf + 1) + packet_m
        replace = np.isin(linear, packet_linear, assume_unique=True)
        keep = ~replace
        listn = np.concatenate((listn[keep], packet_n))
        listm = np.concatenate((listm[keep], packet_m))
        values = np.concatenate((values[keep], packet_values))
        order = np.argsort(listn * (shape.nf + 1) + listm)
        listn = listn[order]
        listm = listm[order]
        values = values[order]

    nmid = np.full(shape.nf, -1, dtype=np.int64)
    nsize = np.zeros(shape.nf, dtype=np.int64)
    for part in spa_parts + packet_parts:
        merge_pixel_plans(nmid, nsize, part.nmid, part.nsize, shape)
    combine_seconds = time.perf_counter() - stage_start
    if timing_diagnostics is not None:
        timing_diagnostics.update(
            {
                "turnover_search": search_seconds,
                "pure_spa": 0.0,
                "turnover_fft": turnover_fft_seconds,
                "turnover_spa": turnover_spa_seconds,
                "combine": combine_seconds,
                "total": time.perf_counter() - total_start,
            }
        )
    return (
        WDMChannel(
            np.empty(0),
            np.empty(0),
            np.empty(0),
            nmid,
            nsize,
            listn,
            listm,
            values,
        ),
        regions,
        fft_samples,
    )


def normalized_match(reference: np.ndarray, candidate: np.ndarray) -> float:
    denominator = math.sqrt(
        float(np.vdot(reference, reference).real)
        * float(np.vdot(candidate, candidate).real)
    )
    if denominator == 0.0:
        return float("nan")
    return float(abs(np.vdot(reference, candidate)) / denominator)


def sparse_values_on_coordinates(
    reference_n: np.ndarray,
    reference_m: np.ndarray,
    reference_values: np.ndarray,
    query_n: np.ndarray,
    query_m: np.ndarray,
    shape: WDMShape,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted sparse reference values at query pixels."""

    reference_linear = reference_n * (shape.nf + 1) + reference_m
    query_linear = query_n * (shape.nf + 1) + query_m
    locations = np.searchsorted(reference_linear, query_linear)
    present = locations < reference_linear.size
    present_indices = np.flatnonzero(present)
    present[present_indices] = (
        reference_linear[locations[present_indices]] == query_linear[present_indices]
    )
    output = np.zeros(query_linear.size, dtype=np.float64)
    output[present] = reference_values[locations[present]]
    return output, present


def sparse_wdm_from_global_rfft(
    spectrum: np.ndarray,
    nmid: np.ndarray,
    nsize: np.ndarray,
    shape: WDMShape,
    quadrature_output: list[WDMChannel] | None = None,
) -> WDMChannel:
    """Evaluate selected Meyer packets from one full-duration real FFT.

    ``spectrum`` uses the positive-frequency convention
    ``2*dt*rfft(h)``.  Packet frequencies are exact bins of the global FFT,
    so no Fourier interpolation is required.  This provides the direct sparse
    WDM reference for the partitioned, heterodyned block-FFT path.  When
    ``quadrature_output`` is supplied, the same analytic packets also emit the
    minus-sine WDM quadrature without another FFT.
    """

    expected = shape.n // 2 + 1
    spectrum = np.asarray(spectrum, dtype=np.complex128)
    if spectrum.ndim != 1 or spectrum.size != expected:
        raise ValueError(
            f"global real FFT has {spectrum.size} bins; expected {expected}"
        )

    active_layers = [
        int(layer)
        for layer in np.flatnonzero((nmid >= 0) & (nsize > 0))
        if 0 < layer < shape.nf
    ]
    groups: dict[int, list[int]] = {}
    for layer in active_layers:
        groups.setdefault(int(nsize[layer]), []).append(layer)

    listn: list[np.ndarray] = []
    listm: list[np.ndarray] = []
    values: list[np.ndarray] = []
    quadrature_values: list[np.ndarray] = []
    for packet_size, layers in groups.items():
        half = packet_size // 2
        base_n = np.arange(packet_size, dtype=np.int64)
        offsets = np.arange(-half, half, dtype=np.float64)
        packet_df = 1.0 / (float(packet_size) * shape.DT)
        dom = 2.0 * PI * packet_df
        window = phitilde(
            np.arange(0, half + 1, dtype=np.float64) * dom, shape
        )[np.abs(base_n - half)]
        window[0] = 0.0
        scale = math.sqrt(8.0 * PI / 15.0) / (
            float(packet_size) * shape.DT
        )

        layer_array = np.asarray(layers, dtype=np.int64)
        fgrid = (
            layer_array[:, np.newaxis].astype(np.float64) * shape.DF
            + offsets[np.newaxis, :] * packet_df
        )
        bins_float = fgrid * shape.Tobs
        bins = np.rint(bins_float).astype(np.int64)
        valid = (
            (base_n[np.newaxis, :] > 0)
            & (fgrid > 0.0)
            & (bins > 0)
            & (bins < spectrum.size)
            & (np.abs(bins_float - bins.astype(np.float64)) < 1.0e-7)
        )
        data = np.zeros(fgrid.shape, dtype=np.complex128)
        start_times = (
            nmid[layer_array].astype(np.float64) - float(half)
        ) * shape.DT
        phase = 2.0 * PI * fgrid * (
            shape.Tobs + start_times[:, np.newaxis]
        )
        data[valid] = spectrum[bins[valid]] * np.exp(1j * phase[valid])

        packet = np.fft.ifft(data * window[np.newaxis, :], axis=1) * packet_size
        even = ((base_n[np.newaxis, :] + layer_array[:, np.newaxis]) % 2) == 0
        imag_sign = np.where((layer_array % 2) == 0, 1.0, -1.0)[:, np.newaxis]
        coefficients = scale * np.where(
            even, packet.real, imag_sign * packet.imag
        )
        if quadrature_output is not None:
            quadrature_coefficients = scale * np.where(
                even, -packet.imag, imag_sign * packet.real
            )
        for row, layer in enumerate(layers):
            output_time = base_n + int(nmid[layer]) - half
            keep = (output_time >= 0) & (output_time < shape.nt)
            listn.append(output_time[keep])
            listm.append(
                np.full(np.count_nonzero(keep), layer, dtype=np.int64)
            )
            values.append(np.asarray(coefficients[row, keep], dtype=np.float64))
            if quadrature_output is not None:
                quadrature_values.append(
                    np.asarray(quadrature_coefficients[row, keep], dtype=np.float64)
                )

    if not values:
        empty = empty_wdm_channel(shape)
        if quadrature_output is not None:
            quadrature_output.append(empty_wdm_channel(shape))
        return empty
    channel_out = WDMChannel(
        np.empty(0),
        np.empty(0),
        np.empty(0),
        nmid.copy(),
        nsize.copy(),
        np.concatenate(listn),
        np.concatenate(listm),
        np.concatenate(values),
    )
    if quadrature_output is not None:
        quadrature_output.append(
            WDMChannel(
                np.empty(0),
                np.empty(0),
                np.empty(0),
                nmid.copy(),
                nsize.copy(),
                channel_out.listn.copy(),
                channel_out.listm.copy(),
                np.concatenate(quadrature_values),
            )
        )
    return channel_out


def empty_wdm_channel(shape: WDMShape) -> WDMChannel:
    empty_i = np.empty(0, dtype=np.int64)
    empty = np.empty(0, dtype=np.float64)
    return WDMChannel(
        empty,
        empty,
        empty,
        np.full(shape.nf, -1, dtype=np.int64),
        np.zeros(shape.nf, dtype=np.int64),
        empty_i,
        empty_i.copy(),
        empty,
    )


def plunge_partition(
    times: np.ndarray, start: float, rise: float
) -> np.ndarray:
    """Late-time member of a smooth partition of unity."""

    values = np.asarray(times, dtype=np.float64)
    out = np.zeros_like(values)
    if rise <= 0.0:
        out[values >= start] = 1.0
        return out
    transition = (values > start) & (values < start + rise)
    out[values >= start + rise] = 1.0
    x = (values[transition] - start) / rise
    out[transition] = 0.5 * (1.0 - np.cos(PI * x))
    return out


def reconstruct_mode_set_sparse_tdi(
    collection: FEWModeCollection,
    tdi_modes: Sequence[TDIGrid],
    channel: str,
    times: np.ndarray,
    p_splines: list[CubicSpline],
    chunk_size: int = 100000,
) -> np.ndarray:
    """Reconstruct the summed sparse-AP response without dense FEW calls."""

    times = np.asarray(times, dtype=np.float64)
    output = np.empty(times.size)
    amp_splines = [make_ap_spline(tdi.time, tdi.amplitude[channel]) for tdi in tdi_modes]
    offset_splines = [make_ap_spline(tdi.time, tdi.phase_offset[channel]) for tdi in tdi_modes]
    for first in range(0, times.size, chunk_size):
        last = min(first + chunk_size, times.size)
        local = times[first:last]
        reference = spacecraft0_reference_time(local, collection.source, p_splines)
        phases = collection.phase_matrix(reference)
        value = np.zeros(local.size)
        for index, (amp_spline, offset_spline) in enumerate(
            zip(amp_splines, offset_splines)
        ):
            value += amp_spline(local) * np.cos(
                phases[:, index] + offset_spline(local)
            )
        output[first:last] = value
    return output


def _one_block_plan(
    start_time: float,
    stop_time: float,
    shape: WDMShape,
    maximum_frequency: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one power-of-two time packet for the requested frequency band."""

    jlo = max(int(math.floor(start_time / shape.DT)) - shape.mult, 0)
    jhi = min(int(math.ceil(stop_time / shape.DT)) + shape.mult, shape.nt - 1)
    needed = max(jhi - jlo + 3, 2 * shape.mult)
    block = min(next_power_of_two(needed), shape.nt)
    center = (jlo + jhi + 1) // 2
    if center % 2:
        center -= 1
    center = max(center, block // 2)
    center = min(center, shape.nt - block // 2)
    if center % 2:
        center -= 1
    while center - block // 2 > jlo:
        center -= 2
    while center + block // 2 - 1 < jhi:
        center += 2
    center = max(center, block // 2)
    center = min(center, shape.nt - block // 2)
    if center % 2:
        center -= 1
    nmid = np.full(shape.nf, -1, dtype=np.int64)
    nsize = np.zeros(shape.nf, dtype=np.int64)
    if maximum_frequency is None:
        last_layer = shape.nf - 1
    else:
        last_layer = min(
            int(math.ceil((maximum_frequency + shape.FB) / shape.DF)),
            shape.nf - 1,
        )
    if last_layer >= 1:
        nmid[1 : last_layer + 1] = center
        nsize[1 : last_layer + 1] = block
    return nmid, nsize


def common_plunge_wdm(
    collection: FEWModeCollection,
    channel: str,
    shape: WDMShape,
    l_splines: list[CubicSpline],
    p_splines: list[CubicSpline],
    v_splines: list[CubicSpline],
    partition_start: float,
    partition_rise: float,
    taper_width: float,
    maximum_physical_frequency: float,
    spectral_power_tolerance: float = 1.0e-10,
    dense_chunk: int = 100000,
    tdi_generation: int = 1,
) -> tuple[WDMChannel, np.ndarray, np.ndarray, dict[str, float]]:
    """Transform the completed endpoint through one compact native short FFT."""

    if not 0.0 < spectral_power_tolerance < 1.0:
        raise ValueError("spectral_power_tolerance must lie between zero and one")
    transition_bandwidth = 2.0 / max(partition_rise, collection.mass_seconds)
    damping_bandwidth = 0.0
    if collection.ringdown_groups:
        damping_bandwidth = max(
            group.maximum_damping for group in collection.ringdown_groups
        ) / PI
    estimated_stop = maximum_physical_frequency + transition_bandwidth + damping_bandwidth
    estimated_stop = min(max(estimated_stop, 4.0 * shape.DF), 0.5 / shape.dt)

    samples_per_pixel = next_power_of_two(
        max(1, int(math.ceil(4.0 * shape.DT * (estimated_stop + 2.0 * shape.FB))))
    )
    sample_dt = shape.DT / float(samples_per_pixel)
    sample_dt = max(sample_dt, shape.dt)
    if collection.endpoint_model == "hard":
        sample_dt = shape.dt
        estimated_stop = 0.5 / sample_dt

    tes = math.floor(partition_start / sample_dt) * sample_dt
    _, ringdown_output_stop = tdi_source_event_output_bounds(
        collection.ringdown_stop,
        collection.source,
        channel,
        l_splines,
        p_splines,
        v_splines,
        tdi_generation=tdi_generation,
    )
    response_stop = min(ringdown_output_stop, shape.Tobs)
    sample_count = int(math.ceil((response_stop - tes) / sample_dt)) + 1
    sample_times = tes + sample_dt * np.arange(sample_count, dtype=np.float64)
    strain = sampled_sum_tdi_response(
        collection,
        sample_times,
        channel,
        sample_dt,
        l_splines,
        p_splines,
        v_splines,
        chunk_size=dense_chunk,
        tdi_generation=tdi_generation,
    )
    strain *= plunge_partition(sample_times, partition_start, partition_rise)
    strain *= edge_window(sample_times, shape.Tobs, taper_width)

    scale = max(float(np.max(np.abs(strain))), np.finfo(float).tiny)
    live = np.flatnonzero(np.abs(strain) > 1.0e-13 * scale)
    live_stop = sample_times[live[-1]] if live.size else collection.ringdown_stop

    frequency_stop = estimated_stop
    if collection.endpoint_model == "qnm" and np.any(strain):
        spectrum = np.fft.rfft(strain)
        power = np.abs(spectrum) ** 2
        frequencies = np.fft.rfftfreq(strain.size, sample_dt)
        total_power = float(np.sum(power))
        if total_power > 0.0:
            tail = np.cumsum(power[::-1])[::-1]
            above = np.flatnonzero(tail > spectral_power_tolerance * total_power)
            if above.size:
                frequency_stop = min(
                    max(float(frequencies[above[-1]]) + 2.0 * shape.FB, shape.DF),
                    0.5 / sample_dt,
                )
    nmid, nsize = _one_block_plan(
        partition_start, live_stop, shape, maximum_frequency=frequency_stop
    )
    dense = np.zeros((shape.nt, shape.nf + 1), dtype=np.float64)
    setup = np.array([sample_dt, float(sample_count), tes], dtype=np.float64)
    apply_short_fft_threshold(
        dense,
        nmid,
        nsize,
        strain,
        setup,
        -shape.FB,
        frequency_stop,
        shape,
    )
    result = wdm_channel_from_dense(
        np.empty(0),
        np.empty(0),
        np.empty(0),
        nmid,
        nsize,
        dense,
        shape,
    )
    diagnostics = {
        "sample_dt_seconds": sample_dt,
        "sample_count": float(sample_count),
        "frequency_stop_hz": frequency_stop,
        "frequency_layers": float(np.count_nonzero(nsize)),
        "live_stop_seconds": live_stop,
    }
    return result, sample_times, strain, diagnostics


def validate_modes(
    source: FEWSourceParams,
    shape: WDMShape,
    channel: str = "X",
    *,
    mode_selection: str | Sequence[Mode] = "threshold",
    mode_selection_threshold: float = 1.0e-3,
    taper_pixels: float = 8.0,
    endpoint_model: str = "qnm",
    plunge_pre_m: float = 200.0,
    plunge_rise_m: float = 50.0,
    plunge_pre_pixels: float | None = None,
    plunge_rise_pixels: float | None = None,
    qnm_transition_m: float = 10.0,
    qnm_efolds: float = 8.0,
    qnm_fade_efolds: float = 2.0,
    endpoint_spectral_power_tolerance: float = 1.0e-10,
    tdi_maximum_step: float = 1.0e4,
    dense_chunk: int = 100000,
    fast_method: str = "auto",
    early_fft_bandwidth_hz: float = 1.0e-3,
    early_fft_roll_seconds: float = EARLY_BLOCK_FFT_ROLL_SECONDS,
    early_fft_tile_strategy: str = "dyadic",
    direct_validation: bool = True,
    global_fft_reference: bool = False,
    tdi_generation: int = 1,
) -> FEWMultiModeValidationResult:
    """Generate selected FEW modes, optionally including the dense reference."""

    if shape.nt < 2 or shape.nt & (shape.nt - 1):
        raise ValueError("WDM nt must be a power of two")
    if shape.nf < 2 or shape.nf & (shape.nf - 1):
        raise ValueError("WDM nf must be a power of two")
    if tdi_maximum_step <= 0.0:
        raise ValueError("TDI maximum step must be positive")
    if tdi_generation not in (1, 2):
        raise ValueError("tdi_generation must be 1 or 2")
    if fast_method not in {
        "auto",
        "spa",
        "turnover-packets",
        "band-fft",
        "block-fft",
    }:
        raise ValueError(f"unknown FEW fast transform method {fast_method!r}")
    fast_generation_start = time.perf_counter()
    timings: dict[str, float] = {}
    taper_width = taper_pixels * shape.DT
    mass_seconds = source.mass_solar * MTSUN_SI
    slowest_damping = min(
        abs(kerr_qnm_fundamental(2, m, source.spin).imag)
        for m in range(-2, 3)
    ) / mass_seconds
    ringdown_extent = max(qnm_transition_m * mass_seconds, qnm_efolds / slowest_damping)
    ringdown_extent += qnm_fade_efolds / slowest_damping
    if endpoint_model == "hard":
        ringdown_extent = 0.0
    endpoint_delay_margin = (800.0 if tdi_generation == 2
                             else TDI_ENDPOINT_DELAY_MARGIN_SECONDS)
    endpoint_padding = taper_width + ringdown_extent + endpoint_delay_margin + 100.0

    few_runtime_diagnostics: dict[str, float] = {}
    start = time.perf_counter()
    collection = build_few_mode_collection(
        source,
        shape,
        mode_selection=mode_selection,
        mode_selection_threshold=mode_selection_threshold,
        endpoint_padding=endpoint_padding,
        endpoint_model=endpoint_model,
        qnm_transition_m=qnm_transition_m,
        qnm_efolds=qnm_efolds,
        qnm_fade_efolds=qnm_fade_efolds,
        runtime_diagnostics=few_runtime_diagnostics,
    )
    collection_seconds = time.perf_counter() - start
    initialization_seconds = few_runtime_diagnostics["initialization_seconds"]
    timings["few_model_initialization"] = initialization_seconds
    timings["few_sparse_intrinsic_and_mode_selection"] = max(
        0.0, collection_seconds - initialization_seconds
    )
    timings["few_collection_total"] = collection_seconds

    start = time.perf_counter()
    _, l_splines, p_splines, v_splines = build_constellation_splines(shape)
    timings["constellation_setup"] = time.perf_counter() - start

    response_endpoint_start, response_endpoint_stop = tdi_source_event_output_bounds(
        collection.plunge_time,
        collection.source,
        channel,
        l_splines,
        p_splines,
        v_splines,
        tdi_generation=tdi_generation,
    )
    endpoint_partition_minimum = 2.0 * float(shape.mult) * shape.DT
    if plunge_pre_pixels is None:
        # The common endpoint FFT must turn on slowly enough for the early
        # SPA complement to remain smooth on the compact Meyer time support.
        # A purely mass-scaled 50M rise is only a few hundred seconds for the
        # reference EMRI and leaves a visible endpoint scar.  Two Meyer
        # half-supports (16 WDM time pixels for the default window) restore the
        # reference accuracy while keeping this one all-mode FFT compact.
        plunge_pre = max(
            plunge_pre_m * collection.mass_seconds,
            endpoint_partition_minimum,
        )
    else:
        plunge_pre = plunge_pre_pixels * shape.DT
    if plunge_rise_pixels is None:
        plunge_rise = max(
            plunge_rise_m * collection.mass_seconds,
            endpoint_partition_minimum,
        )
    else:
        plunge_rise = plunge_rise_pixels * shape.DT
    partition_start = max(0.0, response_endpoint_start - plunge_pre)
    partition_rise = min(
        plunge_rise,
        max(response_endpoint_start - partition_start, 0.0),
    )
    partition_stop = partition_start + partition_rise

    start = time.perf_counter()
    planner_diagnostics: dict[str, float] = {}
    tdi_times = build_shared_tdi_time_grid(
        collection,
        shape,
        p_splines,
        maximum_step=tdi_maximum_step,
        taper_width=taper_width,
        planning_stop=partition_stop,
        diagnostics=planner_diagnostics,
        l_splines=l_splines,
        channels=(channel,),
        tdi_generation=tdi_generation,
    )
    if partition_rise > 0.0:
        # The partition is defined in SSB output time.  Add a modest, fixed-size
        # sampling scaffold only across its smooth rise; delayed waveform
        # arguments remain source-time quantities inside the TDI evaluator.  A dense
        # pre-endpoint guard changes finite-difference classifications at its
        # junction and was less accurate than this local construction.
        transition = np.linspace(partition_start, partition_stop, 65)
        tdi_times = np.unique(
            np.concatenate((tdi_times, transition, np.array([partition_stop])))
        )
    timings["shared_tdi_grid_plan"] = time.perf_counter() - start

    start = time.perf_counter()
    tdi_modes = extract_mode_set_tdi_ap(
        collection,
        tdi_times,
        (channel,),
        l_splines,
        p_splines,
        v_splines,
        tdi_generation=tdi_generation,
    )
    timings["sparse_tdi_all_modes"] = time.perf_counter() - start

    def early_weight(values: np.ndarray) -> np.ndarray:
        return 1.0 - plunge_partition(values, partition_start, partition_rise)

    # ``partition_start`` is an SSB output time, whereas the intrinsic FEW
    # frequency evaluator takes source times.  Probe a source-time
    # interval wide enough to include every delayed endpoint contribution.
    source_probe_start = max(
        0.0,
        collection.plunge_time
        - plunge_pre
        - TDI_ENDPOINT_DELAY_MARGIN_SECONDS,
    )
    plunge_probe = np.linspace(source_probe_start, collection.plunge_time, 257)
    plunge_frequency_by_mode = np.max(
        np.abs(collection.frequency_matrix(plunge_probe)), axis=0
    )
    plunge_driver_index = int(np.argmax(plunge_frequency_by_mode))
    plunge_driver_frequency = float(plunge_frequency_by_mode[plunge_driver_index])
    nyquist = 0.5 / shape.dt
    if plunge_driver_frequency >= nyquist:
        driver = collection.modes[plunge_driver_index]
        raise ValueError(
            f"mode {driver} reaches {plunge_driver_frequency:.6e} Hz, "
            f"above the {nyquist:.6e} Hz WDM Nyquist frequency"
        )
    if collection.maximum_qnm_frequency >= nyquist:
        raise ValueError(
            f"the endpoint QNM catalog reaches "
            f"{collection.maximum_qnm_frequency:.6e} Hz, above the "
            f"{nyquist:.6e} Hz WDM Nyquist frequency"
        )

    early_parts: list[WDMChannel] = []
    mode_methods: list[str] = []
    early_frequency_maxima: list[float] = []
    early_fft_samples: list[int] = []
    early_block_fft_diagnostics: list[FEWEarlyBlockFFTDiagnostic] = []
    turnover_region_count = 0
    early_stage_seconds = {
        "turnover_search": 0.0,
        "pure_spa": 0.0,
        "turnover_fft": 0.0,
        "turnover_spa": 0.0,
        "combine": 0.0,
        "fallback": 0.0,
        "block_plan": 0.0,
        "block_fft_build": 0.0,
        "block_packet_wdm": 0.0,
        "block_setup": 0.0,
        "block_total": 0.0,
    }
    start = time.perf_counter()
    for carrier_index, (carrier, tdi) in enumerate(
        zip(collection.carriers, tdi_modes)
    ):
        track_frequency, _ = tdi_track_frequency(carrier, tdi, channel)
        active = early_weight(tdi.time) > 1.0e-10
        finite_active = active & np.isfinite(track_frequency)
        early_frequency_maxima.append(
            float(np.max(np.abs(track_frequency[finite_active])))
            if np.any(finite_active)
            else 0.0
        )
        if fast_method == "block-fft":
            mode_timing = {}
            part, fft_samples, block_diagnostics = partitioned_band_fft_wdm(
                carrier,
                tdi,
                channel,
                shape,
                taper_width,
                early_weight,
                bandwidth_hz=early_fft_bandwidth_hz,
                roll_seconds=early_fft_roll_seconds,
                tile_strategy=early_fft_tile_strategy,
                endpoint_start=partition_start,
                endpoint_rise=partition_rise,
                carrier_index=carrier_index,
                track_frequency=track_frequency,
                timing_diagnostics=mode_timing,
            )
            early_stage_seconds["block_plan"] += mode_timing.get("plan", 0.0)
            early_stage_seconds["block_fft_build"] += mode_timing.get(
                "fft_build", 0.0
            )
            early_stage_seconds["block_packet_wdm"] += mode_timing.get(
                "packet_wdm", 0.0
            )
            early_stage_seconds["block_setup"] += mode_timing.get("setup", 0.0)
            early_stage_seconds["block_total"] += mode_timing.get("total", 0.0)
            early_fft_samples.append(fft_samples)
            early_block_fft_diagnostics.extend(block_diagnostics)
            mode_methods.append("partitioned-band-fft")
            early_parts.append(part)
            continue
        if fast_method == "band-fft":
            part, _, _, fft_samples = narrow_band_fft_wdm(
                carrier,
                tdi,
                channel,
                shape,
                taper_width,
                time_weight=early_weight,
            )
            early_fft_samples.append(fft_samples)
            mode_methods.append("band-fft")
            early_parts.append(part)
            continue
        if fast_method == "spa":
            part, _, _ = numerical_spa_wdm(
                carrier,
                tdi,
                channel,
                shape,
                taper_width,
                time_weight=early_weight,
            )
            early_fft_samples.append(0)
            mode_methods.append("spa")
            early_parts.append(part)
            continue
        try:
            mode_timing: dict[str, float] = {}
            part, turnover_regions, fft_samples = turnover_partition_wdm(
                carrier,
                tdi,
                channel,
                shape,
                taper_width,
                early_weight,
                timing_diagnostics=mode_timing,
            )
            for key in (
                "turnover_search",
                "pure_spa",
                "turnover_fft",
                "turnover_spa",
                "combine",
            ):
                early_stage_seconds[key] += mode_timing.get(key, 0.0)
            if turnover_regions:
                mode_methods.append("spa+local-turnover-fft")
                turnover_region_count += len(turnover_regions)
            else:
                mode_methods.append("spa")
            early_fft_samples.append(fft_samples)
            early_parts.append(part)
        except RuntimeError:
            try:
                fallback_start = time.perf_counter()
                part, _, _, band_fft_samples = narrow_band_fft_wdm(
                    carrier,
                    tdi,
                    channel,
                    shape,
                    taper_width,
                    time_weight=early_weight,
                )
                early_stage_seconds["fallback"] += (
                    time.perf_counter() - fallback_start
                )
                early_fft_samples.append(band_fft_samples)
                mode_methods.append("band-fft-fallback")
                early_parts.append(part)
            except RuntimeError:
                part = empty_wdm_channel(shape)
                early_fft_samples.append(0)
                mode_methods.append("plunge-only")
                early_parts.append(part)
    timings["early_mode_wdm"] = time.perf_counter() - start
    timings["early_turnover_search"] = early_stage_seconds["turnover_search"]
    timings["early_pure_spa_wdm"] = early_stage_seconds["pure_spa"]
    timings["early_turnover_fft_wdm"] = early_stage_seconds["turnover_fft"]
    timings["early_turnover_spa_wdm"] = early_stage_seconds["turnover_spa"]
    timings["early_turnover_combine"] = early_stage_seconds["combine"]
    timings["early_band_fft_fallback"] = early_stage_seconds["fallback"]
    timings["early_block_fft_plan"] = early_stage_seconds["block_plan"]
    timings["early_block_fft_build"] = early_stage_seconds["block_fft_build"]
    timings["early_block_fft_packet_wdm"] = early_stage_seconds[
        "block_packet_wdm"
    ]
    timings["early_block_fft_setup"] = early_stage_seconds["block_setup"]
    timings["early_block_fft_total"] = early_stage_seconds["block_total"]

    start = time.perf_counter()
    plunge_wdm, _, _, endpoint_diagnostics = common_plunge_wdm(
        collection,
        channel,
        shape,
        l_splines,
        p_splines,
        v_splines,
        partition_start,
        partition_rise,
        taper_width,
        max(plunge_driver_frequency, collection.maximum_qnm_frequency),
        spectral_power_tolerance=endpoint_spectral_power_tolerance,
        dense_chunk=dense_chunk,
        tdi_generation=tdi_generation,
    )
    timings["common_plunge_wdm"] = time.perf_counter() - start

    early_listn, early_listm, early_values = combine_sparse_wdm(
        shape,
        [part.listn for part in early_parts],
        [part.listm for part in early_parts],
        [part.values for part in early_parts],
    )
    listn, listm, values = combine_sparse_wdm(
        shape,
        [part.listn for part in early_parts] + [plunge_wdm.listn],
        [part.listm for part in early_parts] + [plunge_wdm.listm],
        [part.values for part in early_parts] + [plunge_wdm.values],
    )
    early_nmid = np.full(shape.nf, -1, dtype=np.int64)
    early_nsize = np.zeros(shape.nf, dtype=np.int64)
    for part in early_parts:
        merge_pixel_plans(early_nmid, early_nsize, part.nmid, part.nsize, shape)
    nmid = early_nmid.copy()
    nsize = early_nsize.copy()
    merge_pixel_plans(nmid, nsize, plunge_wdm.nmid, plunge_wdm.nsize, shape)
    fast_wdm = WDMChannel(
        np.empty(0), np.empty(0), np.empty(0), nmid, nsize, listn, listm, values
    )

    max_mode = collection.modes[plunge_driver_index]
    fast_metrics = {
        "central_bh_spin": source.spin,
        "kerr_equatorial_model": float(abs(source.spin) > 1.0e-14),
        "few_model_cache_hit": few_runtime_diagnostics["cache_hit"],
        "few_model_source_call_warm": few_runtime_diagnostics[
            "source_call_warm"
        ],
        "selected_modes": float(len(collection.selected_modes)),
        "exact_phase_carriers": float(collection.size),
        "exact_folded_mode_reduction": float(
            len(collection.selected_modes) - collection.size
        ),
        "few_sparse_points": float(collection.sparse_time.size),
        "tdi_grid_points": float(tdi_times.size),
        "tdi_plan_rejected_intervals": planner_diagnostics["rejected_intervals"],
        "tdi_plan_frequency_calls": planner_diagnostics["frequency_calls"],
        "tdi_plan_mode_frequency_values": planner_diagnostics["frequency_values"],
        "tdi_plan_minimum_step_seconds": planner_diagnostics["minimum_step_seconds"],
        "plunge_time_seconds": collection.plunge_time,
        "detector_endpoint_first_seconds": response_endpoint_start,
        "detector_endpoint_last_seconds": response_endpoint_stop,
        "ringdown_stop_seconds": collection.ringdown_stop,
        "ringdown_duration_seconds": collection.ringdown_stop
        - collection.plunge_time,
        "ringdown_groups": float(len(collection.ringdown_groups)),
        "maximum_qnm_frequency_hz": collection.maximum_qnm_frequency,
        "plunge_partition_start_seconds": partition_start,
        "plunge_partition_rise_seconds": partition_rise,
        "plunge_packet_pixels": float(plunge_wdm.values.size),
        "early_wdm_pixels": float(early_values.size),
        "early_endpoint_overlap_pixels": float(
            early_values.size + plunge_wdm.values.size - values.size
        ),
        "endpoint_unique_wdm_pixels": float(values.size - early_values.size),
        "spa_mode_count": float(mode_methods.count("spa")),
        "local_turnover_fft_mode_count": float(
            mode_methods.count("spa+local-turnover-fft")
        ),
        "local_turnover_region_count": float(turnover_region_count),
        "band_fft_mode_count": float(
            sum(method.startswith("band-fft") for method in mode_methods)
        ),
        "partitioned_band_fft_mode_count": float(
            mode_methods.count("partitioned-band-fft")
        ),
        "early_block_fft_bandwidth_hz": early_fft_bandwidth_hz,
        "early_block_fft_roll_seconds": early_fft_roll_seconds,
        "early_block_fft_blocks": float(len(early_block_fft_diagnostics)),
        "early_fft_samples_total": float(sum(early_fft_samples)),
        "early_fft_samples_maximum": float(max(early_fft_samples, default=0)),
        "plunge_only_mode_count": float(mode_methods.count("plunge-only")),
        "highest_frequency_mode_l": float(max_mode[0]),
        "highest_frequency_mode_m": float(max_mode[1]),
        "highest_frequency_mode_k": float(max_mode[2]),
        "highest_frequency_mode_n": float(max_mode[3]),
        "highest_early_frequency_hz": max(early_frequency_maxima),
        "highest_plunge_frequency_hz": plunge_driver_frequency,
        "highest_frequency_fraction_of_nyquist": (
            plunge_driver_frequency / nyquist
        ),
        "active_wdm_pixels": float(values.size),
        "mode_set_algebra_relative_max": _mode_set_algebra_error(collection),
        "endpoint_sample_dt_seconds": endpoint_diagnostics["sample_dt_seconds"],
        "endpoint_sample_count": endpoint_diagnostics["sample_count"],
        "endpoint_frequency_stop_hz": endpoint_diagnostics["frequency_stop_hz"],
        "endpoint_frequency_layers": endpoint_diagnostics["frequency_layers"],
        "direct_validation": float(direct_validation),
        "global_fft_reference": float(global_fft_reference),
    }
    timings["fast_generation_total"] = time.perf_counter() - fast_generation_start

    global_fft_wdm: WDMChannel | None = None
    if global_fft_reference:
        global_start = time.perf_counter()
        stage_start = time.perf_counter()
        global_time = shape.dt * np.arange(shape.n, dtype=np.float64)
        global_early_strain = reconstruct_mode_set_sparse_tdi(
            collection,
            tdi_modes,
            channel,
            global_time,
            p_splines,
            chunk_size=dense_chunk,
        )
        global_early_strain *= early_weight(global_time)
        global_early_strain *= edge_window(global_time, shape.Tobs, taper_width)
        timings["global_fft_dense_reconstruction"] = (
            time.perf_counter() - stage_start
        )

        stage_start = time.perf_counter()
        global_spectrum = np.fft.rfft(global_early_strain) * (2.0 * shape.dt)
        timings["global_fft_rfft"] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        global_early_wdm = sparse_wdm_from_global_rfft(
            global_spectrum, early_nmid, early_nsize, shape
        )
        global_early_n, global_early_m, global_early_values = combine_sparse_wdm(
            shape,
            [global_early_wdm.listn],
            [global_early_wdm.listm],
            [global_early_wdm.values],
        )
        global_n, global_m, global_values = combine_sparse_wdm(
            shape,
            [global_early_n, plunge_wdm.listn],
            [global_early_m, plunge_wdm.listm],
            [global_early_values, plunge_wdm.values],
        )
        timings["global_fft_sparse_wdm"] = time.perf_counter() - stage_start
        timings["global_fft_reference_total"] = time.perf_counter() - global_start
        timings["global_fft_plus_common_endpoint"] = (
            timings["global_fft_reference_total"] + timings["common_plunge_wdm"]
        )
        timings["fast_wdm_plus_common_endpoint"] = (
            timings["early_mode_wdm"] + timings["common_plunge_wdm"]
        )

        global_early_on_fast, early_present = sparse_values_on_coordinates(
            global_early_n,
            global_early_m,
            global_early_values,
            early_listn,
            early_listm,
            shape,
        )
        global_on_fast, total_present = sparse_values_on_coordinates(
            global_n,
            global_m,
            global_values,
            listn,
            listm,
            shape,
        )
        if not np.all(early_present) or not np.all(total_present):
            raise RuntimeError("global FFT reference is missing a fast-path pixel")
        early_global_match = normalized_match(global_early_on_fast, early_values)
        total_global_match = normalized_match(global_on_fast, values)
        global_early_power = float(np.sum(global_early_values**2))
        global_total_power = float(np.sum(global_values**2))
        fast_metrics.update(
            {
                "global_fft_early_match": early_global_match,
                "global_fft_early_mismatch": 1.0 - early_global_match,
                "global_fft_total_match": total_global_match,
                "global_fft_total_mismatch": 1.0 - total_global_match,
                "fast_to_global_fft_early_power": float(np.sum(early_values**2))
                / max(
                    float(np.sum(global_early_on_fast**2)), np.finfo(float).tiny
                ),
                "fast_to_global_fft_total_power": float(np.sum(values**2))
                / max(float(np.sum(global_on_fast**2)), np.finfo(float).tiny),
                "global_fft_early_power_in_fast_support": float(
                    np.sum(global_early_on_fast**2)
                )
                / max(global_early_power, np.finfo(float).tiny),
                "global_fft_total_power_in_fast_support": float(
                    np.sum(global_on_fast**2)
                )
                / max(global_total_power, np.finfo(float).tiny),
                "global_fft_early_plan_pixels": float(global_early_values.size),
                "global_fft_total_plan_pixels": float(global_values.size),
                "global_fft_dense_samples": float(shape.n),
            }
        )
        global_fft_wdm = WDMChannel(
            np.empty(0),
            np.empty(0),
            np.empty(0),
            nmid.copy(),
            nsize.copy(),
            listn.copy(),
            listm.copy(),
            global_on_fast,
        )
        del global_time, global_early_strain, global_spectrum

    if not direct_validation:
        return FEWMultiModeValidationResult(
            source,
            collection.modes,
            shape,
            collection,
            tdi_modes,
            fast_wdm,
            np.empty((0, 0), dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            tuple(mode_methods),
            fast_metrics,
            timings,
            tuple(early_block_fft_diagnostics),
            global_fft_wdm,
        )

    dense_time = shape.dt * np.arange(shape.n, dtype=np.float64)
    start = time.perf_counter()
    direct = mode_set_tdi_response(
        collection,
        dense_time,
        (channel,),
        l_splines,
        p_splines,
        v_splines,
        per_mode=False,
        chunk_size=dense_chunk,
        tdi_generation=tdi_generation,
    )[channel][0]
    reconstructed = reconstruct_mode_set_sparse_tdi(
        collection,
        tdi_modes,
        channel,
        dense_time,
        p_splines,
        chunk_size=dense_chunk,
    )
    window = edge_window(dense_time, shape.Tobs, taper_width)
    direct *= window
    reconstructed *= window
    timings["direct_dense_tdi"] = time.perf_counter() - start

    start = time.perf_counter()
    direct_wdm = tw_freq(
        direct,
        shape.nf,
        shape.nt,
        1.0 / shape.dt,
        window_choice="meyer",
    )
    direct_wdm *= math.sqrt(8.0 / 15.0)
    timings["full_wdm"] = time.perf_counter() - start

    start = time.perf_counter()
    late_dense_weight = plunge_partition(
        dense_time, partition_start, partition_rise
    )
    direct_early_wdm = tw_freq(
        direct * (1.0 - late_dense_weight),
        shape.nf,
        shape.nt,
        1.0 / shape.dt,
        window_choice="meyer",
    )
    direct_final_wdm = tw_freq(
        direct * late_dense_weight,
        shape.nf,
        shape.nt,
        1.0 / shape.dt,
        window_choice="meyer",
    )
    direct_early_wdm *= math.sqrt(8.0 / 15.0)
    direct_final_wdm *= math.sqrt(8.0 / 15.0)
    timings["partition_reference_wdm"] = time.perf_counter() - start

    active_reference = direct_wdm[listn, listm]
    match = normalized_match(active_reference, values)
    full_power = float(np.sum(direct_wdm[:, 1 : shape.nf] ** 2))
    support_power = float(np.sum(active_reference**2))
    fast_power = float(np.sum(values**2))
    interior = (dense_time >= taper_width) & (dense_time <= shape.Tobs - taper_width)
    early_dense_weight = early_weight(dense_time)
    direct_early = direct * early_dense_weight
    reconstructed_early = reconstructed * early_dense_weight
    direct_norm = math.sqrt(float(np.sum(direct_early[interior] ** 2)))
    early_match = normalized_match(
        direct_early_wdm[early_listn, early_listm], early_values
    )
    plunge_match = normalized_match(
        direct_final_wdm[plunge_wdm.listn, plunge_wdm.listm],
        plunge_wdm.values,
    )
    metrics = {
        **fast_metrics,
        "tdi_reconstruction_relative_l2": float(
            np.linalg.norm(
                reconstructed_early[interior] - direct_early[interior]
            )
            / max(direct_norm, np.finfo(float).tiny)
        ),
        "wdm_match": match,
        "wdm_mismatch": 1.0 - match,
        "early_wdm_match": early_match,
        "early_wdm_mismatch": 1.0 - early_match,
        "common_plunge_wdm_match": plunge_match,
        "common_plunge_wdm_mismatch": 1.0 - plunge_match,
        "reference_power_in_fast_support": support_power
        / max(full_power, np.finfo(float).tiny),
        "fast_to_reference_support_power": fast_power
        / max(support_power, np.finfo(float).tiny),
    }
    return FEWMultiModeValidationResult(
        source,
        collection.modes,
        shape,
        collection,
        tdi_modes,
        fast_wdm,
        direct_wdm,
        dense_time,
        direct,
        reconstructed,
        tuple(mode_methods),
        metrics,
        timings,
        tuple(early_block_fft_diagnostics),
        global_fft_wdm,
    )


def validate_mode(
    source: FEWSourceParams,
    mode: Mode,
    shape: WDMShape,
    channel: str = "X",
    taper_pixels: float = 8.0,
    dense_chunk: int = 100000,
    fast_method: str = "auto",
    tdi_maximum_step: float = 1.0e4,
    early_fft_bandwidth_hz: float = 1.0e-3,
    early_fft_roll_seconds: float = EARLY_BLOCK_FFT_ROLL_SECONDS,
    early_fft_tile_strategy: str = "dyadic",
    tdi_generation: int = 1,
) -> FEWValidationResult:
    if shape.nt < 2 or shape.nt & (shape.nt - 1):
        raise ValueError("WDM nt must be a power of two")
    if shape.nf < 2 or shape.nf & (shape.nf - 1):
        raise ValueError("WDM nf must be a power of two")
    if tdi_maximum_step <= 0.0:
        raise ValueError("TDI maximum step must be positive")
    timings: dict[str, float] = {}
    few_runtime_diagnostics: dict[str, float] = {}
    start = time.perf_counter()
    carrier = build_few_mode_carrier(
        source,
        mode,
        shape,
        runtime_diagnostics=few_runtime_diagnostics,
    )
    carrier_seconds = time.perf_counter() - start
    initialization_seconds = few_runtime_diagnostics["initialization_seconds"]
    timings["few_model_initialization"] = initialization_seconds
    timings["few_sparse_intrinsic"] = max(
        0.0, carrier_seconds - initialization_seconds
    )
    timings["few_carrier_total"] = carrier_seconds

    start = time.perf_counter()
    _, l_splines, p_splines, v_splines = build_constellation_splines(shape)
    timings["constellation_setup"] = time.perf_counter() - start
    taper_width = taper_pixels * shape.DT

    start = time.perf_counter()
    planner_diagnostics: dict[str, float] = {}
    tdi_times = build_tdi_time_grid(
        carrier,
        shape,
        p_splines,
        maximum_step=tdi_maximum_step,
        taper_width=taper_width,
        diagnostics=planner_diagnostics,
        l_splines=l_splines,
        channel=channel,
        tdi_generation=tdi_generation,
    )
    timings["tdi_grid_plan"] = time.perf_counter() - start
    start = time.perf_counter()
    tdi = extract_mode_pair_tdi_ap(
        carrier, tdi_times, (channel,), l_splines, p_splines, v_splines,
        tdi_generation=tdi_generation,
    )
    timings["sparse_tdi_response"] = time.perf_counter() - start
    timings["sparse_tdi"] = (
        timings["tdi_grid_plan"] + timings["sparse_tdi_response"]
    )
    for key in (
        "reference_time_seconds",
        "few_frequency_seconds",
        "interval_tests_seconds",
        "edge_merge_seconds",
        "loop_overhead_seconds",
    ):
        timings[f"tdi_plan_{key.removesuffix('_seconds')}"] = planner_diagnostics[key]

    start = time.perf_counter()
    track_frequency, track_fdot = tdi_track_frequency(carrier, tdi, channel)
    has_turnover = bool(np.any(track_fdot[:-1] * track_fdot[1:] < 0.0))
    selected_method = fast_method
    if selected_method == "auto":
        selected_method = "band-fft" if has_turnover else "spa"
    band_fft_samples = 0
    block_fft_count = 0
    turnover_regions: list[FEWTurnoverRegion] = []
    if selected_method == "spa":
        fast_wdm, spa_frequency, spa_fdot = numerical_spa_wdm(
            carrier, tdi, channel, shape, taper_width
        )
    elif selected_method == "band-fft":
        fast_wdm, spa_frequency, spa_fdot, band_fft_samples = narrow_band_fft_wdm(
            carrier, tdi, channel, shape, taper_width
        )
    elif selected_method == "block-fft":
        block_timing: dict[str, float] = {}
        fast_wdm, band_fft_samples, block_diagnostics = partitioned_band_fft_wdm(
            carrier,
            tdi,
            channel,
            shape,
            taper_width,
            lambda values: np.ones_like(values),
            bandwidth_hz=early_fft_bandwidth_hz,
            roll_seconds=early_fft_roll_seconds,
            tile_strategy=early_fft_tile_strategy,
            track_frequency=track_frequency,
            timing_diagnostics=block_timing,
        )
        block_fft_count = len(block_diagnostics)
        timings["early_block_fft_plan"] = block_timing.get("plan", 0.0)
        timings["early_block_fft_build"] = block_timing.get("fft_build", 0.0)
        timings["early_block_fft_packet_wdm"] = block_timing.get(
            "packet_wdm", 0.0
        )
        spa_frequency, spa_fdot = track_frequency, track_fdot
    elif selected_method == "turnover-packets":
        partition_parts: dict[str, list[WDMChannel]] = {}
        fast_wdm, turnover_regions, band_fft_samples = turnover_partition_wdm(
            carrier,
            tdi,
            channel,
            shape,
            taper_width,
            lambda values: np.ones_like(values),
            diagnostic_parts=partition_parts,
        )
        spa_frequency, spa_fdot = tdi_track_frequency(carrier, tdi, channel)
        if not turnover_regions:
            selected_method = "spa"
    else:
        raise ValueError(f"unknown FEW fast transform method {fast_method!r}")
    timings[f"fast_{selected_method}_wdm"] = time.perf_counter() - start

    dense_time = shape.dt * np.arange(shape.n, dtype=np.float64)
    start = time.perf_counter()
    direct = mode_pair_tdi_response(
        carrier,
        dense_time,
        (channel,),
        l_splines,
        p_splines,
        v_splines,
        chunk_size=dense_chunk,
        tdi_generation=tdi_generation,
    )[channel][0]
    reconstructed = reconstruct_sparse_tdi(
        carrier, tdi, channel, dense_time, p_splines
    )
    window = edge_window(dense_time, shape.Tobs, taper_width)
    direct *= window
    reconstructed *= window
    timings["direct_dense_tdi"] = time.perf_counter() - start

    start = time.perf_counter()
    direct_wdm = tw_freq(
        direct,
        shape.nf,
        shape.nt,
        1.0 / shape.dt,
        window_choice="meyer",
    )
    # WDG.py retains the original frequency-domain Meyer normalization.  The
    # production sparse transform follows the later wd_viafreq/wd_transform
    # correction, whose coefficients are smaller by sqrt(8/15).
    direct_wdm *= math.sqrt(8.0 / 15.0)
    timings["full_wdm"] = time.perf_counter() - start

    partition_metrics: dict[str, float] = {}
    if selected_method == "turnover-packets" and turnover_regions:
        packet_weight = np.zeros_like(dense_time)
        packet_taper = 4.0 * shape.DT
        for region in turnover_regions:
            packet_weight += turnover_packet_envelope(
                dense_time, region.start, region.stop, packet_taper
            )
        packet_weight = np.clip(packet_weight, 0.0, 1.0)
        packet_candidates = partition_parts.get("turnover", [])
        if packet_candidates:
            listn, listm, values = combine_sparse_wdm(
                shape,
                [part.listn for part in packet_candidates],
                [part.listm for part in packet_candidates],
                [part.values for part in packet_candidates],
            )
            reference_part = tw_freq(
                direct * packet_weight,
                shape.nf,
                shape.nt,
                1.0 / shape.dt,
                window_choice="meyer",
            )
            reference_part *= math.sqrt(8.0 / 15.0)
            reference_values = reference_part[listn, listm]
            partition_metrics["turnover_wdm_match"] = normalized_match(
                reference_values, values
            )
            partition_metrics["turnover_fast_to_reference_power"] = float(
                np.sum(values * values)
                / max(np.sum(reference_values * reference_values), np.finfo(float).tiny)
            )
        spa_candidates = partition_parts.get("spa", [])
        if spa_candidates:
            listn, listm, values = combine_sparse_wdm(
                shape,
                [part.listn for part in spa_candidates],
                [part.listm for part in spa_candidates],
                [part.values for part in spa_candidates],
            )
            if packet_candidates:
                packet_n, packet_m, _ = combine_sparse_wdm(
                    shape,
                    [part.listn for part in packet_candidates],
                    [part.listm for part in packet_candidates],
                    [part.values for part in packet_candidates],
                )
                packet_linear = packet_n * (shape.nf + 1) + packet_m
                keep = ~np.isin(
                    listn * (shape.nf + 1) + listm,
                    packet_linear,
                    assume_unique=True,
                )
                listn, listm, values = listn[keep], listm[keep], values[keep]
            reference_values = direct_wdm[listn, listm]
            partition_metrics["spa_outside_turnover_wdm_match"] = normalized_match(
                reference_values, values
            )
            partition_metrics["spa_outside_turnover_fast_to_reference_power"] = float(
                np.sum(values * values)
                / max(np.sum(reference_values * reference_values), np.finfo(float).tiny)
            )
            for index, candidate in enumerate(spa_candidates):
                keep = np.ones(candidate.values.size, dtype=bool)
                if packet_candidates:
                    keep &= ~np.isin(
                        candidate.listn * (shape.nf + 1) + candidate.listm,
                        packet_linear,
                    )
                branch_reference = direct_wdm[
                    candidate.listn[keep], candidate.listm[keep]
                ]
                branch_values = candidate.values[keep]
                partition_metrics[f"spa_branch_{index}_wdm_match"] = normalized_match(
                    branch_reference, branch_values
                )
                partition_metrics[f"spa_branch_{index}_fast_to_reference_power"] = float(
                    np.sum(branch_values * branch_values)
                    / max(
                        np.sum(branch_reference * branch_reference),
                        np.finfo(float).tiny,
                    )
                )
                partition_metrics[f"spa_branch_{index}_first_time_pixel"] = float(
                    np.min(candidate.listn[keep]) if np.any(keep) else -1
                )
                partition_metrics[f"spa_branch_{index}_last_time_pixel"] = float(
                    np.max(candidate.listn[keep]) if np.any(keep) else -1
                )

    active_reference = direct_wdm[fast_wdm.listn, fast_wdm.listm]
    fast_values = fast_wdm.values
    match = normalized_match(active_reference, fast_values)
    full_power = float(np.sum(direct_wdm[:, 1 : shape.nf] ** 2))
    support_power = float(np.sum(active_reference**2))
    fast_power = float(np.sum(fast_values**2))
    interior = (dense_time >= taper_width) & (dense_time <= shape.Tobs - taper_width)
    direct_norm = math.sqrt(float(np.sum(direct[interior] ** 2)))
    tdi_rel_l2 = float(
        np.linalg.norm(reconstructed[interior] - direct[interior])
        / max(direct_norm, np.finfo(float).tiny)
    )
    amp_rms, amp_max = native_amplitude_dense_reevaluation_difference(carrier)
    metrics = {
        "central_bh_spin": source.spin,
        "kerr_equatorial_model": float(abs(source.spin) > 1.0e-14),
        "few_model_cache_hit": few_runtime_diagnostics["cache_hit"],
        "few_model_source_call_warm": few_runtime_diagnostics[
            "source_call_warm"
        ],
        "few_sparse_points": float(carrier.sparse_time.size),
        "tdi_grid_points": float(tdi.time.size),
        "tdi_plan_accepted_intervals": planner_diagnostics["accepted_intervals"],
        "tdi_plan_rejected_intervals": planner_diagnostics["rejected_intervals"],
        "tdi_plan_frequency_calls": planner_diagnostics["frequency_calls"],
        "tdi_plan_frequency_values": planner_diagnostics["frequency_values"],
        "tdi_plan_points_before_edges": planner_diagnostics[
            "adaptive_points_before_edges"
        ],
        "tdi_plan_minimum_step_seconds": planner_diagnostics[
            "minimum_step_seconds"
        ],
        "tdi_plan_maximum_step_seconds": planner_diagnostics[
            "maximum_step_seconds"
        ],
        "active_wdm_pixels": float(fast_values.size),
        "mode_pair_algebra_relative_max": _mode_pair_algebra_error(carrier),
        "few_native_vs_dense_roman_relative_rms": amp_rms,
        "few_native_vs_dense_roman_relative_max": amp_max,
        "tdi_reconstruction_relative_l2": tdi_rel_l2,
        "wdm_match": match,
        "wdm_mismatch": 1.0 - match,
        "reference_power_in_fast_support": support_power / max(full_power, np.finfo(float).tiny),
        "fast_to_reference_support_power": fast_power / max(support_power, np.finfo(float).tiny),
        "spa_frequency_min_hz": float(np.min(spa_frequency[np.isfinite(spa_frequency)])),
        "spa_frequency_max_hz": float(np.max(spa_frequency[np.isfinite(spa_frequency)])),
        "spa_fdot_min_hz_s": float(np.min(spa_fdot[np.isfinite(spa_fdot)])),
        "track_has_turnover": float(has_turnover),
        "local_turnover_regions": float(len(turnover_regions)),
        "band_fft_samples": float(band_fft_samples),
        "early_block_fft_bandwidth_hz": early_fft_bandwidth_hz,
        "early_block_fft_roll_seconds": early_fft_roll_seconds,
        "early_block_fft_blocks": float(block_fft_count),
        **partition_metrics,
    }
    return FEWValidationResult(
        source,
        mode,
        shape,
        carrier,
        tdi,
        fast_wdm,
        direct_wdm,
        dense_time,
        direct,
        reconstructed,
        selected_method,
        metrics,
        timings,
    )


def write_results(result: FEWValidationResult, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "summary.csv").open("w", newline="") as stream:
        rows = {**result.metrics, **{f"seconds_{k}": v for k, v in result.timings.items()}}
        writer = csv.writer(stream)
        writer.writerow(("quantity", "value"))
        writer.writerow(("fast_method", result.fast_method))
        writer.writerows(rows.items())
    np.savetxt(
        output / "tdi_sparse_ap.dat",
        np.column_stack(
            [
                result.tdi.time,
                result.tdi.detector_time,
                result.tdi.amplitude[next(iter(result.tdi.amplitude))],
                result.tdi.channel_phase(next(iter(result.tdi.amplitude))),
            ]
        ),
        header="ssb_output_time_s center_source_time_u_s amplitude phase",
    )
    np.savetxt(
        output / "track_pixels_fast.dat",
        np.column_stack(
            [result.fast_wdm.listn, result.fast_wdm.listm, result.fast_wdm.values]
        ),
        fmt=["%d", "%d", "%.15e"],
        header="n m fast_wdm",
    )
    if result.direct_wdm.size == 0:
        return
    reference = result.direct_wdm[result.fast_wdm.listn, result.fast_wdm.listm]
    np.savetxt(
        output / "track_pixels_compare.dat",
        np.column_stack(
            [
                result.fast_wdm.listn,
                result.fast_wdm.listm,
                reference,
                result.fast_wdm.values,
            ]
        ),
        fmt=["%d", "%d", "%.15e", "%.15e"],
        header="n m direct_wdm fast_wdm",
    )
    layer_rows = []
    for layer in np.unique(result.fast_wdm.listm):
        select = result.fast_wdm.listm == layer
        direct = reference[select]
        fast = result.fast_wdm.values[select]
        layer_rows.append(
            (
                int(layer),
                int(np.count_nonzero(select)),
                float(np.sum(direct * direct)),
                float(np.sum(fast * fast)),
                normalized_match(direct, fast),
            )
        )
    np.savetxt(
        output / "layer_matches.dat",
        np.asarray(layer_rows),
        fmt=["%d", "%d", "%.15e", "%.15e", "%.15e"],
        header="m pixels direct_power fast_power match",
    )


def write_multimode_results(
    result: FEWMultiModeValidationResult, output: Path
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "summary.csv").open("w", newline="") as stream:
        rows = {
            **result.metrics,
            **{f"seconds_{key}": value for key, value in result.timings.items()},
        }
        writer = csv.writer(stream)
        writer.writerow(("quantity", "value"))
        writer.writerow(("mode_selection", "FEW frozen selected set"))
        writer.writerows(rows.items())
    with (output / "selected_modes.dat").open("w") as stream:
        stream.write("# ell m k n early_method exact_carrier\n")
        for carrier_index, members in enumerate(result.collection.carrier_members):
            method = result.mode_methods[carrier_index]
            for mode in members:
                stream.write(
                    f"{mode[0]} {mode[1]} {mode[2]} {mode[3]} "
                    f"{method} {carrier_index}\n"
                )
    with (output / "folded_carriers.dat").open("w") as stream:
        stream.write("# carrier representative_ell m k n member_count early_method\n")
        for index, (mode, members, method) in enumerate(
            zip(
                result.collection.modes,
                result.collection.carrier_members,
                result.mode_methods,
            )
        ):
            vector = result.collection.mode_vectors[index].astype(int)
            stream.write(
                f"{index} {mode[0]} {vector[0]} {vector[1]} {vector[2]} "
                f"{len(members)} {method}\n"
            )

    if result.early_block_fft_diagnostics:
        rows = np.asarray(
            [
                (
                    item.carrier_index,
                    item.block_index,
                    item.tile_lo,
                    item.tile_hi,
                    item.support_lo,
                    item.support_hi,
                    item.nonzero_start,
                    item.nonzero_stop,
                    item.frequency_min,
                    item.frequency_max,
                    item.frequency_pad,
                    item.heterodyne_layer,
                    item.sample_dt,
                    item.sample_count,
                    item.packet_time_pixels,
                )
                for item in result.early_block_fft_diagnostics
            ]
        )
        np.savetxt(
            output / "early_block_fft_plan.dat",
            rows,
            fmt=["%d"] * 6 + ["%.15e"] * 5 + ["%d", "%.15e", "%d", "%d"],
            header=(
                "carrier block tile_lo tile_hi support_lo support_hi "
                "nonzero_start nonzero_stop frequency_min frequency_max "
                "frequency_pad heterodyne_layer sample_dt sample_count "
                "packet_time_pixels"
            ),
        )

    collection = result.collection
    if collection.ringdown_groups:
        group_rows = np.asarray(
            [
                (
                    group.ell,
                    group.m,
                    group.right_omega / (2.0 * PI),
                    1.0 / group.right_damping,
                    group.left_omega / (2.0 * PI),
                    1.0 / group.left_damping,
                    group.transition,
                    group.fade_start,
                    group.fade_duration,
                    abs(group.right_value),
                    abs(group.left_value),
                )
                for group in collection.ringdown_groups
            ]
        )
        np.savetxt(
            output / "ringdown_groups.dat",
            group_rows,
            fmt=["%d", "%d"] + ["%.15e"] * 9,
            header=(
                "ell abs_m right_qnm_frequency_hz right_damping_time_seconds "
                "left_qnm_frequency_hz left_damping_time_seconds "
                "bridge_seconds fade_start_seconds fade_duration_seconds "
                "right_endpoint_amplitude left_endpoint_amplitude"
            ),
        )

        source_start = max(
            0.0, collection.plunge_time - 250.0 * collection.mass_seconds
        )
        source_times = np.arange(
            source_start,
            collection.ringdown_stop + 0.5 * result.shape.dt,
            result.shape.dt,
        )
        hp, hc = collection.summed_polarizations(source_times)
        np.savetxt(
            output / "endpoint_source_waveform.dat",
            np.column_stack(
                (
                    source_times,
                    source_times - collection.plunge_time,
                    hp,
                    hc,
                    np.hypot(hp, hc),
                )
            ),
            header=(
                "source_time_s time_from_few_endpoint_s "
                "hplus hcross polarization_amplitude"
            ),
        )

        if result.direct_time.size:
            detector_start = (
                result.metrics["detector_endpoint_first_seconds"]
                - 50.0 * collection.mass_seconds
            )
            detector_stop = (
                result.metrics["detector_endpoint_last_seconds"]
                + result.metrics["ringdown_duration_seconds"]
                + 50.0 * collection.mass_seconds
            )
            endpoint_slice = (
                (result.direct_time >= detector_start)
                & (result.direct_time <= detector_stop)
            )
            np.savetxt(
                output / "endpoint_direct_tdi.dat",
                np.column_stack(
                    (
                        result.direct_time[endpoint_slice],
                        result.direct_strain[endpoint_slice],
                    )
                ),
                header="detector_time direct_tdi_strain",
            )
    np.savetxt(
        output / "track_pixels_fast.dat",
        np.column_stack(
            [result.fast_wdm.listn, result.fast_wdm.listm, result.fast_wdm.values]
        ),
        fmt=["%d", "%d", "%.15e"],
        header="n m fast_wdm",
    )
    if result.global_fft_wdm is not None:
        global_wdm = result.global_fft_wdm
        if not (
            np.array_equal(result.fast_wdm.listn, global_wdm.listn)
            and np.array_equal(result.fast_wdm.listm, global_wdm.listm)
        ):
            raise RuntimeError("fast and global FFT output supports differ")
        np.savetxt(
            output / "track_pixels_global_fft_compare.dat",
            np.column_stack(
                (
                    result.fast_wdm.listn,
                    result.fast_wdm.listm,
                    global_wdm.values,
                    result.fast_wdm.values,
                )
            ),
            fmt=["%d", "%d", "%.15e", "%.15e"],
            header="n m global_fft_wdm block_or_spa_wdm",
        )
        layer_rows = []
        for layer in np.unique(result.fast_wdm.listm):
            select = result.fast_wdm.listm == layer
            reference = global_wdm.values[select]
            candidate = result.fast_wdm.values[select]
            layer_rows.append(
                (
                    int(layer),
                    int(np.count_nonzero(select)),
                    float(np.sum(reference**2)),
                    float(np.sum(candidate**2)),
                    normalized_match(reference, candidate),
                )
            )
        np.savetxt(
            output / "global_fft_layer_matches.dat",
            np.asarray(layer_rows),
            fmt=["%d", "%d", "%.15e", "%.15e", "%.15e"],
            header="m pixels global_fft_power fast_power match",
        )
    if result.direct_wdm.size == 0:
        return
    reference = result.direct_wdm[result.fast_wdm.listn, result.fast_wdm.listm]
    np.savetxt(
        output / "track_pixels_compare.dat",
        np.column_stack(
            [
                result.fast_wdm.listn,
                result.fast_wdm.listm,
                reference,
                result.fast_wdm.values,
            ]
        ),
        fmt=["%d", "%d", "%.15e", "%.15e"],
        header="n m direct_wdm fast_wdm",
    )
    layer_rows = []
    for layer in np.unique(result.fast_wdm.listm):
        select = result.fast_wdm.listm == layer
        direct = reference[select]
        fast = result.fast_wdm.values[select]
        layer_rows.append(
            (
                int(layer),
                int(np.count_nonzero(select)),
                float(np.sum(direct * direct)),
                float(np.sum(fast * fast)),
                normalized_match(direct, fast),
            )
        )
    np.savetxt(
        output / "layer_matches.dat",
        np.asarray(layer_rows),
        fmt=["%d", "%d", "%.15e", "%.15e", "%.15e"],
        header="m pixels direct_power fast_power match",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", type=parse_mode, default=(2, 2, 0, 1))
    parser.add_argument(
        "--modes",
        choices=("single", "selected", "all"),
        default="single",
        help="single preserves the one-carrier validator; selected uses FEW's threshold selector",
    )
    parser.add_argument(
        "--mode-selection-threshold",
        type=float,
        default=1.0e-3,
        help="approximate FEW mode-selection mismatch for --modes selected",
    )
    parser.add_argument("--mass", type=float, default=1.0e6)
    parser.add_argument("--mu", type=float, default=50.0)
    parser.add_argument(
        "--spin",
        type=float,
        default=0.0,
        help=(
            "signed central-BH spin; zero uses the Schwarzschild model, while "
            "nonzero values use equatorial Kerr (negative is retrograde)"
        ),
    )
    parser.add_argument("--p0", type=float, default=12.510272236947417)
    parser.add_argument("--e0", type=float, default=0.4)
    parser.add_argument("--nt", type=int, default=512, help="WDM time pixels for the direct validation")
    parser.add_argument("--nf", type=int, default=4096)
    parser.add_argument("--dt", type=float, default=1.875)
    parser.add_argument("--channel", choices=("X", "Y", "Z"), default="X")
    parser.add_argument("--tdi-generation", type=int, choices=(1, 2), default=1)
    parser.add_argument("--taper-pixels", type=float, default=8.0)
    parser.add_argument(
        "--tdi-maximum-step",
        type=float,
        default=1.0e4,
        help=(
            "maximum SSB-output spacing in seconds before track, transfer, "
            "taper, and endpoint refinements"
        ),
    )
    parser.add_argument(
        "--endpoint-model",
        choices=("qnm", "hard"),
        default="qnm",
        help="grouped Schwarzschild-QNM completion or FEW's legacy hard cutoff",
    )
    parser.add_argument(
        "--plunge-pre-m",
        type=float,
        default=200.0,
        help=(
            "mass-scaled time before the earliest SSB-output endpoint "
            "arrival included in the common FFT; the default is floored at "
            "two Meyer half-supports"
        ),
    )
    parser.add_argument(
        "--plunge-rise-m",
        type=float,
        default=50.0,
        help=(
            "mass-scaled partition-of-unity rise time for the common FFT; "
            "the default is floored at two Meyer half-supports"
        ),
    )
    parser.add_argument(
        "--plunge-pre-pixels",
        type=float,
        default=None,
        help="legacy WDM-pixel override for --plunge-pre-m",
    )
    parser.add_argument(
        "--plunge-rise-pixels",
        type=float,
        default=None,
        help="legacy WDM-pixel override for --plunge-rise-m",
    )
    parser.add_argument(
        "--qnm-transition-m",
        type=float,
        default=10.0,
        help="C1 bridge timescale from FEW to the QNM logarithmic derivative",
    )
    parser.add_argument(
        "--qnm-efolds",
        type=float,
        default=8.0,
        help="QNM damping e-folds before the negligible-tail fade begins",
    )
    parser.add_argument(
        "--qnm-fade-efolds",
        type=float,
        default=2.0,
        help="raised-cosine fade length after the retained QNM tail",
    )
    parser.add_argument(
        "--endpoint-spectral-power-tolerance",
        type=float,
        default=1.0e-10,
        help="maximum short-FFT power discarded above the endpoint layer cutoff",
    )
    parser.add_argument(
        "--fast-method",
        choices=("auto", "spa", "turnover-packets", "band-fft", "block-fft"),
        default="auto",
        help=(
            "the multimode path uses SPA branches plus compact turnover FFT "
            "packets; block-fft instead uses an ungrouped bandwidth-limited "
            "FFT partition for every early carrier, followed by the same "
            "common endpoint FFT"
        ),
    )
    parser.add_argument(
        "--early-fft-bandwidth-hz",
        type=float,
        default=1.0e-3,
        help="maximum carrier sweep plus Meyer support in each block-FFT tile",
    )
    parser.add_argument(
        "--early-fft-roll-seconds",
        type=float,
        default=EARLY_BLOCK_FFT_ROLL_SECONDS,
        help="partition-of-unity overlap at internal block-FFT boundaries",
    )
    parser.add_argument(
        "--early-fft-tile-strategy",
        choices=("dyadic", "maximal"),
        default="dyadic",
        help=(
            "dyadic keeps local FFTs cache-sized (default); maximal is a "
            "diagnostic that minimizes block count under the bandwidth limit"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("few_tdi_wdm_validation"))
    parser.add_argument(
        "--fast-only",
        action="store_true",
        help="skip the full-cadence TDI and direct WDM validation (multimode only)",
    )
    parser.add_argument(
        "--global-fft-reference",
        action="store_true",
        help=(
            "reconstruct the same sparse-TDI early signal at full cadence, "
            "then compare one global real FFT plus sparse Meyer packets"
        ),
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=0,
        help="discard this many complete calls before reporting the measured call",
    )
    parser.add_argument("--no-write", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.warmup_runs < 0:
        raise SystemExit("--warmup-runs must be non-negative")
    if args.fast_only and args.modes == "single":
        raise SystemExit("--fast-only currently requires --modes selected or --modes all")
    if args.global_fft_reference and args.modes == "single":
        raise SystemExit("--global-fft-reference requires --modes selected or --modes all")
    shape = WDMShape(nf=args.nf, nt=args.nt, dt=args.dt)
    source = FEWSourceParams(
        mass_solar=args.mass,
        mu_solar=args.mu,
        spin=args.spin,
        p0=args.p0,
        eccentricity0=args.e0,
    )
    if args.modes == "single":
        def generate_single() -> FEWValidationResult:
            return validate_mode(
                source,
                args.mode,
                shape,
                channel=args.channel,
                taper_pixels=args.taper_pixels,
                fast_method=args.fast_method,
                tdi_maximum_step=args.tdi_maximum_step,
                early_fft_bandwidth_hz=args.early_fft_bandwidth_hz,
                early_fft_roll_seconds=args.early_fft_roll_seconds,
                early_fft_tile_strategy=args.early_fft_tile_strategy,
                tdi_generation=args.tdi_generation,
            )

        for _ in range(args.warmup_runs):
            generate_single()
        result = generate_single()
        print(f"mode {args.mode} channel {args.channel} fast_method {result.fast_method}")
    else:
        def generate_multimode(
            include_global_fft: bool,
        ) -> FEWMultiModeValidationResult:
            return validate_modes(
                source,
                shape,
                channel=args.channel,
                mode_selection=(
                    "threshold" if args.modes == "selected" else "all"
                ),
                mode_selection_threshold=args.mode_selection_threshold,
                taper_pixels=args.taper_pixels,
                endpoint_model=args.endpoint_model,
                plunge_pre_m=args.plunge_pre_m,
                plunge_rise_m=args.plunge_rise_m,
                plunge_pre_pixels=args.plunge_pre_pixels,
                plunge_rise_pixels=args.plunge_rise_pixels,
                qnm_transition_m=args.qnm_transition_m,
                qnm_efolds=args.qnm_efolds,
                qnm_fade_efolds=args.qnm_fade_efolds,
                endpoint_spectral_power_tolerance=(
                    args.endpoint_spectral_power_tolerance
                ),
                tdi_maximum_step=args.tdi_maximum_step,
                fast_method=args.fast_method,
                early_fft_bandwidth_hz=args.early_fft_bandwidth_hz,
                early_fft_roll_seconds=args.early_fft_roll_seconds,
                early_fft_tile_strategy=args.early_fft_tile_strategy,
                direct_validation=not args.fast_only,
                global_fft_reference=include_global_fft,
                tdi_generation=args.tdi_generation,
            )

        for _ in range(args.warmup_runs):
            generate_multimode(False)
        result = generate_multimode(args.global_fft_reference)
        print(
            f"modes {args.modes} selected {len(result.collection.selected_modes)} "
            f"exact_carriers {result.collection.size} channel {args.channel} "
            f"threshold {args.mode_selection_threshold:.3e} "
            f"endpoint_model {args.endpoint_model} fast_method {args.fast_method}"
        )
    print(
        f"wdm_grid nt {shape.nt} nf {shape.nf} dt {shape.dt:.9e} "
        f"Tobs {shape.Tobs:.9e}"
    )
    for key, value in result.metrics.items():
        print(f"{key} {value:.12e}")
    for key, value in result.timings.items():
        print(f"seconds_{key} {value:.6f}")
    if not args.no_write:
        if isinstance(result, FEWMultiModeValidationResult):
            write_multimode_results(result, args.output_dir)
        else:
            write_results(result, args.output_dir)
        print(f"output_dir {args.output_dir}")


if __name__ == "__main__":
    main()
