#!/usr/bin/env python3
# Python LISA response and WDM port: Copyright (C) 2026 Neil Cornish.
# SPDX-License-Identifier: GPL-3.0-or-later
# Distributed without warranty; see the GNU GPL for details.
# The imported LAL-derived waveform model retains its upstream notices.
"""Plain NumPy/Numba IMRPhenomTHM, LISA TDI, and fast WDM generator.

This is the full-mode counterpart of :mod:`phenomt_tdi_wdm`.  It uses the
local LAL-free :mod:`phenomthm` port, folds each aligned-spin ``+m/-m`` pair
into one real carrier, and applies the unequal-arm X/Y/Z response on a shared
adaptive grid.  The default WDM path uses complementary, heterodyned complex
FFT blocks for the inspiral and one direct summed-carrier FFT for the common
merger/ringdown endpoint.  The older numerical-SPA plus endpoint-FFT path is
retained as an explicit comparison option.

The module has no JAX, Phentax, or LAL runtime dependency.  Its public API
returns arrays; file output is confined to the command-line driver.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import math
import time
from typing import Iterable, Sequence

import numpy as np

try:
    from numba import njit
except ImportError:
    def njit(*_args, **_kwargs):
        return lambda function: function

from phenomthm import DEFAULT_POSITIVE_MODES, IMRPhenomTHM
from phenomt22 import IMRPhenomT22
from tdi2_response import complex_tdi2
from phenomt_tdi_wdm import (
    AP_DTM_MAX,
    CONSTELLATION_LIGHT_TIME_SECONDS,
    DEFAULT_WDM_BLEND_ENDPOINT,
    DEFAULT_WDM_BLEND_HALF_WIDTH_LAYERS,
    DTMIN,
    INTRINSIC_TDI_SWITCH_GUARD_SECONDS,
    NOMINAL_ARM_LIGHT_TIME_SECONDS,
    NUMBA_AVAILABLE,
    RESPONSE_LATE_MARGIN_SECONDS,
    SHORTFFT_MERGER_TAPER_MARGIN_SECONDS,
    SourceParams,
    T_CUT_FREQ,
    WDMChannel,
    WDMShape,
    _build_intrinsic_model_grid,
    _detector_adaptive_step,
    _detector_step_proposal,
    _extract_ap_jit,
    _intrinsic_tdi_switch_source_time,
    _stack_spline_coefficients,
    _unique_sorted_times,
    apply_short_fft_threshold,
    barycenter_time,
    build_constellation_splines,
    extract_ap,
    ftran_spa_only,
    make_ap_spline,
    merge_pixel_plans,
    phitilde,
    plan_endpoint_taper_flat_time,
    prune_nonincreasing_frequency_samples,
    sky_vectors,
    tdi_frequency_track,
    transform_plan,
    wdm_channel_from_dense,
    wdm_pixels_add_merger_frequency_tail,
    wdm_pixels_range,
    wdm_track,
    write_track_pixels,
    write_wdm_matrix,
)


PI = math.pi
MODE_CODES = (22, 21, 33, 44, 55)
MODE_BY_CODE = dict(zip(MODE_CODES, DEFAULT_POSITIVE_MODES))
CHANNELS = ("X", "Y", "Z")
TRIPLES = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
PARTITION_FFT_BANDWIDTH_HZ = 1.6e-2
PARTITION_FFT_ROLL_SECONDS = 1.0e5
PARTITION_FFT_MARGIN_CYCLES = 3.0
PARTITION_FFT_NYQUIST_GUARD_LAYERS = 2


@dataclass(frozen=True)
class THMIntrinsicGrid:
    """Shared source grid and the five positive-m real-amplitude carriers."""

    waveform_time: np.ndarray
    spline_time: np.ndarray
    response_time: np.ndarray
    detector_time: np.ndarray
    modes: tuple[tuple[int, int], ...]
    amplitude: np.ndarray
    phase: np.ndarray
    frequency: np.ndarray
    model_amplitude: np.ndarray
    model_phase: np.ndarray
    model_frequency: np.ndarray
    ring_frequency_hz: np.ndarray
    damping_rate_hz: np.ndarray
    model: IMRPhenomTHM
    response_switch_detector_time: np.ndarray
    exact_response_samples: int


@dataclass(frozen=True)
class THMTDIGrid:
    """Folded-carrier TDI quadratures and signed AP representations.

    ``time`` is the SSB output/data timestamp ``t``.  The compatibility field
    ``detector_time`` is the guiding-center retarded source argument
    ``u=t-k.r_0(t)``, not a second output timestamp.
    """

    time: np.ndarray
    detector_time: np.ndarray
    modes: tuple[tuple[int, int], ...]
    intrinsic_amplitude: np.ndarray
    intrinsic_phase: np.ndarray
    intrinsic_frequency: np.ndarray
    raw_m: np.ndarray
    raw_mf: np.ndarray
    signed_amplitude: np.ndarray
    channel_phase: np.ndarray
    reference_phase: np.ndarray
    channel_strain: np.ndarray

    @property
    def ssb_output_time(self) -> np.ndarray:
        return self.time

    @property
    def center_source_time(self) -> np.ndarray:
        return self.detector_time


@dataclass(frozen=True)
class THMCarrierFourier:
    mode: tuple[int, int]
    frequency: np.ndarray
    phase: np.ndarray
    amplitude: np.ndarray
    setup: np.ndarray
    join_frequency: float


@dataclass(frozen=True)
class THMTDIWDMResult:
    source: SourceParams
    shape: WDMShape
    intrinsic: THMIntrinsicGrid
    tdi: THMTDIGrid
    channels: dict[str, WDMChannel]
    carrier_fourier: dict[str, tuple[THMCarrierFourier, ...]]
    timings: dict[str, float]


@dataclass(frozen=True)
class THMPartitionBlockDiagnostic:
    """Compact description of one heterodyned inspiral FFT block."""

    carrier_index: int
    block_index: int
    sample_start: float
    sample_dt: float
    sample_count: int
    nonzero_start: float
    nonzero_stop: float
    frequency_min: float
    frequency_max: float
    heterodyne_frequency: float


@dataclass
class _THMPartitionSpectrum:
    """One local spectrum shared by the requested X/Y/Z channels."""

    carrier_index: int
    block_start: float
    nonzero_start: float
    nonzero_stop: float
    frequency_min: float
    frequency_max: float
    frequency_pad: float
    heterodyne_layer: int
    sample_dt: float
    spectrum: np.ndarray


class _ScaledPlanningModel:
    """Cheap 22-based frequency proxy used only to place the shared grid."""

    def __init__(self, model: IMRPhenomT22, abs_m: int):
        self.model = model
        self.scale = 0.5 * float(abs_m)
        self.omega_ring = self.scale * model.omega_ring
        self.alpha1rd = model.alpha1rd
        self._omega: dict[float, float] = {}

    def omega22(self, tau: float) -> float:
        key = float(tau)
        value = self._omega.get(key)
        if value is None:
            value = self.scale * self.model.omega22(key)
            self._omega[key] = value
        return value

    def amplitude_complex(self, tau: float) -> complex:
        return self.model.amplitude_complex(tau)


class _CachedPlanning22:
    """Memoize identical 2,2 probes without changing planner decisions."""

    def __init__(self, model: IMRPhenomT22):
        self.model = model
        self.omega_ring = model.omega_ring
        self.alpha1rd = model.alpha1rd
        self._omega: dict[float, float] = {}
        self._amplitude: dict[float, complex] = {}

    def omega22(self, tau: float) -> float:
        key = float(tau)
        value = self._omega.get(key)
        if value is None:
            value = self.model.omega22(key)
            self._omega[key] = value
        return value

    def amplitude_complex(self, tau: float) -> complex:
        key = float(tau)
        value = self._amplitude.get(key)
        if value is None:
            value = self.model.amplitude_complex(key)
            self._amplitude[key] = value
        return value


def _normalize_modes(modes: Sequence[int | tuple[int, int]] | str) -> tuple[tuple[int, int], ...]:
    if isinstance(modes, str):
        name = modes.strip().lower()
        if name in {"default", "all", "hm"}:
            return DEFAULT_POSITIVE_MODES
        if name in {"22", "22pair", "2|2|"}:
            return ((2, 2),)
        tokens = [token.strip() for token in name.split(",") if token.strip()]
        codes = tuple(int(token) for token in tokens)
    else:
        codes = tuple(10 * value[0] + abs(value[1]) if isinstance(value, tuple) else int(value) for value in modes)
    selected: list[tuple[int, int]] = []
    for code in codes:
        if code not in MODE_BY_CODE:
            raise ValueError(f"unsupported folded THM carrier {code}")
        mode = MODE_BY_CODE[code]
        if mode not in selected:
            selected.append(mode)
    if not selected:
        raise ValueError("at least one THM carrier must be selected")
    return tuple(selected)


def _ring_data(model: IMRPhenomTHM, modes: tuple[tuple[int, int], ...]) -> tuple[np.ndarray, np.ndarray]:
    from phenomthm import P_ALPHA1, P_OMEGA_RING

    ring: list[float] = []
    damping: list[float] = []
    for mode in modes:
        if mode == (2, 2):
            ring.append(model.mode22.omega_ring / (2.0 * PI * model.total_mass))
            damping.append(model.mode22.alpha1rd / model.total_mass)
        else:
            coeffs = model._positive_states[mode].phase_coeffs
            if coeffs is None:
                raise RuntimeError(f"missing phase coefficients for mode {mode}")
            ring.append(float(coeffs[P_OMEGA_RING]) / (2.0 * PI * model.total_mass))
            damping.append(float(coeffs[P_ALPHA1]) / model.total_mass)
    return np.asarray(ring), np.asarray(damping)


def build_thm_intrinsic_grid(
    source: SourceParams = SourceParams(),
    shape: WDMShape = WDMShape(),
    *,
    modes: Sequence[int | tuple[int, int]] | str = "default",
    nsmax: int = 10000,
    coefficient_backend: str = "auto",
    tdi_generation: int = 1,
    constellation_override: tuple[np.ndarray, list, list, list] | None = None,
) -> tuple[THMIntrinsicGrid, tuple[np.ndarray, list, list, list]]:
    """Build one source grid that is safe for every selected folded carrier."""

    selected_modes = _normalize_modes(modes)
    constellation = (build_constellation_splines(shape) if constellation_override is None
                     else constellation_override)
    constellation_time, _l_splines, p_splines, _v_splines = constellation
    m1, m2, chi1, chi2 = source.masses_seconds()
    total_mass = m1 + m2
    planning_22 = IMRPhenomT22.from_masses(
        m1, m2, chi1, chi2, coefficient_backend=coefficient_backend
    )
    cached_planning_22 = _CachedPlanning22(planning_22)
    planning_models = tuple(
        _ScaledPlanningModel(cached_planning_22, mode[1]) for mode in selected_modes
    )
    model = IMRPhenomTHM(
        m1,
        m2,
        chi1,
        chi2,
        modes=selected_modes,
        coefficient_backend=coefficient_backend,
    )

    tstart = max(0.0, float(constellation_time[0]) + CONSTELLATION_LIGHT_TIME_SECONDS)
    tstop = min(
        shape.Tobs,
        source.tc + RESPONSE_LATE_MARGIN_SECONDS + 1000.0 * total_mass,
        float(constellation_time[-1]) - CONSTELLATION_LIGHT_TIME_SECONDS,
    )
    if tstop <= tstart:
        raise ValueError("the requested merger and WDM observation have no overlapping response interval")
    source_bounds = barycenter_time(
        np.asarray((tstart, tstop), dtype=np.float64), source, p_splines
    )

    # The largest selected |m| drives the intrinsic curvature grid.  The
    # detector grid below still checks every carrier independently because
    # their TDI transfer crossings occur at different source times.
    driver = max(planning_models, key=lambda item: item.scale)
    model_time = _build_intrinsic_model_grid(
        driver,
        float(source_bounds[0]),
        float(source_bounds[1]),
        source.tc,
        total_mass,
        2 * nsmax + 4,
    )
    switches: list[float] = []
    for planning in planning_models:
        source_switch = _intrinsic_tdi_switch_source_time(
            planning,
            model_time,
            float(model_time[0]),
            float(model_time[-1]),
            source.tc,
            total_mass,
        )
        switches.append(
            min(tstop, max(tstart, source_switch - INTRINSIC_TDI_SWITCH_GUARD_SECONDS))
        )

    _, _, kv = sky_vectors(source)
    const_x = np.ascontiguousarray(p_splines[0].x, dtype=np.float64)
    p_coeffs = _stack_spline_coefficients(p_splines)
    kv = np.ascontiguousarray(kv, dtype=np.float64)
    detector_points: list[float] = []
    t = tstart
    last_step = AP_DTM_MAX
    previous_step = AP_DTM_MAX
    history_count = 0
    while True:
        if len(detector_points) >= nsmax:
            raise RuntimeError("shared THM detector-response grid exceeded nsmax")
        detector_points.append(t)
        if t >= tstop:
            break
        proposal = _detector_step_proposal(last_step, previous_step, history_count)
        candidates: list[float] = []
        for planning, switch in zip(planning_models, switches):
            if t < switch:
                candidates.append(min(AP_DTM_MAX, switch - t))
            else:
                candidates.append(
                    _detector_adaptive_step(
                        planning,
                        t,
                        tstop,
                        proposal,
                        source.tc,
                        total_mass,
                        abs(planning.omega_ring) / (2.0 * PI * total_mass),
                        const_x,
                        p_coeffs,
                        kv,
                    )
                )
        step = min(candidates)
        tnext = min(t + max(step, DTMIN), tstop)
        if tnext <= t:
            tnext = min(t + DTMIN, tstop)
        if t >= min(switches):
            previous_step = last_step
            last_step = tnext - t
            history_count += 1
        t = tnext

    detector_all = np.asarray(detector_points, dtype=np.float64)
    bary_all = barycenter_time(detector_all, source, p_splines)
    exact = np.zeros(detector_all.size, dtype=bool)
    exact[0] = True
    exact[-1] = True
    spacing = np.diff(detector_all)
    eps = 1.0e-10 * AP_DTM_MAX
    exact[:-1] |= spacing < AP_DTM_MAX - eps
    exact[1:] |= spacing < AP_DTM_MAX - eps
    waveform_time = _unique_sorted_times(np.concatenate((model_time, bary_all[exact])))

    evaluated = model.evaluate_times(
        waveform_time,
        tc=source.tc,
        phic=source.phic,
        distance_gpc=source.distance_gpc,
    )
    model_amplitude = np.stack([evaluated[mode].amplitude for mode in selected_modes])
    model_phase = np.stack([evaluated[mode].phase for mode in selected_modes])
    model_frequency = np.stack(
        [np.abs(evaluated[mode].omega) / (2.0 * PI) for mode in selected_modes]
    )
    amplitude = np.stack(
        [make_ap_spline(waveform_time, values)(bary_all) for values in model_amplitude]
    )
    phase = np.stack(
        [make_ap_spline(waveform_time, values)(bary_all) for values in model_phase]
    )
    frequency = np.stack(
        [make_ap_spline(waveform_time, values)(bary_all) for values in model_frequency]
    )

    # Preserve the C trimming guard at the upper edge of the source spline.
    trim = 0
    while True:
        trim += 1
        if not detector_all[detector_all.size - trim] > waveform_time[-1]:
            break
    last = detector_all.size - trim
    if last < 3:
        raise RuntimeError("too few response samples remain after the endpoint guard")
    ring, damping = _ring_data(model, selected_modes)
    intrinsic = THMIntrinsicGrid(
        waveform_time=np.ascontiguousarray(waveform_time),
        spline_time=np.ascontiguousarray(bary_all),
        response_time=np.ascontiguousarray(bary_all[:last]),
        detector_time=np.ascontiguousarray(detector_all[:last]),
        modes=selected_modes,
        amplitude=np.ascontiguousarray(amplitude),
        phase=np.ascontiguousarray(phase),
        frequency=np.ascontiguousarray(frequency),
        model_amplitude=np.ascontiguousarray(model_amplitude),
        model_phase=np.ascontiguousarray(model_phase),
        model_frequency=np.ascontiguousarray(model_frequency),
        ring_frequency_hz=np.ascontiguousarray(ring),
        damping_rate_hz=np.ascontiguousarray(damping),
        model=model,
        response_switch_detector_time=np.asarray(switches),
        exact_response_samples=int(np.count_nonzero(exact)),
    )
    return intrinsic, constellation


def _swsh_minus2(theta: float, phi: float, ell: int, emm: int) -> complex:
    """The explicit Phentax/C conventions for the modes used by THM."""

    c = math.cos(theta)
    s = math.sin(theta)
    ch = math.cos(0.5 * theta)
    sh = math.sin(0.5 * theta)
    if (ell, emm) == (2, 2):
        angular = math.sqrt(5.0 / (64.0 * PI)) * (1.0 + c) ** 2
    elif (ell, emm) == (2, -2):
        angular = math.sqrt(5.0 / (64.0 * PI)) * (1.0 - c) ** 2
    elif (ell, emm) == (2, 1):
        angular = math.sqrt(5.0 / (16.0 * PI)) * s * (1.0 + c)
    elif (ell, emm) == (2, -1):
        angular = math.sqrt(5.0 / (16.0 * PI)) * s * (1.0 - c)
    elif (ell, emm) == (3, 3):
        angular = -math.sqrt(21.0 / (128.0 * PI)) * (1.0 + c) ** 2 * s
    elif (ell, emm) == (3, -3):
        angular = math.sqrt(21.0 / (128.0 * PI)) * (1.0 - c) ** 2 * s
    elif (ell, emm) == (4, 4):
        angular = 3.0 * math.sqrt(7.0 / PI) * ch**6 * sh**2
    elif (ell, emm) == (4, -4):
        angular = 3.0 * math.sqrt(7.0 / PI) * sh**6 * ch**2
    elif (ell, emm) == (5, 5):
        angular = -math.sqrt(330.0 / PI) * ch**7 * sh**3
    elif (ell, emm) == (5, -5):
        angular = math.sqrt(330.0 / PI) * ch**3 * sh**7
    else:
        raise ValueError(f"unsupported spin-weighted harmonic {(ell, emm)}")
    return angular * complex(math.cos(emm * phi), math.sin(emm * phi))


def folded_projection(source: SourceParams, modes: tuple[tuple[int, int], ...]) -> np.ndarray:
    """Return the eight plus/cross quadrature coefficients for each pair."""

    theta = math.acos(max(-1.0, min(1.0, source.cos_inclination)))
    orbital_azimuth = 0.5 * PI
    rotation = complex(math.cos(2.0 * source.polarization), math.sin(2.0 * source.polarization))
    out = np.empty((len(modes), 8), dtype=np.float64)
    for i, (ell, emm) in enumerate(modes):
        ypos = _swsh_minus2(theta, orbital_azimuth, ell, emm) * rotation
        yneg = _swsh_minus2(theta, orbital_azimuth, ell, -emm) * rotation
        parity = 1.0 if ell % 2 == 0 else -1.0
        hp_cos = ypos.real + parity * yneg.real
        hp_sin = ypos.imag - parity * yneg.imag
        hc_cos = -ypos.imag - parity * yneg.imag
        hc_sin = ypos.real - parity * yneg.real
        out[i] = (
            hp_cos,
            hp_sin,
            hc_cos,
            hc_sin,
            hp_sin,
            -hp_cos,
            hc_sin,
            -hc_cos,
        )
    return out


def _response_geometry(times: np.ndarray, source: SourceParams, constellation) -> tuple[np.ndarray, ...]:
    _constellation_time, l_splines, p_splines, v_splines = constellation
    u, v, kv = sky_vectors(source)
    eplus = np.outer(v, v) - np.outer(u, u)
    ecross = np.outer(u, v) + np.outer(v, u)
    arm_length = np.stack([spline(times) for spline in l_splines], axis=1)
    position = np.stack([spline(times) for spline in p_splines], axis=1).reshape(times.size, 3, 3)
    arms = np.stack([spline(times) for spline in v_splines], axis=1).reshape(times.size, 3, 3)
    kr = np.einsum("nij,j->ni", position, kv)
    kn = np.einsum("nij,j->ni", arms, kv)
    plus = np.einsum("nij,nik,jk->ni", arms, arms, eplus)
    cross = np.einsum("nij,nik,jk->ni", arms, arms, ecross)
    return (
        arm_length,
        kr,
        0.5 * plus / (1.0 + kn),
        0.5 * plus / (1.0 - kn),
        0.5 * cross / (1.0 + kn),
        0.5 * cross / (1.0 - kn),
    )


def _tdi_delays_and_coefficients(
    triple: tuple[int, int, int],
    times: np.ndarray,
    arm_length: np.ndarray,
    kr: np.ndarray,
    app: np.ndarray,
    apm: np.ndarray,
    acp: np.ndarray,
    acm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a, b, c = triple
    delays = np.stack(
        (
            times - kr[:, a] - 2.0 * arm_length[:, c] - 2.0 * arm_length[:, b],
            times - kr[:, b] - arm_length[:, c] - 2.0 * arm_length[:, b],
            times - kr[:, c] - arm_length[:, b] - 2.0 * arm_length[:, c],
            times - kr[:, a] - 2.0 * arm_length[:, b],
            times - kr[:, a] - 2.0 * arm_length[:, c],
            times - kr[:, c] - arm_length[:, b],
            times - kr[:, b] - arm_length[:, c],
            times - kr[:, a],
        ),
        axis=1,
    )
    coefp = np.stack(
        (
            app[:, c] - apm[:, b], -app[:, c] + apm[:, c],
            apm[:, b] - app[:, b], -apm[:, c] + apm[:, b],
            app[:, b] - app[:, c], -apm[:, b] + app[:, b],
            app[:, c] - apm[:, c], -app[:, b] + apm[:, c],
        ), axis=1,
    )
    coefc = np.stack(
        (
            acp[:, c] - acm[:, b], -acp[:, c] + acm[:, c],
            acm[:, b] - acp[:, b], -acm[:, c] + acm[:, b],
            acp[:, b] - acp[:, c], -acm[:, b] + acp[:, b],
            acp[:, c] - acm[:, c], -acp[:, b] + acm[:, c],
        ), axis=1,
    )
    return delays, coefp, coefc


def _response_from_modes(
    times: np.ndarray,
    source: SourceParams,
    intrinsic: THMIntrinsicGrid,
    constellation,
    *,
    extract_carrier_ap: bool,
    tdi_generation: int = 1,
) -> THMTDIGrid:
    geometry = _response_geometry(times, source, constellation)
    arm_length, kr, app, apm, acp, acm = geometry
    projection = folded_projection(source, intrinsic.modes)
    ncarrier = len(intrinsic.modes)
    raw_m = np.empty((3, ncarrier, times.size), dtype=np.float64)
    raw_mf = np.empty_like(raw_m)
    amp_splines = [make_ap_spline(intrinsic.spline_time, values) for values in intrinsic.amplitude]
    phase_splines = [make_ap_spline(intrinsic.spline_time, values) for values in intrinsic.phase]

    if tdi_generation == 2:
        model_amp = [make_ap_spline(intrinsic.waveform_time, row)
                     for row in intrinsic.model_amplitude]
        model_phase = [make_ap_spline(intrinsic.waveform_time, row)
                       for row in intrinsic.model_phase]

        def polarizations(source_time: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            live = ((source_time >= intrinsic.waveform_time[0]) &
                    (source_time <= intrinsic.waveform_time[-1]))
            oscillation = np.zeros((ncarrier, len(source_time)), dtype=np.complex128)
            if np.any(live):
                amplitude = np.stack([s(source_time[live]) for s in model_amp])
                phase = np.stack([s(source_time[live]) for s in model_phase])
                oscillation[:, live] = amplitude*np.exp(1j*phase)
            plus_coeff = projection[:, 0]-1j*projection[:, 1]
            cross_coeff = projection[:, 2]-1j*projection[:, 3]
            return oscillation*plus_coeff[:, None], oscillation*cross_coeff[:, None]

        analytic = complex_tdi2(times, source.ecliptic_colatitude,
                                source.ecliptic_longitude, constellation[2],
                                polarizations)
        raw_m[:] = analytic.real
        raw_mf[:] = -analytic.imag
    else:
        for channel, triple in enumerate(TRIPLES):
            delays, coefp, coefc = _tdi_delays_and_coefficients(
                triple, times, arm_length, kr, app, apm, acp, acm
            )
            for carrier in range(ncarrier):
                amplitude = np.asarray(amp_splines[carrier](delays), dtype=np.float64)
                phase = np.asarray(phase_splines[carrier](delays), dtype=np.float64)
                cp = np.cos(phase)
                sp = np.sin(phase)
                coeff = projection[carrier]
                hp = amplitude * (coeff[0] * cp + coeff[1] * sp)
                hc = amplitude * (coeff[2] * cp + coeff[3] * sp)
                hpf = amplitude * (coeff[4] * cp + coeff[5] * sp)
                hcf = amplitude * (coeff[6] * cp + coeff[7] * sp)
                raw_m[channel, carrier] = np.sum(hp * coefp + hc * coefc, axis=1)
                raw_mf[channel, carrier] = np.sum(hpf * coefp + hcf * coefc, axis=1)

    detector_reference = intrinsic.detector_time
    if times.shape != intrinsic.response_time.shape or not np.array_equal(times, intrinsic.response_time):
        detector_reference = times
    reference = np.stack(
        [np.asarray(spline(detector_reference), dtype=np.float64) for spline in phase_splines]
    )
    amplitude_at_response = np.stack(
        [np.asarray(spline(times), dtype=np.float64) for spline in amp_splines]
    )
    signed = np.empty_like(raw_m)
    channel_phase = np.empty_like(raw_m)
    if extract_carrier_ap:
        for channel in range(3):
            for carrier in range(ncarrier):
                if NUMBA_AVAILABLE:
                    signed[channel, carrier], offset = _extract_ap_jit(
                        np.ascontiguousarray(raw_m[channel, carrier]),
                        np.ascontiguousarray(raw_mf[channel, carrier]),
                        np.ascontiguousarray(reference[carrier]),
                    )
                else:
                    signed[channel, carrier], offset = extract_ap(
                        raw_m[channel, carrier], raw_mf[channel, carrier], reference[carrier]
                    )
                channel_phase[channel, carrier] = reference[carrier] + offset
    else:
        signed.fill(np.nan)
        channel_phase.fill(np.nan)

    frequency_at_response = np.stack(
        [make_ap_spline(intrinsic.spline_time, values)(times) for values in intrinsic.frequency]
    )
    return THMTDIGrid(
        time=np.ascontiguousarray(times),
        detector_time=np.ascontiguousarray(detector_reference),
        modes=intrinsic.modes,
        intrinsic_amplitude=np.ascontiguousarray(amplitude_at_response),
        intrinsic_phase=np.ascontiguousarray(reference),
        intrinsic_frequency=np.ascontiguousarray(frequency_at_response),
        raw_m=raw_m,
        raw_mf=raw_mf,
        signed_amplitude=signed,
        channel_phase=channel_phase,
        reference_phase=reference,
        channel_strain=np.sum(raw_m, axis=1),
    )


def compute_thm_tdi_grid(
    source: SourceParams,
    intrinsic: THMIntrinsicGrid,
    constellation,
    *,
    extract_carrier_ap: bool = True,
    tdi_generation: int = 1,
) -> THMTDIGrid:
    """Apply the folded-carrier TDI response on the shared sparse grid."""

    return _response_from_modes(
        intrinsic.response_time,
        source,
        intrinsic,
        constellation,
        extract_carrier_ap=extract_carrier_ap,
        tdi_generation=tdi_generation,
    )


def _empty_channel(shape: WDMShape) -> WDMChannel:
    empty_i = np.empty(0, dtype=np.int64)
    empty = np.empty(0, dtype=np.float64)
    nmid = np.full(shape.nf, -1, dtype=np.int64)
    nsize = np.zeros(shape.nf, dtype=np.int64)
    return WDMChannel(empty, empty, empty, nmid, nsize, empty_i, empty_i, empty)


def _next_power_of_two(value: int) -> int:
    return 1 if value <= 1 else 1 << (int(value) - 1).bit_length()


def _smooth_step(times: np.ndarray, start: float, stop: float) -> np.ndarray:
    """Half-cosine step used by adjacent blocks and the endpoint."""

    values = np.asarray(times, dtype=np.float64)
    if stop <= start:
        return np.asarray(values >= start, dtype=np.float64)
    out = np.zeros_like(values)
    out[values >= stop] = 1.0
    transition = (values > start) & (values < stop)
    x = (values[transition] - start) / (stop - start)
    out[transition] = 0.5 * (1.0 - np.cos(PI * x))
    return out


def _track_range_linear(
    times: np.ndarray,
    frequency: np.ndarray,
    lower: float,
    upper: float,
) -> tuple[float, float]:
    """Conservative extrema of a sampled track without spline overshoot."""

    lower = max(float(lower), float(times[0]))
    upper = min(float(upper), float(times[-1]))
    if upper <= lower:
        value = float(np.interp(lower, times, frequency))
        return value, value
    inside = (times > lower) & (times < upper) & np.isfinite(frequency)
    values = np.concatenate(
        (
            np.asarray((np.interp(lower, times, frequency),), dtype=np.float64),
            np.asarray(frequency[inside], dtype=np.float64),
            np.asarray((np.interp(upper, times, frequency),), dtype=np.float64),
        )
    )
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise RuntimeError("no finite THM track samples in FFT block")
    return float(np.min(values)), float(np.max(values))


def _centered_even_heterodyne_layer(layer_lo: int, layer_hi: int) -> int:
    center = 0.5 * float(layer_lo + layer_hi)
    return max(0, 2 * int(math.floor(0.5 * center + 0.5)))


def _packet_plan_from_bounds(
    jlo: int,
    jhi: int,
    shape: WDMShape,
) -> tuple[int, int]:
    """Return the standard even-centered, power-of-two packet container."""

    jlo = max(int(jlo), 0)
    jhi = min(int(jhi), shape.nt - 1)
    needed = max(jhi - jlo + 1, 1)
    block = _next_power_of_two(needed + 2)
    block = max(block, 2 * shape.mult)
    block = min(block, shape.nt)
    center = (jlo + jhi + 1) // 2
    center -= center % 2
    center = max(center, block // 2)
    center = min(center, shape.nt - block // 2)
    center -= center % 2
    while center - block // 2 > jlo:
        center -= 2
    while center + block // 2 - 1 < jhi:
        center += 2
    center = max(center, block // 2)
    center = min(center, shape.nt - block // 2)
    center -= center % 2
    return center, block


def _merge_sparse_channels(
    first: WDMChannel,
    second: WDMChannel,
    shape: WDMShape,
) -> WDMChannel:
    """Add two sparse channels, combining coefficients on shared pixels."""

    if first.values.size == 0:
        return second
    if second.values.size == 0:
        return first
    listn = np.concatenate((first.listn, second.listn))
    listm = np.concatenate((first.listm, second.listm))
    values = np.concatenate((first.values, second.values))
    order = np.lexsort((listn, listm))
    listn = listn[order]
    listm = listm[order]
    values = values[order]
    new_pixel = np.ones(values.size, dtype=bool)
    new_pixel[1:] = (listn[1:] != listn[:-1]) | (listm[1:] != listm[:-1])
    starts = np.flatnonzero(new_pixel)
    values = np.add.reduceat(values, starts)
    listn = listn[starts]
    listm = listm[starts]
    nmid = first.nmid.copy()
    nsize = first.nsize.copy()
    merge_pixel_plans(nmid, nsize, second.nmid, second.nsize, shape)
    empty = np.empty(0, dtype=np.float64)
    return WDMChannel(empty, empty, empty, nmid, nsize, listn, listm, values)


@njit(cache=True)
def _accumulate_partition_bins(
    data: np.ndarray,
    fgrid: np.ndarray,
    spectrum: np.ndarray,
    frequency_step: float,
    heterodyne_layer: int,
    sample_dt: float,
    phase_time: float,
) -> bool:
    """Add one native block spectrum to a packet's local Fourier bins."""

    nfft = spectrum.shape[1]
    heterodyne_frequency = heterodyne_layer * frequency_step
    used = False
    for bin_index in range(1, fgrid.size):
        frequency = fgrid[bin_index]
        residual = frequency - heterodyne_frequency
        if frequency <= 0.0 or abs(residual) >= 0.5 / sample_dt:
            continue
        bin_float = residual * nfft * sample_dt
        native_bin = int(round(bin_float))
        if abs(bin_float - native_bin) >= 1.0e-7:
            continue
        phase = 2.0 * math.pi * frequency * phase_time
        rotation = complex(math.cos(phase), math.sin(phase))
        native_bin %= nfft
        for channel in range(data.shape[0]):
            data[channel, bin_index] += spectrum[channel, native_bin] * rotation
        used = True
    return used


def _partition_blocks_to_wdm(
    blocks: Sequence[_THMPartitionSpectrum],
    nmid: np.ndarray,
    nsize: np.ndarray,
    shape: WDMShape,
    channel_count: int,
    *,
    quadratures: bool = False,
    frequency_moment_order: int = 0,
) -> list[WDMChannel] | tuple[list[WDMChannel], ...]:
    """Assemble native block bins before each local Meyer packet IFFT."""

    if frequency_moment_order not in (0, 1) or (frequency_moment_order and not quadratures):
        raise ValueError("frequency moments require quadratures and order 0 or 1")

    groups: dict[int, list[int]] = {}
    time_pixel = shape.DT
    frequency_step = shape.DF
    frequency_band = shape.FB
    observation_time = shape.Tobs
    block_frequency_lower = np.asarray(
        [block.frequency_min - block.frequency_pad for block in blocks], dtype=np.float64
    )
    block_frequency_upper = np.asarray(
        [block.frequency_max + block.frequency_pad for block in blocks], dtype=np.float64
    )
    block_time_lower = np.asarray([block.nonzero_start for block in blocks], dtype=np.float64)
    block_time_upper = np.asarray([block.nonzero_stop for block in blocks], dtype=np.float64)
    for layer in np.flatnonzero(nsize > 0):
        groups.setdefault(int(nsize[layer]), []).append(int(layer))
    output_nmid = np.full(shape.nf, -1, dtype=np.int64)
    output_nsize = np.zeros(shape.nf, dtype=np.int64)
    listn_parts: list[np.ndarray] = []
    listm_parts: list[np.ndarray] = []
    value_parts: list[list[np.ndarray]] = [[] for _ in range(channel_count)]
    quadrature_parts: list[list[np.ndarray]] = [[] for _ in range(channel_count)] if quadratures else []
    moment_parts: list[list[np.ndarray]] = [[] for _ in range(channel_count)] if frequency_moment_order else []
    moment_quadrature_parts: list[list[np.ndarray]] = [[] for _ in range(channel_count)] if frequency_moment_order else []

    for packet_size, layers in groups.items():
        half = packet_size // 2
        base_n = np.arange(packet_size, dtype=np.int64)
        offsets = np.arange(-half, half, dtype=np.float64)
        packet_df = 1.0 / (float(packet_size) * time_pixel)
        omega = np.arange(0, half + 1, dtype=np.float64) * (2.0 * PI * packet_df)
        window = phitilde(omega, shape)[np.abs(base_n - half)]
        window[0] = 0.0
        scale = math.sqrt(8.0 * PI / 15.0) / (float(packet_size) * time_pixel)
        for layer in layers:
            center = int(nmid[layer])
            packet_start = (float(center) - float(half)) * time_pixel
            packet_stop = packet_start + float(packet_size) * time_pixel
            fgrid = float(layer) * frequency_step + offsets * packet_df
            data = np.zeros((channel_count, packet_size), dtype=np.complex128)
            used = False
            layer_lower = float(layer) * frequency_step - frequency_band
            layer_upper = float(layer) * frequency_step + frequency_band
            candidate_blocks = np.flatnonzero(
                (layer_upper >= block_frequency_lower)
                & (layer_lower <= block_frequency_upper)
                & (packet_stop > block_time_lower)
                & (packet_start < block_time_upper)
            )
            for block_index in candidate_blocks:
                block = blocks[int(block_index)]
                used |= _accumulate_partition_bins(
                    data,
                    fgrid,
                    block.spectrum,
                    frequency_step,
                    block.heterodyne_layer,
                    block.sample_dt,
                    observation_time + packet_start - block.block_start,
                )
            if not used:
                continue
            packet = np.fft.ifft(data * window[np.newaxis, :], axis=1) * packet_size
            even = ((base_n + layer) % 2) == 0
            imag_sign = 1.0 if layer % 2 == 0 else -1.0
            coefficients = scale * np.where(
                even[np.newaxis, :], packet.real, imag_sign * packet.imag
            )
            if quadratures:
                shifted = scale * np.where(
                    even[np.newaxis, :], -packet.imag, imag_sign * packet.real
                )
            if frequency_moment_order:
                moment_packet = np.fft.ifft(
                    data * (window * offsets * packet_df)[np.newaxis, :], axis=1
                ) * packet_size
                moment_coefficients = scale * np.where(
                    even[np.newaxis, :], moment_packet.real, imag_sign * moment_packet.imag
                )
                moment_shifted = scale * np.where(
                    even[np.newaxis, :], -moment_packet.imag, imag_sign * moment_packet.real
                )
            output_time = base_n + center - half
            keep = (output_time >= 0) & (output_time < shape.nt)
            listn_parts.append(output_time[keep])
            listm_parts.append(np.full(np.count_nonzero(keep), layer, dtype=np.int64))
            for channel in range(channel_count):
                value_parts[channel].append(
                    np.asarray(coefficients[channel, keep], dtype=np.float64)
                )
                if quadratures:
                    quadrature_parts[channel].append(
                        np.asarray(shifted[channel, keep], dtype=np.float64)
                    )
                if frequency_moment_order:
                    moment_parts[channel].append(
                        np.asarray(moment_coefficients[channel, keep], dtype=np.float64)
                    )
                    moment_quadrature_parts[channel].append(
                        np.asarray(moment_shifted[channel, keep], dtype=np.float64)
                    )
            output_nmid[layer] = center
            output_nsize[layer] = packet_size

    if not listn_parts:
        empty_channels = [_empty_channel(shape) for _ in range(channel_count)]
        if quadratures:
            if frequency_moment_order:
                return tuple(
                    [_empty_channel(shape) for _ in range(channel_count)]
                    for _ in range(4)
                )
            return empty_channels, [_empty_channel(shape) for _ in range(channel_count)]
        return empty_channels
    listn = np.concatenate(listn_parts)
    listm = np.concatenate(listm_parts)
    empty = np.empty(0, dtype=np.float64)
    channels = [
        WDMChannel(
            empty,
            empty,
            empty,
            output_nmid.copy(),
            output_nsize.copy(),
            listn.copy(),
            listm.copy(),
            np.concatenate(value_parts[channel]),
        )
        for channel in range(channel_count)
    ]
    if not quadratures:
        return channels
    quadrature_channels = [
        WDMChannel(
            empty,
            empty,
            empty,
            output_nmid.copy(),
            output_nsize.copy(),
            listn.copy(),
            listm.copy(),
            np.concatenate(quadrature_parts[channel]),
        )
        for channel in range(channel_count)
    ]
    if not frequency_moment_order:
        return channels, quadrature_channels

    def moment_channels(parts: list[list[np.ndarray]]) -> list[WDMChannel]:
        return [
            WDMChannel(
                empty,
                empty,
                empty,
                output_nmid.copy(),
                output_nsize.copy(),
                listn.copy(),
                listm.copy(),
                np.concatenate(parts[channel]),
            )
            for channel in range(channel_count)
        ]

    return (
        channels,
        quadrature_channels,
        moment_channels(moment_parts),
        moment_channels(moment_quadrature_parts),
    )


def _build_response_envelope_splines(
    tdi: THMTDIGrid,
    intrinsic: THMIntrinsicGrid,
    channel_indices: Sequence[int],
) -> tuple[list, list, list]:
    """Factor each sparse TDI family into carrier phasor times envelope."""

    real_splines: list = []
    imag_splines: list = []
    phase_splines: list = []
    for carrier in range(len(tdi.modes)):
        # The response grid resolves the slow detector envelope, not the
        # increasingly rapid carrier phase.  Restore the latter from the
        # independently refined intrinsic spline, as in the C implementation.
        phase_spline = make_ap_spline(
            intrinsic.spline_time, intrinsic.phase[carrier]
        )
        phase_splines.append(phase_spline)
        phase = np.asarray(phase_spline(tdi.time), dtype=np.float64)
        phasor_conjugate = np.exp(-1j * phase)
        for channel in channel_indices:
            # raw_mf is the +pi/2 folded-carrier response, hence
            # raw_m - i raw_mf is the positive-frequency analytic signal.
            analytic = (
                np.asarray(tdi.raw_m[channel, carrier], dtype=np.float64)
                - 1j * np.asarray(tdi.raw_mf[channel, carrier], dtype=np.float64)
            )
            envelope = analytic * phasor_conjugate
            real_splines.append(make_ap_spline(tdi.time, envelope.real))
            imag_splines.append(make_ap_spline(tdi.time, envelope.imag))
    return real_splines, imag_splines, phase_splines


def _partition_pixel_plans(
    tdi: THMTDIGrid,
    endpoint_start: float,
    endpoint_rise: float,
    waveform_stop: float,
    shape: WDMShape,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Construct compact early and common-endpoint packet supports."""

    early_mid = np.full(shape.nf, -1, dtype=np.int64)
    early_size = np.zeros(shape.nf, dtype=np.int64)
    endpoint_stop = min(endpoint_start + endpoint_rise, float(tdi.time[-1]))
    endpoint_frequency_low = math.inf
    endpoint_frequency_high = 0.0
    for track in tdi.intrinsic_frequency:
        local_mid, local_size = wdm_pixels_range(
            tdi.time,
            track,
            float(tdi.time[0]),
            endpoint_stop,
            shape,
        )
        merge_pixel_plans(early_mid, early_size, local_mid, local_size, shape)
        mask = (
            (tdi.time >= endpoint_start)
            & (tdi.time <= waveform_stop)
            & np.isfinite(track)
            & (track > 0.0)
        )
        if np.any(mask):
            endpoint_frequency_low = min(
                endpoint_frequency_low, float(np.min(track[mask]))
            )
            endpoint_frequency_high = max(
                endpoint_frequency_high, float(np.max(track[mask]))
            )
    if not math.isfinite(endpoint_frequency_low):
        endpoint_frequency_low = max(shape.DF, float(np.min(tdi.intrinsic_frequency[:, -1])))
        endpoint_frequency_high = float(np.max(tdi.intrinsic_frequency[:, -1]))
    margin = shape.FB + PARTITION_FFT_MARGIN_CYCLES / max(endpoint_rise, shape.dt)
    endpoint_frequency_start = max(0.0, endpoint_frequency_low - margin)
    endpoint_frequency_stop = min(
        (shape.nf - 1) * shape.DF + shape.FB,
        endpoint_frequency_high + margin,
    )
    endpoint_mid = np.full(shape.nf, -1, dtype=np.int64)
    endpoint_size = np.zeros(shape.nf, dtype=np.int64)
    mlo = max(1, int(math.ceil((endpoint_frequency_start - shape.FB) / shape.DF)))
    mhi = min(
        shape.nf - 1,
        int(math.floor((endpoint_frequency_stop + shape.FB) / shape.DF)),
    )
    center, packet_size = _packet_plan_from_bounds(
        int(math.floor(endpoint_start / shape.DT)),
        int(math.ceil(waveform_stop / shape.DT)),
        shape,
    )
    if mhi >= mlo:
        endpoint_mid[mlo : mhi + 1] = center
        endpoint_size[mlo : mhi + 1] = packet_size
    return (
        early_mid,
        early_size,
        endpoint_mid,
        endpoint_size,
        endpoint_frequency_start,
        endpoint_frequency_stop,
    )


def _build_partitioned_early_blocks(
    tdi: THMTDIGrid,
    intrinsic: THMIntrinsicGrid,
    channel_indices: Sequence[int],
    early_mid: np.ndarray,
    early_size: np.ndarray,
    endpoint_start: float,
    endpoint_rise: float,
    shape: WDMShape,
    *,
    bandwidth_hz: float,
    roll_seconds: float,
) -> tuple[list[_THMPartitionSpectrum], list[THMPartitionBlockDiagnostic], int]:
    """Build all per-carrier blocks from shared Cartesian response envelopes."""

    if bandwidth_hz <= 2.0 * shape.FB:
        raise ValueError("partition FFT bandwidth must exceed the Meyer support width")
    if roll_seconds <= 0.0:
        raise ValueError("partition FFT roll must be positive")
    times = np.asarray(tdi.time, dtype=np.float64)
    time_start = max(0.0, float(times[0]))
    plan_stop = min(float(endpoint_start), float(times[-1]))
    endpoint_support_stop = min(
        float(endpoint_start) + float(endpoint_rise), float(times[-1])
    )
    start_pixel = max(0, int(math.floor(time_start / shape.DT)))
    endpoint_pixel = min(shape.nt, int(math.floor(plan_stop / shape.DT)))
    endpoint_support_pixel = min(
        shape.nt, int(math.ceil(endpoint_support_stop / shape.DT))
    )
    if endpoint_pixel <= start_pixel:
        raise RuntimeError("THM partition ends before its first time pixel")
    roll_pixels = max(1, int(math.ceil(roll_seconds / shape.DT)))
    real_splines, imag_splines, phase_splines = _build_response_envelope_splines(
        tdi, intrinsic, channel_indices
    )
    channel_count = len(channel_indices)
    blocks: list[_THMPartitionSpectrum] = []
    diagnostics: list[THMPartitionBlockDiagnostic] = []
    total_samples = 0

    for carrier, track_values in enumerate(tdi.intrinsic_frequency):
        track = np.asarray(track_values, dtype=np.float64)
        tiles: list[tuple[int, int]] = []
        tile_lo = start_pixel
        while tile_lo < endpoint_pixel:
            remaining = endpoint_pixel - tile_lo

            def width_fits(width: int) -> bool:
                support_lo = max(tile_lo - roll_pixels, start_pixel)
                support_hi = min(
                    tile_lo + width + roll_pixels, endpoint_support_pixel
                )
                fmin, fmax = _track_range_linear(
                    times,
                    track,
                    float(support_lo) * shape.DT,
                    float(support_hi) * shape.DT,
                )
                center = max(abs(fmin), abs(fmax))
                intrinsic_margin = 3.0e-4 * center + 4.0 / (365.25 * 86400.0)
                return (
                    abs(fmax - fmin)
                    + 2.0 * (shape.FB + intrinsic_margin)
                    <= bandwidth_hz
                )

            width = 1 << (remaining.bit_length() - 1)
            selected = 0
            while width >= 1:
                if width_fits(width):
                    selected = width
                    break
                width //= 2
            selected = max(selected, 1)
            tile_hi = min(tile_lo + selected, endpoint_pixel)
            tiles.append((tile_lo, tile_hi))
            tile_lo = tile_hi

        boundaries = np.asarray(
            [float(tile_hi) * shape.DT for _, tile_hi in tiles], dtype=np.float64
        )
        boundary_roll = np.full(len(tiles), roll_seconds, dtype=np.float64)
        for index in range(max(len(tiles) - 1, 0)):
            available = boundaries[index + 1] - boundaries[index]
            boundary_roll[index] = max(
                min(roll_seconds, available, plan_stop - boundaries[index]),
                shape.DT,
            )

        for index, (tile_lo, tile_hi) in enumerate(tiles):
            nonzero_start = time_start if index == 0 else boundaries[index - 1]
            nonzero_stop = (
                boundaries[index] + boundary_roll[index]
                if index < len(tiles) - 1
                else endpoint_support_stop
            )
            nonzero_stop = min(nonzero_stop, float(times[-1]), shape.Tobs)
            if nonzero_stop <= nonzero_start:
                continue
            rise_start = time_start if index == 0 else boundaries[index - 1]
            rise_stop = (
                rise_start if index == 0 else rise_start + boundary_roll[index - 1]
            )
            fall_start = boundaries[index] if index < len(tiles) - 1 else endpoint_start
            fall_stop = (
                fall_start + boundary_roll[index]
                if index < len(tiles) - 1
                else endpoint_start + endpoint_rise
            )
            fmin, fmax = _track_range_linear(
                times, track, nonzero_start, nonzero_stop
            )
            frequency_pad = shape.FB
            for duration in (rise_stop - rise_start, fall_stop - fall_start):
                if duration > 0.0:
                    frequency_pad = max(
                        frequency_pad,
                        shape.FB + PARTITION_FFT_MARGIN_CYCLES / duration,
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
            local_bins = min(
                shape.nf,
                _next_power_of_two(
                    shifted_extent + PARTITION_FFT_NYQUIST_GUARD_LAYERS + 1
                ),
            )
            sample_start = math.floor(nonzero_start / shape.DT) * shape.DT
            support_pixels = max(
                1, int(math.ceil((nonzero_stop - sample_start) / shape.DT))
            )
            packet_time_pixels = _next_power_of_two(
                support_pixels + 2 * shape.mult
            )
            for layer in np.flatnonzero(early_size > 0):
                packet_start = (
                    float(early_mid[layer]) - 0.5 * float(early_size[layer])
                ) * shape.DT
                packet_stop = packet_start + float(early_size[layer]) * shape.DT
                layer_lower = float(layer) * shape.DF - shape.FB
                layer_upper = float(layer) * shape.DF + shape.FB
                if layer_upper < fmin - frequency_pad or layer_lower > fmax + frequency_pad:
                    continue
                if packet_stop <= nonzero_start or packet_start >= nonzero_stop:
                    continue
                packet_time_pixels = max(packet_time_pixels, int(early_size[layer]))
            packet_time_pixels = min(packet_time_pixels, shape.nt)
            sample_dt = shape.DT / float(local_bins)
            sample_count = packet_time_pixels * local_bins
            sample_times = sample_start + sample_dt * np.arange(sample_count)
            live = (
                (sample_times >= nonzero_start)
                & (sample_times <= nonzero_stop)
                & (sample_times >= times[0])
                & (sample_times <= times[-1])
                & (sample_times <= shape.Tobs)
            )
            residual = np.zeros(
                (channel_count, sample_count), dtype=np.complex128
            )
            if np.any(live):
                live_times = sample_times[live]
                weight = np.ones(live_times.size, dtype=np.float64)
                if rise_stop > rise_start:
                    weight *= _smooth_step(live_times, rise_start, rise_stop)
                if fall_stop > fall_start:
                    weight *= 1.0 - _smooth_step(live_times, fall_start, fall_stop)
                phase = np.asarray(phase_splines[carrier](live_times), dtype=np.float64)
                heterodyne_frequency = float(heterodyne_layer) * shape.DF
                demodulated_phasor = np.exp(
                    1j
                    * (
                        phase
                        - 2.0
                        * PI
                        * heterodyne_frequency
                        * (live_times - sample_start)
                    )
                )
                for local_channel in range(channel_count):
                    spline_index = carrier * channel_count + local_channel
                    envelope = np.asarray(
                        real_splines[spline_index](live_times), dtype=np.float64
                    ) + 1j * np.asarray(
                        imag_splines[spline_index](live_times), dtype=np.float64
                    )
                    residual[local_channel, live] = (
                        weight * envelope * demodulated_phasor
                    )
            spectrum = np.fft.fft(residual, axis=1) * sample_dt
            blocks.append(
                _THMPartitionSpectrum(
                    carrier_index=carrier,
                    block_start=sample_start,
                    nonzero_start=nonzero_start,
                    nonzero_stop=nonzero_stop,
                    frequency_min=fmin,
                    frequency_max=fmax,
                    frequency_pad=frequency_pad,
                    heterodyne_layer=heterodyne_layer,
                    sample_dt=sample_dt,
                    spectrum=spectrum,
                )
            )
            diagnostics.append(
                THMPartitionBlockDiagnostic(
                    carrier_index=carrier,
                    block_index=index,
                    sample_start=sample_start,
                    sample_dt=sample_dt,
                    sample_count=sample_count,
                    nonzero_start=nonzero_start,
                    nonzero_stop=nonzero_stop,
                    frequency_min=fmin,
                    frequency_max=fmax,
                    heterodyne_frequency=float(heterodyne_layer) * shape.DF,
                )
            )
            total_samples += sample_count
    return blocks, diagnostics, total_samples


def _build_partition_endpoint_block(
    source: SourceParams,
    intrinsic: THMIntrinsicGrid,
    constellation,
    channel_indices: Sequence[int],
    endpoint_mid: np.ndarray,
    endpoint_size: np.ndarray,
    endpoint_start: float,
    endpoint_rise: float,
    waveform_stop: float,
    endpoint_frequency_start: float,
    endpoint_frequency_stop: float,
    shape: WDMShape,
    tdi_generation: int = 1,
) -> tuple[_THMPartitionSpectrum, int]:
    """Build the one direct, all-carrier analytic endpoint transform."""

    sample_start = math.floor(endpoint_start / shape.DT) * shape.DT
    support_pixels = max(
        1, int(math.ceil((waveform_stop - sample_start) / shape.DT))
    )
    packet_time_pixels = _next_power_of_two(support_pixels + 2 * shape.mult)
    active_sizes = endpoint_size[endpoint_size > 0]
    if active_sizes.size:
        packet_time_pixels = max(packet_time_pixels, int(np.max(active_sizes)))
    packet_time_pixels = min(packet_time_pixels, shape.nt)
    sample_count = packet_time_pixels * shape.nf
    sample_times = sample_start + shape.dt * np.arange(sample_count)
    live = (
        (sample_times >= endpoint_start)
        & (sample_times <= waveform_stop)
        & (sample_times >= intrinsic.response_time[0])
    )
    analytic = np.zeros((len(channel_indices), sample_count), dtype=np.complex128)
    if np.any(live):
        live_times = sample_times[live]
        endpoint_tdi = _response_from_modes(
            live_times,
            source,
            intrinsic,
            constellation,
            extract_carrier_ap=False,
            tdi_generation=tdi_generation,
        )
        weight = _smooth_step(
            live_times, endpoint_start, endpoint_start + endpoint_rise
        )
        # The physical ringdown is negligible at waveform_stop.  A one-sample
        # numerical close avoids extending an ordinary long taper before the
        # merger, matching the current C endpoint construction.
        weight *= 1.0 - _smooth_step(
            live_times, waveform_stop, waveform_stop + shape.dt
        )
        for local_channel, channel in enumerate(channel_indices):
            value = np.sum(
                endpoint_tdi.raw_m[channel] - 1j * endpoint_tdi.raw_mf[channel],
                axis=0,
            )
            analytic[local_channel, live] = weight * value
    spectrum = np.fft.fft(analytic, axis=1) * shape.dt
    block = _THMPartitionSpectrum(
        carrier_index=-1,
        block_start=sample_start,
        nonzero_start=endpoint_start,
        nonzero_stop=waveform_stop + shape.dt,
        frequency_min=endpoint_frequency_start,
        frequency_max=endpoint_frequency_stop,
        frequency_pad=shape.FB,
        heterodyne_layer=0,
        sample_dt=shape.dt,
        spectrum=spectrum,
    )
    return block, sample_count


def _generate_partitioned_wdm(
    source: SourceParams,
    shape: WDMShape,
    intrinsic: THMIntrinsicGrid,
    tdi: THMTDIGrid,
    constellation,
    channel_indices: Sequence[int],
    endpoint_start: float,
    endpoint_rise: float,
    waveform_stop: float,
    *,
    bandwidth_hz: float,
    roll_seconds: float,
    tdi_generation: int = 1,
) -> tuple[dict[str, WDMChannel], dict[str, tuple[THMCarrierFourier, ...]], dict[str, float]]:
    """Complex partitioned-FFT path shared by aligned-spin THM carriers."""

    local_timings: dict[str, float] = {}
    endpoint_start = float(endpoint_start)
    endpoint_rise = float(endpoint_rise)
    waveform_stop = min(shape.Tobs, float(waveform_stop))
    local_timings["partition_endpoint_start"] = endpoint_start
    local_timings["partition_endpoint_rise"] = endpoint_rise
    local_timings["partition_waveform_stop"] = waveform_stop
    start = time.perf_counter()
    (
        early_mid,
        early_size,
        endpoint_mid,
        endpoint_size,
        endpoint_frequency_start,
        endpoint_frequency_stop,
    ) = _partition_pixel_plans(
        tdi, endpoint_start, endpoint_rise, waveform_stop, shape
    )
    local_timings["partition_plan"] = time.perf_counter() - start

    start = time.perf_counter()
    early_blocks, diagnostics, early_samples = _build_partitioned_early_blocks(
        tdi,
        intrinsic,
        channel_indices,
        early_mid,
        early_size,
        endpoint_start,
        endpoint_rise,
        shape,
        bandwidth_hz=bandwidth_hz,
        roll_seconds=roll_seconds,
    )
    local_timings["partition_early_fft"] = time.perf_counter() - start

    start = time.perf_counter()
    endpoint_block, endpoint_samples = _build_partition_endpoint_block(
        source,
        intrinsic,
        constellation,
        channel_indices,
        endpoint_mid,
        endpoint_size,
        endpoint_start,
        endpoint_rise,
        waveform_stop,
        endpoint_frequency_start,
        endpoint_frequency_stop,
        shape,
        tdi_generation=tdi_generation,
    )
    # The physical track does not bound the frequency width introduced by
    # the common endpoint taper.  Retain its measured FFT tail before
    # constructing the sparse Meyer packets.
    spectrum_power = np.sum(np.abs(endpoint_block.spectrum[:, :endpoint_samples//2+1])**2,
                            axis=0)
    total_spectrum_power = float(np.sum(spectrum_power))
    if total_spectrum_power > 0.0:
        tail = np.cumsum(spectrum_power[::-1])[::-1]
        occupied = np.flatnonzero(tail > 1.0e-8*total_spectrum_power)
        spectral_stop = (float(occupied[-1])/(endpoint_samples*shape.dt)
                         if occupied.size else endpoint_frequency_stop)
        spectral_stop = min(0.5/shape.dt,
                            max(endpoint_frequency_stop, spectral_stop+shape.FB))
        old_mhi = int(np.flatnonzero(endpoint_size > 0)[-1])
        new_mhi = min(shape.nf-1,
                      int(math.floor((spectral_stop+shape.FB)/shape.DF)))
        if new_mhi > old_mhi:
            endpoint_mid[old_mhi+1:new_mhi+1] = endpoint_mid[old_mhi]
            endpoint_size[old_mhi+1:new_mhi+1] = endpoint_size[old_mhi]
            endpoint_block.frequency_max = spectral_stop
        local_timings["partition_endpoint_tail_last_layer"] = float(new_mhi)
    local_timings["partition_endpoint_fft"] = time.perf_counter() - start

    start = time.perf_counter()
    early_channels = _partition_blocks_to_wdm(
        early_blocks, early_mid, early_size, shape, len(channel_indices)
    )
    endpoint_channels = _partition_blocks_to_wdm(
        (endpoint_block,), endpoint_mid, endpoint_size, shape, len(channel_indices)
    )
    channels_out: dict[str, WDMChannel] = {}
    carrier_fourier: dict[str, tuple[THMCarrierFourier, ...]] = {}
    for local_channel, channel in enumerate(channel_indices):
        name = CHANNELS[channel]
        channels_out[name] = _merge_sparse_channels(
            early_channels[local_channel], endpoint_channels[local_channel], shape
        )
        carrier_fourier[name] = ()
    local_timings["partition_packet_wdm"] = time.perf_counter() - start
    local_timings["partition_blocks"] = float(len(diagnostics) + 1)
    local_timings["partition_early_samples"] = float(early_samples)
    local_timings["partition_endpoint_samples"] = float(endpoint_samples)
    return channels_out, carrier_fourier, local_timings


def generate_thm_tdi_wdm(
    source: SourceParams = SourceParams(),
    shape: WDMShape = WDMShape(),
    channels: Iterable[str] = CHANNELS,
    *,
    modes: Sequence[int | tuple[int, int]] | str = "default",
    nsmax: int = 10000,
    coefficient_backend: str = "auto",
    compute_wdm: bool = True,
    wdm_method: str = "partitioned-fft",
    partition_bandwidth_hz: float = PARTITION_FFT_BANDWIDTH_HZ,
    partition_roll_seconds: float = PARTITION_FFT_ROLL_SECONDS,
    blend_endpoint: bool = DEFAULT_WDM_BLEND_ENDPOINT,
    blend_half_width_layers: float = DEFAULT_WDM_BLEND_HALF_WIDTH_LAYERS,
    tdi_generation: int = 1,
    constellation_override: tuple[np.ndarray, list, list, list] | None = None,
) -> THMTDIWDMResult:
    """Generate selected folded THM carriers, X/Y/Z TDI, and sparse WDM.

    ``partitioned-fft`` is the production default.  ``spa`` retains the older
    numerical-SPA plus common endpoint FFT for timing and regression studies.
    """

    requested = tuple(channel.upper() for channel in channels)
    if not requested or any(channel not in CHANNELS for channel in requested):
        raise ValueError("channels must be selected from X, Y, Z")
    method = wdm_method.strip().lower()
    if method in {"partition", "partitioned", "fft", "blocked-fft"}:
        method = "partitioned-fft"
    if method not in {"partitioned-fft", "spa"}:
        raise ValueError("wdm_method must be 'partitioned-fft' or 'spa'")
    if tdi_generation not in (1, 2):
        raise ValueError("tdi_generation must be 1 or 2")
    if tdi_generation == 2 and method != "partitioned-fft":
        raise ValueError("TDI-2 currently requires partitioned-fft WDM")
    timings: dict[str, float] = {}

    start = time.perf_counter()
    intrinsic, constellation = build_thm_intrinsic_grid(
        source,
        shape,
        modes=modes,
        nsmax=nsmax,
        coefficient_backend=coefficient_backend,
        tdi_generation=tdi_generation,
        constellation_override=constellation_override,
    )
    timings["adaptive_ap"] = time.perf_counter() - start

    start = time.perf_counter()
    tdi = compute_thm_tdi_grid(
        source,
        intrinsic,
        constellation,
        extract_carrier_ap=(not compute_wdm or method == "spa"),
        tdi_generation=tdi_generation,
    )
    timings["fast_tdi"] = time.perf_counter() - start

    if not compute_wdm:
        start_fd = time.perf_counter()
        m1, m2, _chi1, _chi2 = source.masses_seconds()
        total_mass = m1 + m2
        chirp_mass = (m1 * m2) ** (3.0 / 5.0) / total_mass ** (1.0 / 5.0)
        t_transition = source.tc + total_mass * T_CUT_FREQ
        carrier_fourier: dict[str, tuple[THMCarrierFourier, ...]] = {}
        for name in requested:
            channel = CHANNELS.index(name)
            spectra: list[THMCarrierFourier] = []
            for carrier, mode in enumerate(intrinsic.modes):
                track = tdi_frequency_track(
                    tdi.time,
                    tdi.channel_phase[channel, carrier],
                    tdi.intrinsic_frequency[carrier],
                )
                setup = transform_plan(
                    chirp_mass,
                    total_mass,
                    source.tc,
                    tdi.time,
                    2.0 * PI * track,
                    t_transition,
                    float(intrinsic.ring_frequency_hz[carrier]),
                    float(intrinsic.damping_rate_hz[carrier]),
                    float(tdi.time[-1]),
                    shape,
                    center_source_time=tdi.detector_time,
                )
                join = float(np.interp(float(setup[5]), tdi.time, track))
                setup_spa = setup.copy()
                jstop = min(max(int(setup_spa[4]), 1), tdi.time.size - 1)
                fstop = join + shape.FB
                while jstop + 1 < tdi.time.size and float(track[jstop]) < fstop:
                    jstop += 1
                setup_spa[4] = float(jstop)
                freq, phase, amplitude = ftran_spa_only(
                    setup_spa,
                    tdi.time,
                    tdi.signed_amplitude[channel, carrier],
                    tdi.channel_phase[channel, carrier],
                    shape,
                )
                freq, phase, amplitude = prune_nonincreasing_frequency_samples(
                    freq, phase, amplitude
                )
                spectra.append(
                    THMCarrierFourier(mode, freq, phase, amplitude, setup_spa, join)
                )
            carrier_fourier[name] = tuple(spectra)
        channels_out = {channel: _empty_channel(shape) for channel in requested}
        timings.update(
            fast_frequency_domain=time.perf_counter() - start_fd,
            fast_wdm_packets=0.0,
            fast_wdm=0.0,
        )
        return THMTDIWDMResult(
            source, shape, intrinsic, tdi, channels_out, carrier_fourier, timings
        )

    start_wdm = time.perf_counter()
    if method == "partitioned-fft":
        m1, m2, _chi1, _chi2 = source.masses_seconds()
        total_mass = m1 + m2
        delay_margin = (
            CONSTELLATION_LIGHT_TIME_SECONDS
            + (8.0 if tdi_generation == 2 else 4.0) * NOMINAL_ARM_LIGHT_TIME_SECONDS
        )
        endpoint_start = source.tc - delay_margin - 10000.0 * total_mass
        endpoint_rise = 5000.0 * total_mass
        waveform_stop = min(
            shape.Tobs,
            source.tc + CONSTELLATION_LIGHT_TIME_SECONDS + 1000.0 * total_mass,
        )
        if endpoint_start <= 0.0:
            endpoint_start = min(
                0.25 * waveform_stop,
                max(shape.DT, waveform_stop - endpoint_rise),
            )
        if endpoint_start >= waveform_stop:
            # A merger beyond the observation needs no physical plunge block,
            # but retaining one short complementary end block keeps the same
            # packet engine and avoids a special finite-observation boundary.
            endpoint_start = max(0.0, waveform_stop - endpoint_rise)
        if endpoint_start + endpoint_rise > waveform_stop:
            endpoint_rise = waveform_stop - endpoint_start
        if endpoint_rise <= 0.0:
            raise ValueError("partition endpoint has no overlap with the observation")
        channels_out, carrier_fourier, partition_timings = _generate_partitioned_wdm(
            source,
            shape,
            intrinsic,
            tdi,
            constellation,
            [CHANNELS.index(channel) for channel in requested],
            endpoint_start,
            endpoint_rise,
            waveform_stop,
            bandwidth_hz=float(partition_bandwidth_hz),
            roll_seconds=float(partition_roll_seconds),
            tdi_generation=tdi_generation,
        )
        timings.update(partition_timings)
        timings["fast_frequency_domain"] = 0.0
        timings["fast_wdm_packets"] = partition_timings["partition_packet_wdm"]
        timings["fast_wdm"] = time.perf_counter() - start_wdm
        return THMTDIWDMResult(
            source, shape, intrinsic, tdi, channels_out, carrier_fourier, timings
        )

    m1, m2, _chi1, _chi2 = source.masses_seconds()
    total_mass = m1 + m2
    chirp_mass = (m1 * m2) ** (3.0 / 5.0) / total_mass ** (1.0 / 5.0)
    t_transition = source.tc + total_mass * T_CUT_FREQ
    blend_width = float(blend_half_width_layers) * shape.DF if blend_endpoint else 0.0
    channel_indices = [CHANNELS.index(channel) for channel in requested]
    host: dict[int, list[dict[str, object]]] = {index: [] for index in channel_indices}
    driver_setup: np.ndarray | None = None
    driver_score = -1.0
    frequency_domain_time = 0.0

    for channel in channel_indices:
        for carrier, mode in enumerate(intrinsic.modes):
            track = tdi_frequency_track(
                tdi.time,
                tdi.channel_phase[channel, carrier],
                tdi.intrinsic_frequency[carrier],
            )
            setup = transform_plan(
                chirp_mass,
                total_mass,
                source.tc,
                tdi.time,
                2.0 * PI * track,
                t_transition,
                float(intrinsic.ring_frequency_hz[carrier]),
                float(intrinsic.damping_rate_hz[carrier]),
                float(tdi.time[-1]),
                shape,
                center_source_time=tdi.detector_time,
            )
            join = float(np.interp(float(setup[5]), tdi.time, track))
            if not math.isfinite(join) or join <= 0.0:
                valid = track[np.isfinite(track) & (track > 0.0)]
                join = float(valid[-1]) if valid.size else shape.DF
            host[channel].append(
                {"carrier": carrier, "mode": mode, "track": track, "setup": setup, "join": join}
            )
            score = float(setup[6]) / (float(setup[0]) * float(setup[1]))
            if score > driver_score:
                driver_score = score
                driver_setup = setup.copy()
    if driver_setup is None:
        raise RuntimeError("THM endpoint planner did not find a valid FFT driver")

    response_merger = float(
        np.interp(source.tc, tdi.center_source_time, tdi.ssb_output_time)
    )
    desired_replace_start: dict[int, float] = {}
    blend_lower_frequencies: list[float] = []
    channel_tracks: list[list[np.ndarray]] = []
    for channel in channel_indices:
        entries = host[channel]
        preferred = [float(entry["join"]) for entry in entries if entry["mode"][1] >= 3]
        if not preferred:
            preferred = [float(entry["join"]) for entry in entries]
        original_flat_time = float(driver_setup[2]) + float(driver_setup[3])
        original_taper_guard = max(
            float(np.interp(original_flat_time, tdi.time, np.asarray(entry["track"])))
            for entry in entries
        )
        # Preserve the former center-at-taper-guard handoff frequency.  The
        # endpoint planner below moves taper-flat earlier so that the *lower*
        # edge of the blend, rather than its midpoint, is safe at that same
        # frequency.
        desired_replace_start[channel] = max(min(preferred), original_taper_guard)
        blend_lower_frequencies.append(
            max(0.0, desired_replace_start[channel] - blend_width)
        )
        channel_tracks.append(
            [np.asarray(entry["track"], dtype=np.float64) for entry in entries]
        )
    driver_setup = plan_endpoint_taper_flat_time(
        driver_setup,
        tdi.time,
        channel_tracks,
        blend_lower_frequencies,
        response_merger,
    )

    endpoint_time = float(driver_setup[2]) + float(driver_setup[0]) * np.arange(
        int(driver_setup[1]), dtype=np.float64
    )
    endpoint_tdi = _response_from_modes(
        endpoint_time, source, intrinsic, constellation, extract_carrier_ap=False
    )
    rise = float(driver_setup[3])
    taper = np.ones(endpoint_time.size, dtype=np.float64)
    if rise > 0.0:
        local = endpoint_time - float(driver_setup[2])
        mask = local < rise
        taper[mask] = 0.5 * (1.0 - np.cos(PI * local[mask] / rise))

    channels_out: dict[str, WDMChannel] = {}
    carrier_fourier: dict[str, tuple[THMCarrierFourier, ...]] = {}
    packet_time = 0.0
    direct_start = float(driver_setup[2]) + rise + SHORTFFT_MERGER_TAPER_MARGIN_SECONDS
    direct_stop = float(driver_setup[2]) + float(driver_setup[0]) * float(driver_setup[1])
    f_replace_stop = min(driver_score, 0.5 / float(driver_setup[0]))

    for channel in channel_indices:
        entries = host[channel]
        replace_start = desired_replace_start[channel]
        flat_time = float(driver_setup[2]) + rise
        taper_guard = max(
            float(np.interp(flat_time, tdi.time, np.asarray(entry["track"]))) for entry in entries
        )
        if math.isfinite(taper_guard) and taper_guard > 0.0:
            # ``replace_start`` is the center of the coefficient blend.  Keep
            # its lower edge, not merely its midpoint, beyond the frequency
            # reached when the endpoint taper first becomes flat.
            replace_start = max(replace_start, taper_guard + blend_width)

        spa_stop_frequency = replace_start + shape.FB + blend_width
        spa_union_mid = np.full(shape.nf, -1, dtype=np.int64)
        spa_union_size = np.zeros(shape.nf, dtype=np.int64)
        endpoint_mid = np.full(shape.nf, -1, dtype=np.int64)
        endpoint_size = np.zeros(shape.nf, dtype=np.int64)
        spectra: list[THMCarrierFourier] = []
        dense = np.zeros((shape.nt, shape.nf + 1), dtype=np.float64)

        for entry in entries:
            fd_start = time.perf_counter()
            setup_spa = np.asarray(entry["setup"], dtype=np.float64).copy()
            track = np.asarray(entry["track"], dtype=np.float64)
            jstop = min(max(int(setup_spa[4]), 1), tdi.time.size - 1)
            while jstop + 1 < tdi.time.size and float(track[jstop]) < spa_stop_frequency:
                jstop += 1
            setup_spa[4] = float(jstop)
            freq, phase, amplitude = ftran_spa_only(
                setup_spa,
                tdi.time,
                tdi.signed_amplitude[channel, int(entry["carrier"])],
                tdi.channel_phase[channel, int(entry["carrier"])],
                shape,
            )
            freq, phase, amplitude = prune_nonincreasing_frequency_samples(freq, phase, amplitude)
            frequency_domain_time += time.perf_counter() - fd_start
            spa_mid, spa_size = wdm_pixels_range(
                tdi.time, track, float(tdi.time[0]), float(tdi.time[jstop]), shape
            )
            merge_pixel_plans(spa_union_mid, spa_union_size, spa_mid, spa_size, shape)
            local_mid, local_size = wdm_pixels_range(
                tdi.time, track, direct_start, direct_stop, shape
            )
            merge_pixel_plans(endpoint_mid, endpoint_size, local_mid, local_size, shape)

            packet_start = time.perf_counter()
            component = wdm_track(freq, phase, amplitude, spa_mid, spa_size, shape)
            # Each carrier packet is already sparse.  Expanding it to a full
            # 4096x4097 matrix here allocated and cleared about 134 MB per
            # carrier solely to perform this addition.
            dense[component.listn, component.listm] += component.values
            packet_time += time.perf_counter() - packet_start
            spectra.append(
                THMCarrierFourier(
                    mode=entry["mode"],
                    frequency=freq,
                    phase=phase,
                    amplitude=amplitude,
                    setup=setup_spa,
                    join_frequency=float(entry["join"]),
                )
            )

        peak_index = int(np.argmax(np.abs(tdi.channel_strain[channel])))
        wdm_pixels_add_merger_frequency_tail(
            endpoint_mid,
            endpoint_size,
            f_replace_stop,
            float(tdi.time[peak_index]),
            shape,
        )
        total_mid = spa_union_mid.copy()
        total_size = spa_union_size.copy()
        merge_pixel_plans(total_mid, total_size, endpoint_mid, endpoint_size, shape)
        packet_start = time.perf_counter()
        apply_short_fft_threshold(
            dense,
            endpoint_mid,
            endpoint_size,
            endpoint_tdi.channel_strain[channel] * taper,
            driver_setup,
            replace_start,
            f_replace_stop,
            shape,
            blend_half_width=blend_width,
        )
        packet_time += time.perf_counter() - packet_start
        all_frequency = np.concatenate([item.frequency for item in spectra])
        all_phase = np.concatenate([item.phase for item in spectra])
        all_amplitude = np.concatenate([item.amplitude for item in spectra])
        name = CHANNELS[channel]
        channels_out[name] = wdm_channel_from_dense(
            all_frequency,
            all_phase,
            all_amplitude,
            total_mid,
            total_size,
            dense,
            shape,
        )
        carrier_fourier[name] = tuple(spectra)

    timings["fast_frequency_domain"] = frequency_domain_time
    timings["fast_wdm_packets"] = packet_time
    timings["fast_wdm"] = time.perf_counter() - start_wdm
    return THMTDIWDMResult(
        source, shape, intrinsic, tdi, channels_out, carrier_fourier, timings
    )


def _write_carrier_frequency(prefix: str, channel: str, carrier: THMCarrierFourier) -> str:
    code = f"{carrier.mode[0]}{carrier.mode[1]}"
    path = f"{prefix}_{channel}_{code}_freq_ap.dat"
    real = carrier.amplitude * np.cos(carrier.phase)
    imag = carrier.amplitude * np.sin(carrier.phase)
    np.savetxt(
        path,
        np.column_stack((carrier.frequency, carrier.amplitude, carrier.phase, real, imag)),
        fmt="%.15e",
        header="f amplitude phase real imag",
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", default="default", help="Folded carriers: default, 22pair, or comma-separated 22,21,33,44,55.")
    parser.add_argument("--channel", action="append", choices=CHANNELS, help="Repeat to select channels; defaults to X,Y,Z.")
    parser.add_argument("--m1-solar", type=float, default=2.0e5)
    parser.add_argument("--m2-solar", type=float, default=1.0e5)
    parser.add_argument("--chi1", type=float, default=0.42)
    parser.add_argument("--chi2", type=float, default=0.85)
    parser.add_argument("--phic", type=float, default=0.0)
    parser.add_argument("--tc", type=float, default=3.0e7)
    parser.add_argument("--distance-gpc", type=float, default=1.0)
    parser.add_argument("--theta", type=float, default=2.31, help="Ecliptic colatitude in radians.")
    parser.add_argument("--lambda", dest="longitude", type=float, default=0.57, help="Ecliptic longitude in radians.")
    parser.add_argument("--psi", type=float, default=0.4, help="Polarization angle in radians.")
    parser.add_argument("--cosi", type=float, default=0.3, help="Cosine of inclination.")
    parser.add_argument("--nf", type=int, default=4096)
    parser.add_argument("--nt", type=int, default=4096)
    parser.add_argument("--dt", type=float, default=1.875)
    parser.add_argument("--tdi-generation", type=int, choices=(1, 2), default=1)
    parser.add_argument("--nsmax", type=int, default=10000)
    parser.add_argument("--coefficient-backend", choices=("auto", "native", "python", "reference"), default="auto")
    parser.add_argument(
        "--wdm-method",
        choices=("partitioned-fft", "spa"),
        default="partitioned-fft",
        help="Sparse WDM engine; partitioned-fft is the production default.",
    )
    parser.add_argument("--partition-bandwidth-hz", type=float, default=PARTITION_FFT_BANDWIDTH_HZ)
    parser.add_argument("--partition-roll-seconds", type=float, default=PARTITION_FFT_ROLL_SECONDS)
    parser.add_argument("--wdm-blend-endpoint", action=argparse.BooleanOptionalAction, default=DEFAULT_WDM_BLEND_ENDPOINT)
    parser.add_argument("--wdm-blend-half-width-layers", type=float, default=DEFAULT_WDM_BLEND_HALF_WIDTH_LAYERS)
    parser.add_argument("--frequency-domain-only", action="store_true", help="Stop after per-carrier SPA spectra.")
    parser.add_argument("--write-prefix", default="phenomthm_tdi")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--sparse-only", action="store_true")
    parser.add_argument("--write-frequency-domain", action="store_true")
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--timing-warmup-runs", type=int, default=1)
    parser.add_argument("--timing-repeat-runs", type=int, default=1)
    args = parser.parse_args()
    if args.timing_warmup_runs < 0 or args.timing_repeat_runs < 1:
        raise SystemExit("timing warmup must be non-negative and repeat count at least one")

    source = SourceParams(
        m1_solar=args.m1_solar,
        m2_solar=args.m2_solar,
        chi1=args.chi1,
        chi2=args.chi2,
        phic=args.phic,
        tc=args.tc,
        distance_gpc=args.distance_gpc,
        ecliptic_colatitude=args.theta,
        ecliptic_longitude=args.longitude,
        polarization=args.psi,
        cos_inclination=args.cosi,
    )
    shape = WDMShape(nf=args.nf, nt=args.nt, dt=args.dt)
    channels = tuple(args.channel) if args.channel else CHANNELS

    def run_once() -> THMTDIWDMResult:
        return generate_thm_tdi_wdm(
            source,
            shape,
            channels,
            modes=args.modes,
            nsmax=args.nsmax,
            coefficient_backend=args.coefficient_backend,
            compute_wdm=not args.frequency_domain_only,
            wdm_method=args.wdm_method,
            partition_bandwidth_hz=args.partition_bandwidth_hz,
            partition_roll_seconds=args.partition_roll_seconds,
            blend_endpoint=args.wdm_blend_endpoint,
            blend_half_width_layers=args.wdm_blend_half_width_layers,
            tdi_generation=args.tdi_generation,
        )

    for _ in range(args.timing_warmup_runs if args.timing else 0):
        run_once()
    repeats = args.timing_repeat_runs if args.timing else 1
    results: list[THMTDIWDMResult] = []
    elapsed: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        results.append(run_once())
        elapsed.append(time.perf_counter() - start)
    result = results[-1]

    print(f"modes {' '.join(f'{ell}{emm}' for ell, emm in result.intrinsic.modes)}")
    print(
        f"intrinsic_samples {result.intrinsic.waveform_time.size} "
        f"response_samples {result.tdi.time.size} "
        f"exact_response_samples {result.intrinsic.exact_response_samples}"
    )
    print(f"wdm_method {args.wdm_method}")
    if args.wdm_method == "partitioned-fft" and not args.frequency_domain_only:
        print(
            f"partition_blocks {int(result.timings['partition_blocks'])} "
            f"early_fft_samples {int(result.timings['partition_early_samples'])} "
            f"endpoint_fft_samples {int(result.timings['partition_endpoint_samples'])}"
        )
    print(f"wdm_grid nt {shape.nt} nf {shape.nf} dt {shape.dt:.15e} Tobs {shape.Tobs:.15e}")
    written: list[str] = []
    for channel in channels:
        output = result.channels[channel]
        if args.frequency_domain_only:
            print(f"channel {channel} active_pixels skipped")
        else:
            print(f"channel {channel} active_pixels {output.values.size}")
        if args.no_write:
            continue
        if args.write_frequency_domain or args.frequency_domain_only:
            for carrier in result.carrier_fourier.get(channel, ()):
                written.append(_write_carrier_frequency(args.write_prefix, channel, carrier))
        if not args.frequency_domain_only:
            track_path = f"{args.write_prefix}_{channel}_track_pixels.dat"
            write_track_pixels(track_path, output)
            written.append(track_path)
            if not args.sparse_only:
                dense_path = f"{args.write_prefix}_{channel}_wtranfast.dat"
                write_wdm_matrix(dense_path, output.dense(shape))
                written.append(dense_path)
    print("output_files " + ("none" if not written else " ".join(written)))
    if args.timing:
        keys = ["adaptive_ap", "fast_tdi"]
        if args.wdm_method == "partitioned-fft" and not args.frequency_domain_only:
            keys.extend(
                (
                    "partition_plan",
                    "partition_early_fft",
                    "partition_endpoint_fft",
                    "partition_packet_wdm",
                )
            )
        keys.extend(("fast_frequency_domain", "fast_wdm_packets", "fast_wdm"))
        print("timing_seconds")
        print(f"warmup_runs {args.timing_warmup_runs}")
        print(f"measured_runs {repeats}")
        for key in keys:
            values = [item.timings[key] for item in results]
            print(f"{key} {np.mean(values):.6f}")
    print(f"elapsed_seconds {np.mean(elapsed):.6f}")


if __name__ == "__main__":
    main()


__all__ = [
    "CHANNELS",
    "MODE_CODES",
    "PARTITION_FFT_BANDWIDTH_HZ",
    "PARTITION_FFT_ROLL_SECONDS",
    "THMCarrierFourier",
    "THMIntrinsicGrid",
    "THMTDIGrid",
    "THMTDIWDMResult",
    "THMPartitionBlockDiagnostic",
    "build_thm_intrinsic_grid",
    "compute_thm_tdi_grid",
    "folded_projection",
    "generate_thm_tdi_wdm",
]
