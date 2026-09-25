#!/usr/bin/env python3
# Python LISA response and WDM port: Copyright (C) 2026 Neil Cornish.
# SPDX-License-Identifier: GPL-3.0-or-later
# Distributed without warranty; see the GNU GPL for details.
# The imported LAL-derived waveform model retains its upstream notices.
"""LAL-free numerical IMRPhenomTPHM, sparse LISA TDI, and fast WDM.

The rapidly varying co-precessing phase of each folded THM parent carrier is
kept explicit.  Precession and the sky projection are represented by slowly
varying complex plus/cross envelopes, splined in Cartesian form.  Unequal-arm
TDI is then evaluated at every delayed source time before complementary,
heterodyned FFT blocks are sent to the sparse Meyer-packet engine.

This is deliberately a partitioned-FFT implementation.  A numerical SPA of a
folded precessing carrier would have to resolve all of its precession sidebands
as separate stationary points, defeating the compact five-carrier interface.
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

from phenomt_tdi_wdm import (
    AP_DTM_MAX,
    CONSTELLATION_LIGHT_TIME_SECONDS,
    DTMIN,
    INTRINSIC_TDI_SWITCH_GUARD_SECONDS,
    NOMINAL_ARM_LIGHT_TIME_SECONDS,
    RESPONSE_LATE_MARGIN_SECONDS,
    SourceParams,
    WDMChannel,
    WDMShape,
    _build_intrinsic_model_grid,
    _detector_adaptive_step,
    _detector_step_proposal,
    _intrinsic_tdi_switch_source_time,
    _stack_spline_coefficients,
    _unique_sorted_times,
    barycenter_time,
    build_constellation_splines,
    make_ap_spline,
    sky_vectors,
)
from phenomthm_tdi_wdm import (
    CHANNELS,
    MODE_BY_CODE,
    PARTITION_FFT_BANDWIDTH_HZ,
    PARTITION_FFT_MARGIN_CYCLES,
    PARTITION_FFT_NYQUIST_GUARD_LAYERS,
    PARTITION_FFT_ROLL_SECONDS,
    THMPartitionBlockDiagnostic,
    THMTDIGrid,
    TRIPLES,
    _CachedPlanning22,
    _ScaledPlanningModel,
    _THMPartitionSpectrum,
    _centered_even_heterodyne_layer,
    _empty_channel,
    _merge_sparse_channels,
    _next_power_of_two,
    _packet_plan_from_bounds,
    _partition_blocks_to_wdm,
    _response_geometry,
    _smooth_step,
    _tdi_delays_and_coefficients,
    _track_range_linear,
)
from phenomtphm import IMRPhenomTPHM
from phenomt22 import TSUN
from tdi2_response import complex_tdi2


PI = math.pi
SECONDS_PER_YEAR = 365.25 * 86400.0
TPHM_ENVELOPE_FAR_DT = 2.0e4


@dataclass(frozen=True)
class TPHMSourceParams(SourceParams):
    """Masses, full spin vectors, and extrinsic TPHM parameters.

    The inherited ``chi1`` and ``chi2`` fields are the components parallel to
    the reference orbital angular momentum.  ``tau_ref=None`` places the spin
    reference at the first source time required by the response.
    """

    chi1x: float = 0.35
    chi1y: float = 0.10
    chi2x: float = -0.20
    chi2y: float = 0.25
    tau_ref: float | None = None

    def masses_and_spins_seconds(
        self,
    ) -> tuple[float, float, tuple[float, float, float], tuple[float, float, float]]:
        m1 = self.m1_solar * TSUN
        m2 = self.m2_solar * TSUN
        spin1 = (self.chi1x, self.chi1y, self.chi1)
        spin2 = (self.chi2x, self.chi2y, self.chi2)
        if m2 > m1:
            m1, m2 = m2, m1
            spin1, spin2 = spin2, spin1
        return m1, m2, spin1, spin2


@dataclass(frozen=True)
class TPHMIntrinsicGrid:
    """Carrier phase tracks and slow precession-polarization envelopes."""

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
    plus_envelope: np.ndarray
    cross_envelope: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    gamma: np.ndarray
    model: IMRPhenomTPHM
    response_switch_detector_time: np.ndarray
    exact_response_samples: int


@dataclass(frozen=True)
class TPHMPlanningBands:
    time: np.ndarray
    center: np.ndarray
    low: np.ndarray
    high: np.ndarray


@dataclass(frozen=True)
class TPHMTDIWDMResult:
    source: TPHMSourceParams
    shape: WDMShape
    intrinsic: TPHMIntrinsicGrid
    tdi: THMTDIGrid
    planning_bands: TPHMPlanningBands
    channels: dict[str, WDMChannel]
    block_diagnostics: tuple[THMPartitionBlockDiagnostic, ...]
    timings: dict[str, float]


def _normalize_modes(
    modes: Sequence[int | tuple[int, int]] | str,
) -> tuple[tuple[int, int], ...]:
    if isinstance(modes, str):
        name = modes.strip().lower()
        if name in {"default", "all", "hm"}:
            return tuple(MODE_BY_CODE.values())
        if name in {"22", "22pair", "2|2|"}:
            return ((2, 2),)
        codes = tuple(int(token) for token in name.split(",") if token.strip())
    else:
        codes = tuple(
            10 * value[0] + abs(value[1]) if isinstance(value, tuple) else int(value)
            for value in modes
        )
    selected: list[tuple[int, int]] = []
    for code in codes:
        if code not in MODE_BY_CODE:
            raise ValueError(f"unsupported folded TPHM parent carrier {code}")
        if MODE_BY_CODE[code] not in selected:
            selected.append(MODE_BY_CODE[code])
    if not selected:
        raise ValueError("at least one TPHM parent carrier is required")
    return tuple(selected)


def _piecewise_envelope_times(lower: float, upper: float, tc: float, mass: float) -> np.ndarray:
    """C-compatible sparse grid for the slow twist-up envelopes."""

    far_dt = TPHM_ENVELOPE_FAR_DT
    near_dt = min(far_dt, min(1000.0, max(50.0, 32.0 * mass)))
    plunge_dt = min(near_dt, min(50.0, max(0.5, 2.0 * mass)))
    near_start = tc - 1.0e5
    plunge_start = max(near_start, tc - 2000.0 * mass)
    values: list[float] = []
    current = lower
    while current < upper:
        values.append(current)
        if current < near_start:
            step = far_dt
            boundary = near_start
        elif current < plunge_start:
            step = near_dt
            boundary = plunge_start
        else:
            step = plunge_dt
            boundary = upper
        following = current + step
        if current < boundary < following:
            following = boundary
        if following <= current:
            following = current + step
        current = min(following, upper)
    values.append(upper)
    return np.asarray(values, dtype=np.float64)


def build_tphm_intrinsic_grid(
    source: TPHMSourceParams = TPHMSourceParams(),
    shape: WDMShape = WDMShape(),
    *,
    modes: Sequence[int | tuple[int, int]] | str = "default",
    nsmax: int = 10000,
    coefficient_backend: str = "auto",
) -> tuple[TPHMIntrinsicGrid, tuple[np.ndarray, list, list, list]]:
    """Build one carrier grid and one shared numerical-precession trajectory."""

    selected_modes = _normalize_modes(modes)
    constellation = build_constellation_splines(shape)
    constellation_time, _l_splines, p_splines, _v_splines = constellation
    m1, m2, spin1, spin2 = source.masses_and_spins_seconds()
    total_mass = m1 + m2
    tstart = max(0.0, float(constellation_time[0]) + CONSTELLATION_LIGHT_TIME_SECONDS)
    tstop = min(
        shape.Tobs,
        source.tc + RESPONSE_LATE_MARGIN_SECONDS + 1000.0 * total_mass,
        float(constellation_time[-1]) - CONSTELLATION_LIGHT_TIME_SECONDS,
    )
    if tstop <= tstart:
        raise ValueError("the requested merger and observation do not overlap")
    source_bounds = barycenter_time(
        np.asarray((tstart, tstop), dtype=np.float64), source, p_splines
    )
    delay_margin = CONSTELLATION_LIGHT_TIME_SECONDS + 4.0 * NOMINAL_ARM_LIGHT_TIME_SECONDS
    source_lower = float(source_bounds[0]) - delay_margin
    source_upper = float(source_bounds[1]) + delay_margin
    tau_ref = (
        float(source.tau_ref)
        if source.tau_ref is not None
        else (source_lower - source.tc) / total_mass
    )
    model = IMRPhenomTPHM(
        m1,
        m2,
        spin1,
        spin2,
        tau_ref=tau_ref,
        modes=selected_modes,
        coefficient_backend=coefficient_backend,
    )

    cached_22 = _CachedPlanning22(model.carrier.mode22)
    planning_models = tuple(
        _ScaledPlanningModel(cached_22, mode[1]) for mode in selected_modes
    )
    driver = max(planning_models, key=lambda item: item.scale)
    model_time = _build_intrinsic_model_grid(
        driver,
        source_lower,
        source_upper,
        source.tc,
        total_mass,
        2 * nsmax + 4,
    )
    switches: list[float] = []
    for planning in planning_models:
        source_switch = _intrinsic_tdi_switch_source_time(
            planning,
            model_time,
            source_lower,
            source_upper,
            source.tc,
            total_mass,
        )
        switches.append(
            min(tstop, max(tstart, source_switch - INTRINSIC_TDI_SWITCH_GUARD_SECONDS))
        )

    _, _, propagation = sky_vectors(source)
    const_x = np.ascontiguousarray(p_splines[0].x, dtype=np.float64)
    p_coeffs = _stack_spline_coefficients(p_splines)
    propagation = np.ascontiguousarray(propagation, dtype=np.float64)
    detector_points: list[float] = []
    output_time = tstart
    last_step = AP_DTM_MAX
    previous_step = AP_DTM_MAX
    history_count = 0
    while True:
        if len(detector_points) >= nsmax:
            raise RuntimeError("shared TPHM detector-response grid exceeded nsmax")
        detector_points.append(output_time)
        if output_time >= tstop:
            break
        proposal = _detector_step_proposal(last_step, previous_step, history_count)
        candidates: list[float] = []
        for planning, switch in zip(planning_models, switches):
            if output_time < switch:
                candidates.append(min(AP_DTM_MAX, switch - output_time))
            else:
                candidates.append(
                    _detector_adaptive_step(
                        planning,
                        output_time,
                        tstop,
                        proposal,
                        source.tc,
                        total_mass,
                        abs(planning.omega_ring) / (2.0 * PI * total_mass),
                        const_x,
                        p_coeffs,
                        propagation,
                    )
                )
        step = min(candidates)
        following = min(output_time + max(step, DTMIN), tstop)
        if following <= output_time:
            following = min(output_time + DTMIN, tstop)
        if output_time >= min(switches):
            previous_step = last_step
            last_step = following - output_time
            history_count += 1
        output_time = following

    center_source_time = np.asarray(detector_points, dtype=np.float64)
    ssb_output_time = barycenter_time(center_source_time, source, p_splines)
    exact = np.zeros(center_source_time.size, dtype=bool)
    exact[0] = True
    exact[-1] = True
    spacing = np.diff(center_source_time)
    epsilon = 1.0e-10 * AP_DTM_MAX
    exact[:-1] |= spacing < AP_DTM_MAX - epsilon
    exact[1:] |= spacing < AP_DTM_MAX - epsilon

    envelope_time = _piecewise_envelope_times(
        source_lower, source_upper, source.tc, total_mass
    )
    waveform_time = _unique_sorted_times(
        np.concatenate((model_time, ssb_output_time[exact], envelope_time))
    )
    waveform = model.evaluate_times(
        waveform_time,
        tc=source.tc,
        phic=source.phic,
        observer_theta=math.acos(max(-1.0, min(1.0, source.cos_inclination))),
        observer_phi=0.5 * PI,
        polarization=source.polarization,
        distance_gpc=source.distance_gpc,
    )

    model_amplitude = np.stack(
        [waveform.carriers[mode].amplitude for mode in selected_modes]
    )
    model_phase = np.stack(
        [waveform.carriers[mode].phase for mode in selected_modes]
    )
    model_frequency = np.stack(
        [np.abs(waveform.carriers[mode].omega) / (2.0 * PI) for mode in selected_modes]
    )
    plus_envelope = []
    cross_envelope = []
    for mode in selected_modes:
        carrier = waveform.carriers[mode]
        analytic_plus = carrier.strain.real - 1j * carrier.quadrature.real
        analytic_cross = -carrier.strain.imag + 1j * carrier.quadrature.imag
        carrier_conjugate = np.exp(-1j * carrier.phase)
        plus_envelope.append(analytic_plus * carrier_conjugate)
        cross_envelope.append(analytic_cross * carrier_conjugate)
    plus_envelope_array = np.asarray(plus_envelope, dtype=np.complex128)
    cross_envelope_array = np.asarray(cross_envelope, dtype=np.complex128)

    amplitude = np.stack(
        [make_ap_spline(waveform_time, values)(ssb_output_time) for values in model_amplitude]
    )
    phase = np.stack(
        [make_ap_spline(waveform_time, values)(ssb_output_time) for values in model_phase]
    )
    frequency = np.stack(
        [make_ap_spline(waveform_time, values)(ssb_output_time) for values in model_frequency]
    )

    # Preserve a strict interpolation guard at both source-time ends.  Delayed
    # source evaluations themselves use the wider waveform_time interval.
    keep = (ssb_output_time >= waveform_time[0]) & (ssb_output_time <= waveform_time[-1])
    indices = np.flatnonzero(keep)
    if indices.size < 3:
        raise RuntimeError("too few TPHM response samples survive endpoint guards")
    last = int(indices[-1]) + 1
    intrinsic = TPHMIntrinsicGrid(
        waveform_time=np.ascontiguousarray(waveform_time),
        spline_time=np.ascontiguousarray(ssb_output_time),
        response_time=np.ascontiguousarray(ssb_output_time[:last]),
        detector_time=np.ascontiguousarray(center_source_time[:last]),
        modes=selected_modes,
        amplitude=np.ascontiguousarray(amplitude),
        phase=np.ascontiguousarray(phase),
        frequency=np.ascontiguousarray(frequency),
        model_amplitude=np.ascontiguousarray(model_amplitude),
        model_phase=np.ascontiguousarray(model_phase),
        model_frequency=np.ascontiguousarray(model_frequency),
        plus_envelope=np.ascontiguousarray(plus_envelope_array),
        cross_envelope=np.ascontiguousarray(cross_envelope_array),
        alpha=np.ascontiguousarray(waveform.alpha),
        beta=np.ascontiguousarray(waveform.beta),
        gamma=np.ascontiguousarray(waveform.gamma),
        model=model,
        response_switch_detector_time=np.asarray(switches),
        exact_response_samples=int(np.count_nonzero(exact)),
    )
    return intrinsic, constellation


def _complex_spline_pair(times: np.ndarray, values: np.ndarray) -> tuple[object, object]:
    return make_ap_spline(times, values.real), make_ap_spline(times, values.imag)


def _source_envelope_splines(intrinsic: TPHMIntrinsicGrid):
    phase = [make_ap_spline(intrinsic.waveform_time, row) for row in intrinsic.model_phase]
    plus = [_complex_spline_pair(intrinsic.waveform_time, row) for row in intrinsic.plus_envelope]
    cross = [_complex_spline_pair(intrinsic.waveform_time, row) for row in intrinsic.cross_envelope]
    return phase, plus, cross


def _response_from_tphm_envelopes(
    times: np.ndarray,
    source: TPHMSourceParams,
    intrinsic: TPHMIntrinsicGrid,
    constellation,
    *,
    envelope_splines=None,
    tdi_generation: int = 1,
) -> THMTDIGrid:
    """Apply unequal-arm TDI to every reconstructed carrier analytic signal."""

    times = np.ascontiguousarray(np.asarray(times, dtype=np.float64))
    geometry = _response_geometry(times, source, constellation)
    arm_length, kr, app, apm, acp, acm = geometry
    phase_splines, plus_splines, cross_splines = (
        _source_envelope_splines(intrinsic)
        if envelope_splines is None
        else envelope_splines
    )
    ncarrier = len(intrinsic.modes)
    raw_m = np.empty((3, ncarrier, times.size), dtype=np.float64)
    raw_mf = np.empty_like(raw_m)
    if tdi_generation == 2:
        def polarizations(source_time: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            live = ((source_time >= intrinsic.waveform_time[0]) &
                    (source_time <= intrinsic.waveform_time[-1]))
            plus = np.zeros((ncarrier, len(source_time)), dtype=np.complex128)
            cross = np.zeros_like(plus)
            if np.any(live):
                local = source_time[live]
                for carrier in range(ncarrier):
                    phasor = np.exp(1j*phase_splines[carrier](local))
                    plus_re, plus_im = plus_splines[carrier]
                    cross_re, cross_im = cross_splines[carrier]
                    plus[carrier, live] = (plus_re(local)+1j*plus_im(local))*phasor
                    cross[carrier, live] = (cross_re(local)+1j*cross_im(local))*phasor
            return plus, cross

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
                carrier_phase = np.asarray(phase_splines[carrier](delays), dtype=np.float64)
                phasor = np.exp(1j * carrier_phase)
                plus_re, plus_im = plus_splines[carrier]
                cross_re, cross_im = cross_splines[carrier]
                hplus = (
                    np.asarray(plus_re(delays), dtype=np.float64)
                    + 1j * np.asarray(plus_im(delays), dtype=np.float64)
                ) * phasor
                hcross = (
                    np.asarray(cross_re(delays), dtype=np.float64)
                    + 1j * np.asarray(cross_im(delays), dtype=np.float64)
                ) * phasor
                analytic = np.sum(hplus * coefp + hcross * coefc, axis=1)
                raw_m[channel, carrier] = analytic.real
                raw_mf[channel, carrier] = -analytic.imag

    reference = np.stack(
        [np.asarray(spline(times), dtype=np.float64) for spline in phase_splines]
    )
    amplitude_splines = [
        make_ap_spline(intrinsic.waveform_time, row) for row in intrinsic.model_amplitude
    ]
    frequency_splines = [
        make_ap_spline(intrinsic.waveform_time, row) for row in intrinsic.model_frequency
    ]
    amplitude = np.stack([spline(times) for spline in amplitude_splines])
    frequency = np.stack([spline(times) for spline in frequency_splines])
    nan_values = np.full_like(raw_m, np.nan)
    return THMTDIGrid(
        time=times,
        detector_time=times.copy(),
        modes=intrinsic.modes,
        intrinsic_amplitude=np.ascontiguousarray(amplitude),
        intrinsic_phase=np.ascontiguousarray(reference),
        intrinsic_frequency=np.ascontiguousarray(frequency),
        raw_m=raw_m,
        raw_mf=raw_mf,
        signed_amplitude=nan_values.copy(),
        channel_phase=nan_values.copy(),
        reference_phase=np.ascontiguousarray(reference),
        channel_strain=np.sum(raw_m, axis=1),
    )


def compute_tphm_tdi_grid(
    source: TPHMSourceParams,
    intrinsic: TPHMIntrinsicGrid,
    constellation,
    *,
    tdi_generation: int = 1,
) -> THMTDIGrid:
    return _response_from_tphm_envelopes(
        intrinsic.response_time, source, intrinsic, constellation,
        tdi_generation=tdi_generation,
    )


def build_tphm_planning_bands(
    intrinsic: TPHMIntrinsicGrid,
    tdi: THMTDIGrid,
    shape: WDMShape,
) -> TPHMPlanningBands:
    """Bound each folded carrier's precession sideband family in frequency."""

    start = max(0.0, math.floor(float(tdi.time[0]) / shape.DT) * shape.DT)
    stop = min(shape.Tobs, float(tdi.time[-1]))
    count = max(2, int(math.ceil((stop - start) / shape.DT)) + 1)
    track_time = start + shape.DT * np.arange(count, dtype=np.float64)
    track_time[-1] = min(track_time[-1], stop)
    alpha_spline = make_ap_spline(intrinsic.waveform_time, intrinsic.alpha)
    gamma_spline = make_ap_spline(intrinsic.waveform_time, intrinsic.gamma)
    alpha = np.asarray(alpha_spline(track_time), dtype=np.float64)
    gamma = np.asarray(gamma_spline(track_time), dtype=np.float64)
    edge_order = 2 if track_time.size > 2 else 1
    alpha_dot = np.gradient(alpha, track_time, edge_order=edge_order)
    gamma_dot = np.gradient(gamma, track_time, edge_order=edge_order)
    center = np.stack(
        [
            np.asarray(
                make_ap_spline(intrinsic.waveform_time, row)(track_time),
                dtype=np.float64,
            )
            for row in intrinsic.model_frequency
        ]
    )
    low = np.empty_like(center)
    high = np.empty_like(center)
    for carrier, (ell, abs_m) in enumerate(intrinsic.modes):
        sideband = (
            float(ell) * np.abs(alpha_dot) + float(abs_m) * np.abs(gamma_dot)
        ) / (2.0 * PI)
        margin = 3.0e-4 * center[carrier] + 4.0 / SECONDS_PER_YEAR
        low[carrier] = np.maximum(0.0, center[carrier] - sideband - margin)
        high[carrier] = center[carrier] + sideband + margin
    return TPHMPlanningBands(track_time, center, low, high)


@njit(cache=True)
def _compact_bounds_kernel(
    times: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    center: np.ndarray,
    endpoint_start: float,
    early_stop: float,
    waveform_stop: float,
    nt: int,
    nf: int,
    dt: float,
    df: float,
    fb: float,
    first_layer: int,
    last_layer: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    earliest = np.full(nf, nt + 1, dtype=np.int64)
    latest = np.full(nf, -1, dtype=np.int64)
    endpoint_center_low = math.inf
    endpoint_high = -math.inf
    for carrier in range(center.shape[0]):
        for index in range(times.size - 1):
            segment_start = times[index]
            segment_stop = times[index + 1]
            flo = min(low[carrier, index], low[carrier, index + 1])
            fhi = max(high[carrier, index], high[carrier, index + 1])
            if segment_stop >= endpoint_start and segment_start <= waveform_stop and fhi > 0.0:
                center_lo = min(center[carrier, index], center[carrier, index + 1])
                if center_lo > 0.0:
                    endpoint_center_low = min(endpoint_center_low, center_lo)
                endpoint_high = max(endpoint_high, fhi)
            if segment_stop < times[0] or segment_start > early_stop or fhi <= 0.0:
                continue
            lower_time = max(segment_start, times[0])
            upper_time = min(segment_stop, early_stop)
            if upper_time < lower_time:
                continue
            layer_lo = max(first_layer, int(math.ceil((flo - fb) / df)))
            layer_hi = min(last_layer, int(math.floor((fhi + fb) / df)))
            if layer_hi < layer_lo:
                continue
            pixel_lo = max(0, int(math.floor(lower_time / dt)))
            pixel_hi = min(nt - 1, int(math.ceil(upper_time / dt)))
            for layer in range(layer_lo, layer_hi + 1):
                earliest[layer] = min(earliest[layer], pixel_lo)
                latest[layer] = max(latest[layer], pixel_hi)
    return earliest, latest, endpoint_center_low, endpoint_high


def _compact_packet_plans(
    bands: TPHMPlanningBands,
    endpoint_start: float,
    endpoint_rise: float,
    waveform_stop: float,
    shape: WDMShape,
    *,
    first_layer: int = 1,
    last_layer: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Port of the compact physical-band support used by the C TPHM path.

    Optional inclusive layer limits restrict the track search and packet
    construction, not just the output map. Defaults preserve the LISA path.
    """

    first_layer = max(1, first_layer)
    last_layer = min(shape.nf - 1, shape.nf - 1 if last_layer is None else last_layer)
    early_stop = min(waveform_stop, endpoint_start + endpoint_rise)
    earliest, latest, endpoint_center_low, endpoint_high = _compact_bounds_kernel(
        bands.time,
        bands.low,
        bands.high,
        bands.center,
        endpoint_start,
        early_stop,
        waveform_stop,
        shape.nt,
        shape.nf,
        shape.DT,
        shape.DF,
        shape.FB,
        first_layer,
        last_layer,
    )

    if not math.isfinite(endpoint_center_low) or not math.isfinite(endpoint_high):
        endpoint_center_low = max(shape.DF, float(np.min(bands.center[:, -1])))
        endpoint_high = float(np.max(bands.high[:, -1]))
    endpoint_margin = 3.0 / endpoint_rise + shape.FB
    endpoint_margin = min(endpoint_margin, 0.5 * endpoint_center_low)
    endpoint_frequency_start = max(0.0, endpoint_center_low - endpoint_margin)
    endpoint_frequency_stop = min(
        (shape.nf - 1) * shape.DF + shape.FB,
        endpoint_high + 3.0 / endpoint_rise + shape.FB,
    )

    early_mid = np.full(shape.nf, -1, dtype=np.int64)
    early_size = np.zeros(shape.nf, dtype=np.int64)
    for layer in np.flatnonzero(latest >= earliest):
        early_mid[layer], early_size[layer] = _packet_plan_from_bounds(
            int(earliest[layer]), int(latest[layer]), shape
        )
    endpoint_mid = np.full(shape.nf, -1, dtype=np.int64)
    endpoint_size = np.zeros(shape.nf, dtype=np.int64)
    layer_lo = max(
        first_layer, int(math.ceil((endpoint_frequency_start - shape.FB) / shape.DF))
    )
    layer_hi = min(
        last_layer,
        int(math.floor((endpoint_frequency_stop + shape.FB) / shape.DF)),
    )
    if layer_hi >= layer_lo:
        center, size = _packet_plan_from_bounds(
            int(math.floor(endpoint_start / shape.DT)),
            int(math.ceil(waveform_stop / shape.DT)),
            shape,
        )
        endpoint_mid[layer_lo : layer_hi + 1] = center
        endpoint_size[layer_lo : layer_hi + 1] = size
    return (
        early_mid,
        early_size,
        endpoint_mid,
        endpoint_size,
        endpoint_frequency_start,
        endpoint_frequency_stop,
    )


def _response_envelope_splines(
    tdi: THMTDIGrid,
    intrinsic: TPHMIntrinsicGrid,
    channel_indices: Sequence[int],
):
    real_splines: list[object] = []
    imag_splines: list[object] = []
    phase_splines: list[object] = []
    for carrier in range(len(intrinsic.modes)):
        phase_spline = make_ap_spline(intrinsic.waveform_time, intrinsic.model_phase[carrier])
        phase_splines.append(phase_spline)
        phase = np.asarray(phase_spline(tdi.time), dtype=np.float64)
        for channel in channel_indices:
            analytic = tdi.raw_m[channel, carrier] - 1j * tdi.raw_mf[channel, carrier]
            envelope = analytic * np.exp(-1j * phase)
            real_splines.append(make_ap_spline(tdi.time, envelope.real))
            imag_splines.append(make_ap_spline(tdi.time, envelope.imag))
    return real_splines, imag_splines, phase_splines


def _build_early_blocks(
    tdi: THMTDIGrid,
    intrinsic: TPHMIntrinsicGrid,
    bands: TPHMPlanningBands,
    channel_indices: Sequence[int],
    early_mid: np.ndarray,
    early_size: np.ndarray,
    endpoint_start: float,
    endpoint_rise: float,
    shape: WDMShape,
    *,
    bandwidth_hz: float,
    roll_seconds: float,
    envelope_splines=None,
) -> tuple[list[_THMPartitionSpectrum], list[THMPartitionBlockDiagnostic], int]:
    frequency_band = shape.FB
    frequency_step = shape.DF
    time_pixel = shape.DT
    if bandwidth_hz <= 2.0 * frequency_band:
        raise ValueError("partition bandwidth must exceed the Meyer support width")
    times = np.asarray(bands.time, dtype=np.float64)
    time_start = max(0.0, float(times[0]))
    plan_stop = min(endpoint_start, float(times[-1]))
    endpoint_support_stop = min(endpoint_start + endpoint_rise, float(times[-1]))
    start_pixel = max(0, int(math.floor(time_start / time_pixel)))
    endpoint_pixel = min(shape.nt, int(math.floor(plan_stop / time_pixel)))
    roll_pixels = max(1, int(math.ceil(roll_seconds / time_pixel)))
    real_splines, imag_splines, phase_splines = (
        _response_envelope_splines(tdi, intrinsic, channel_indices)
        if envelope_splines is None
        else envelope_splines
    )
    channel_count = len(channel_indices)
    blocks: list[_THMPartitionSpectrum] = []
    diagnostics: list[THMPartitionBlockDiagnostic] = []
    total_samples = 0
    active_layers = np.flatnonzero(early_size > 0)
    active_size = early_size[active_layers]
    packet_start_by_layer = (early_mid[active_layers] - 0.5 * active_size) * time_pixel
    packet_stop_by_layer = packet_start_by_layer + active_size * time_pixel
    layer_lower = active_layers * frequency_step - frequency_band
    layer_upper = active_layers * frequency_step + frequency_band

    for carrier in range(len(intrinsic.modes)):
        low_track = bands.low[carrier]
        high_track = bands.high[carrier]
        tiles: list[tuple[int, int]] = []
        tile_lo = start_pixel
        while tile_lo < endpoint_pixel:
            remaining = endpoint_pixel - tile_lo

            def width_fits(width: int) -> bool:
                support_lo = max(tile_lo - roll_pixels, start_pixel)
                support_hi = min(
                    (tile_lo + width + roll_pixels) * time_pixel,
                    endpoint_support_stop,
                )
                flo = _track_range_linear(
                    times, low_track, support_lo * time_pixel, support_hi
                )[0]
                fhi = _track_range_linear(
                    times, high_track, support_lo * time_pixel, support_hi
                )[1]
                taper_pad = frequency_band + PARTITION_FFT_MARGIN_CYCLES / roll_seconds
                return fhi - flo + 2.0 * taper_pad <= bandwidth_hz

            width = 1 << (remaining.bit_length() - 1)
            while width > 1 and not width_fits(width):
                width //= 2
            tiles.append((tile_lo, min(tile_lo + width, endpoint_pixel)))
            tile_lo = tiles[-1][1]

        boundaries = np.asarray(
            [float(tile_hi) * time_pixel for _, tile_hi in tiles], dtype=np.float64
        )
        boundary_roll = np.full(len(tiles), roll_seconds, dtype=np.float64)
        for index in range(max(len(tiles) - 1, 0)):
            available = boundaries[index + 1] - boundaries[index]
            boundary_roll[index] = max(
                min(roll_seconds, available, plan_stop - boundaries[index]), time_pixel
            )

        for index, (_tile_lo, _tile_hi) in enumerate(tiles):
            nonzero_start = time_start if index == 0 else boundaries[index - 1]
            nonzero_stop = (
                boundaries[index] + boundary_roll[index]
                if index < len(tiles) - 1
                else endpoint_support_stop
            )
            if nonzero_stop <= nonzero_start:
                continue
            rise_start = time_start if index == 0 else boundaries[index - 1]
            rise_stop = rise_start if index == 0 else rise_start + boundary_roll[index - 1]
            fall_start = boundaries[index] if index < len(tiles) - 1 else endpoint_start
            fall_stop = (
                fall_start + boundary_roll[index]
                if index < len(tiles) - 1
                else endpoint_start + endpoint_rise
            )
            fmin = _track_range_linear(times, low_track, nonzero_start, nonzero_stop)[0]
            fmax = _track_range_linear(times, high_track, nonzero_start, nonzero_stop)[1]
            frequency_pad = frequency_band
            for duration in (rise_stop - rise_start, fall_stop - fall_start):
                if duration > 0.0:
                    frequency_pad = max(
                        frequency_pad, frequency_band + PARTITION_FFT_MARGIN_CYCLES / duration
                    )
            layer_lo = max(1, int(math.floor((fmin - frequency_pad) / frequency_step)))
            layer_hi = min(shape.nf - 1, int(math.ceil((fmax + frequency_pad) / frequency_step)))
            heterodyne_layer = _centered_even_heterodyne_layer(layer_lo, layer_hi)
            shifted_extent = max(
                abs(layer_lo - heterodyne_layer), abs(layer_hi - heterodyne_layer), 1
            )
            local_bins = min(
                shape.nf,
                _next_power_of_two(
                    shifted_extent + PARTITION_FFT_NYQUIST_GUARD_LAYERS + 1
                ),
            )
            sample_start = math.floor(nonzero_start / time_pixel) * time_pixel
            support_pixels = max(1, int(math.ceil((nonzero_stop - sample_start) / time_pixel)))
            packet_time_pixels = _next_power_of_two(support_pixels + 2 * shape.mult)
            intersects = (
                (layer_upper >= fmin - frequency_pad)
                & (layer_lower <= fmax + frequency_pad)
                & (packet_stop_by_layer > nonzero_start)
                & (packet_start_by_layer < nonzero_stop)
            )
            if np.any(intersects):
                packet_time_pixels = max(packet_time_pixels, int(np.max(active_size[intersects])))
            packet_time_pixels = min(packet_time_pixels, shape.nt)
            sample_dt = time_pixel / float(local_bins)
            sample_count = packet_time_pixels * local_bins
            sample_times = sample_start + sample_dt * np.arange(sample_count)
            live = (
                (sample_times >= nonzero_start)
                & (sample_times <= nonzero_stop)
                & (sample_times >= tdi.time[0])
                & (sample_times <= tdi.time[-1])
                & (sample_times <= shape.Tobs)
            )
            residual = np.zeros((channel_count, sample_count), dtype=np.complex128)
            if np.any(live):
                live_times = sample_times[live]
                weight = np.ones(live_times.size)
                if rise_stop > rise_start:
                    weight *= _smooth_step(live_times, rise_start, rise_stop)
                if fall_stop > fall_start:
                    weight *= 1.0 - _smooth_step(live_times, fall_start, fall_stop)
                phase = np.asarray(phase_splines[carrier](live_times), dtype=np.float64)
                heterodyne_frequency = heterodyne_layer * frequency_step
                phasor = np.exp(
                    1j
                    * (
                        phase
                        - 2.0 * PI * heterodyne_frequency * (live_times - sample_start)
                    )
                )
                for local_channel in range(channel_count):
                    spline_index = carrier * channel_count + local_channel
                    envelope = np.asarray(real_splines[spline_index](live_times)) + 1j * np.asarray(
                        imag_splines[spline_index](live_times)
                    )
                    residual[local_channel, live] = weight * envelope * phasor
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
                    heterodyne_frequency=heterodyne_layer * frequency_step,
                )
            )
            total_samples += sample_count
    return blocks, diagnostics, total_samples


def _build_endpoint_block(
    source: TPHMSourceParams,
    intrinsic: TPHMIntrinsicGrid,
    constellation,
    envelope_splines,
    channel_indices: Sequence[int],
    endpoint_size: np.ndarray,
    endpoint_start: float,
    endpoint_rise: float,
    waveform_stop: float,
    frequency_start: float,
    frequency_stop: float,
    shape: WDMShape,
    tdi_generation: int = 1,
) -> tuple[_THMPartitionSpectrum, int]:
    sample_start = math.floor(endpoint_start / shape.DT) * shape.DT
    support_pixels = max(1, int(math.ceil((waveform_stop - sample_start) / shape.DT)))
    packet_time_pixels = _next_power_of_two(support_pixels + 2 * shape.mult)
    active = endpoint_size[endpoint_size > 0]
    if active.size:
        packet_time_pixels = max(packet_time_pixels, int(np.max(active)))
    packet_time_pixels = min(packet_time_pixels, shape.nt)
    sample_count = packet_time_pixels * shape.nf
    sample_times = sample_start + shape.dt * np.arange(sample_count)
    live = (
        (sample_times >= endpoint_start)
        & (sample_times <= waveform_stop)
        & (sample_times >= intrinsic.waveform_time[0])
        & (sample_times <= intrinsic.waveform_time[-1])
    )
    analytic = np.zeros((len(channel_indices), sample_count), dtype=np.complex128)
    if np.any(live):
        live_times = sample_times[live]
        endpoint_tdi = _response_from_tphm_envelopes(
            live_times,
            source,
            intrinsic,
            constellation,
            envelope_splines=envelope_splines,
            tdi_generation=tdi_generation,
        )
        weight = _smooth_step(live_times, endpoint_start, endpoint_start + endpoint_rise)
        weight *= 1.0 - _smooth_step(live_times, waveform_stop, waveform_stop + shape.dt)
        for local_channel, channel in enumerate(channel_indices):
            value = np.sum(
                endpoint_tdi.raw_m[channel] - 1j * endpoint_tdi.raw_mf[channel], axis=0
            )
            analytic[local_channel, live] = weight * value
    spectrum = np.fft.fft(analytic, axis=1) * shape.dt
    return (
        _THMPartitionSpectrum(
            carrier_index=-1,
            block_start=sample_start,
            nonzero_start=endpoint_start,
            nonzero_stop=waveform_stop + shape.dt,
            frequency_min=frequency_start,
            frequency_max=frequency_stop,
            frequency_pad=shape.FB,
            heterodyne_layer=0,
            sample_dt=shape.dt,
            spectrum=spectrum,
        ),
        sample_count,
    )


def generate_tphm_tdi_wdm(
    source: TPHMSourceParams = TPHMSourceParams(),
    shape: WDMShape = WDMShape(),
    channels: Iterable[str] = CHANNELS,
    *,
    modes: Sequence[int | tuple[int, int]] | str = "default",
    nsmax: int = 10000,
    coefficient_backend: str = "auto",
    compute_wdm: bool = True,
    wdm_method: str = "partitioned-fft",
    tapestry_response_order: int = 1,
    partition_bandwidth_hz: float = PARTITION_FFT_BANDWIDTH_HZ,
    partition_roll_seconds: float = PARTITION_FFT_ROLL_SECONDS,
    tdi_generation: int = 1,
) -> TPHMTDIWDMResult:
    """Generate TPHM, sparse X/Y/Z TDI, and partitioned or Tapestry WDM pixels."""

    requested = tuple(channel.upper() for channel in channels)
    if not requested or any(channel not in CHANNELS for channel in requested):
        raise ValueError("channels must be selected from X, Y, Z")
    if wdm_method not in {"partitioned-fft", "tapestry"}:
        raise ValueError("wdm_method must be 'partitioned-fft' or 'tapestry'")
    if tdi_generation not in (1, 2):
        raise ValueError("tdi_generation must be 1 or 2")
    if tapestry_response_order not in (0, 1):
        raise ValueError("tapestry_response_order must be 0 or 1")
    channel_indices = [CHANNELS.index(channel) for channel in requested]
    timings: dict[str, float] = {}

    started = time.perf_counter()
    intrinsic, constellation = build_tphm_intrinsic_grid(
        source,
        shape,
        modes=modes,
        nsmax=nsmax,
        coefficient_backend=coefficient_backend,
    )
    timings["intrinsic_and_precession"] = time.perf_counter() - started

    envelope_splines = _source_envelope_splines(intrinsic)
    started = time.perf_counter()
    tdi = _response_from_tphm_envelopes(
        intrinsic.response_time,
        source,
        intrinsic,
        constellation,
        envelope_splines=envelope_splines,
        tdi_generation=tdi_generation,
    )
    timings["sparse_tdi"] = time.perf_counter() - started

    started = time.perf_counter()
    bands = build_tphm_planning_bands(intrinsic, tdi, shape)
    timings["track_planning"] = time.perf_counter() - started
    if not compute_wdm:
        empty = {channel: _empty_channel(shape) for channel in requested}
        timings["fast_wdm"] = 0.0
        return TPHMTDIWDMResult(
            source, shape, intrinsic, tdi, bands, empty, (), timings
        )

    total_mass = intrinsic.model.total_mass
    delay_margin = CONSTELLATION_LIGHT_TIME_SECONDS + (
        8.0 if tdi_generation == 2 else 4.0
    ) * NOMINAL_ARM_LIGHT_TIME_SECONDS
    endpoint_start = source.tc - delay_margin - 10000.0 * total_mass
    endpoint_rise = 5000.0 * total_mass
    waveform_stop = min(
        shape.Tobs,
        source.tc + CONSTELLATION_LIGHT_TIME_SECONDS + 1000.0 * total_mass,
    )
    if endpoint_start <= 0.0:
        endpoint_start = min(0.25 * waveform_stop, max(shape.DT, waveform_stop - endpoint_rise))
    if endpoint_start >= waveform_stop:
        endpoint_start = max(0.0, waveform_stop - endpoint_rise)
    endpoint_rise = min(endpoint_rise, waveform_stop - endpoint_start)
    if endpoint_rise <= 0.0:
        raise ValueError("partition endpoint has no overlap with the observation")

    started = time.perf_counter()
    (
        early_mid,
        early_size,
        endpoint_mid,
        endpoint_size,
        endpoint_frequency_start,
        endpoint_frequency_stop,
    ) = _compact_packet_plans(
        bands, endpoint_start, endpoint_rise, waveform_stop, shape
    )
    timings["partition_plan"] = time.perf_counter() - started

    early_blocks = None
    tapestry_channels = None
    if wdm_method == "partitioned-fft":
        started = time.perf_counter()
        early_blocks, diagnostics, early_samples = _build_early_blocks(
            tdi,
            intrinsic,
            bands,
            channel_indices,
            early_mid,
            early_size,
            endpoint_start,
            endpoint_rise,
            shape,
            bandwidth_hz=float(partition_bandwidth_hz),
            roll_seconds=float(partition_roll_seconds),
        )
        timings["partition_early_fft"] = time.perf_counter() - started
    else:
        from phenomtphm_tdi_tapestry import build_tapestry_early_wdm

        started = time.perf_counter()
        tapestry_channels, diagnostics, early_samples, tapestry_timings = (
            build_tapestry_early_wdm(
                source,
                intrinsic,
                tdi,
                bands,
                constellation,
                requested,
                early_mid,
                early_size,
                endpoint_start,
                endpoint_rise,
                shape,
                bandwidth_hz=float(partition_bandwidth_hz),
                roll_seconds=float(partition_roll_seconds),
                response_order=tapestry_response_order,
                tdi_generation=tdi_generation,
            )
        )
        timings.update(tapestry_timings)
        timings["tapestry_early_total"] = time.perf_counter() - started

    started = time.perf_counter()
    endpoint_block, endpoint_samples = _build_endpoint_block(
        source,
        intrinsic,
        constellation,
        envelope_splines,
        channel_indices,
        endpoint_size,
        endpoint_start,
        endpoint_rise,
        waveform_stop,
        endpoint_frequency_start,
        endpoint_frequency_stop,
        shape,
        tdi_generation=tdi_generation,
    )
    timings["partition_endpoint_fft"] = time.perf_counter() - started

    started = time.perf_counter()
    early_channels = (
        _partition_blocks_to_wdm(
            early_blocks, early_mid, early_size, shape, len(channel_indices)
        )
        if early_blocks is not None
        else [tapestry_channels[channel] for channel in requested]
    )
    endpoint_channels = _partition_blocks_to_wdm(
        (endpoint_block,), endpoint_mid, endpoint_size, shape, len(channel_indices)
    )
    output = {
        CHANNELS[channel]: _merge_sparse_channels(
            early_channels[local], endpoint_channels[local], shape
        )
        for local, channel in enumerate(channel_indices)
    }
    timings["partition_packet_wdm"] = time.perf_counter() - started
    timings["partition_blocks"] = float(len(diagnostics) + 1)
    timings["partition_early_samples"] = float(early_samples)
    timings["partition_endpoint_samples"] = float(endpoint_samples)
    timings["partition_endpoint_start"] = endpoint_start
    timings["partition_endpoint_rise"] = endpoint_rise
    timings["fast_wdm"] = (
        timings["partition_plan"]
        + timings.get("partition_early_fft", timings.get("tapestry_early_total", 0.0))
        + timings["partition_endpoint_fft"]
        + timings["partition_packet_wdm"]
    )
    return TPHMTDIWDMResult(
        source,
        shape,
        intrinsic,
        tdi,
        bands,
        output,
        tuple(diagnostics),
        timings,
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", default="default")
    parser.add_argument("--channels", default="XYZ")
    parser.add_argument("--m1-solar", type=float, default=TPHMSourceParams.m1_solar)
    parser.add_argument("--m2-solar", type=float, default=TPHMSourceParams.m2_solar)
    parser.add_argument("--chi1x", type=float, default=TPHMSourceParams.chi1x)
    parser.add_argument("--chi1y", type=float, default=TPHMSourceParams.chi1y)
    parser.add_argument("--chi1z", type=float, default=TPHMSourceParams.chi1)
    parser.add_argument("--chi2x", type=float, default=TPHMSourceParams.chi2x)
    parser.add_argument("--chi2y", type=float, default=TPHMSourceParams.chi2y)
    parser.add_argument("--chi2z", type=float, default=TPHMSourceParams.chi2)
    parser.add_argument("--phic", type=float, default=TPHMSourceParams.phic)
    parser.add_argument("--tc", type=float, default=TPHMSourceParams.tc)
    parser.add_argument("--distance-gpc", type=float, default=TPHMSourceParams.distance_gpc)
    parser.add_argument("--theta", type=float, default=TPHMSourceParams.ecliptic_colatitude)
    parser.add_argument("--lambda", dest="longitude", type=float, default=TPHMSourceParams.ecliptic_longitude)
    parser.add_argument("--psi", type=float, default=TPHMSourceParams.polarization)
    parser.add_argument("--cosi", type=float, default=TPHMSourceParams.cos_inclination)
    parser.add_argument("--tau-ref", type=float)
    parser.add_argument("--nf", type=int, default=WDMShape.nf)
    parser.add_argument("--nt", type=int, default=WDMShape.nt)
    parser.add_argument("--dt", type=float, default=WDMShape.dt)
    parser.add_argument("--tdi-generation", type=int, choices=(1, 2), default=1)
    parser.add_argument("--nsmax", type=int, default=10000)
    parser.add_argument(
        "--coefficient-backend",
        choices=("auto", "native", "python", "reference"),
        default="auto",
    )
    parser.add_argument("--partition-bandwidth-hz", type=float, default=PARTITION_FFT_BANDWIDTH_HZ)
    parser.add_argument("--partition-roll-seconds", type=float, default=PARTITION_FFT_ROLL_SECONDS)
    parser.add_argument("--wdm-method", choices=("partitioned-fft", "tapestry"), default="partitioned-fft")
    parser.add_argument("--tapestry-response-order", type=int, choices=(0, 1), default=1)
    parser.add_argument("--no-wdm", action="store_true")
    parser.add_argument("--timing", action="store_true")
    args = parser.parse_args()
    source = TPHMSourceParams(
        m1_solar=args.m1_solar,
        m2_solar=args.m2_solar,
        chi1=args.chi1z,
        chi2=args.chi2z,
        chi1x=args.chi1x,
        chi1y=args.chi1y,
        chi2x=args.chi2x,
        chi2y=args.chi2y,
        phic=args.phic,
        tc=args.tc,
        distance_gpc=args.distance_gpc,
        ecliptic_colatitude=args.theta,
        ecliptic_longitude=args.longitude,
        polarization=args.psi,
        cos_inclination=args.cosi,
        tau_ref=args.tau_ref,
    )
    shape = WDMShape(nf=args.nf, nt=args.nt, dt=args.dt)
    result = generate_tphm_tdi_wdm(
        source,
        shape,
        channels=tuple(args.channels.upper()),
        modes=args.modes,
        nsmax=args.nsmax,
        coefficient_backend=args.coefficient_backend,
        compute_wdm=not args.no_wdm,
        wdm_method=args.wdm_method,
        tapestry_response_order=args.tapestry_response_order,
        partition_bandwidth_hz=args.partition_bandwidth_hz,
        partition_roll_seconds=args.partition_roll_seconds,
        tdi_generation=args.tdi_generation,
    )
    print(f"waveform_samples {result.intrinsic.waveform_time.size}")
    print(f"response_samples {result.tdi.time.size}")
    for channel in result.channels:
        print(f"channel {channel} active_pixels {result.channels[channel].values.size}")
    if args.timing:
        for key, value in result.timings.items():
            print(f"{key} {value:.9e}")


if __name__ == "__main__":
    _main()


__all__ = [
    "TPHMIntrinsicGrid",
    "TPHMPlanningBands",
    "TPHMSourceParams",
    "TPHMTDIWDMResult",
    "build_tphm_intrinsic_grid",
    "build_tphm_planning_bands",
    "compute_tphm_tdi_grid",
    "generate_tphm_tdi_wdm",
]
