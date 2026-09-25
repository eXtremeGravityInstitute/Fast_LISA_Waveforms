#!/usr/bin/env python3
# Python LISA response and WDM port: Copyright (C) 2026 Neil Cornish.
# SPDX-License-Identifier: GPL-3.0-or-later
# Distributed without warranty; see the GNU GPL for details.
# The imported LAL-derived waveform model retains its upstream notices.
"""Multi-harmonic ``TDI Tapestry`` generator for massive black-hole binaries.

The existing thread path applies TDI and WDM separately to every folded THM
carrier.  This diagnostic instead sums the intrinsic mode spectra into complex
plus/cross Meyer packets, obtains both real WDM quadratures from those two
packet transforms, and applies one shared complex TDI response at each WDM
pixel center.  Merger and ringdown are supplied by one short, exact summed-mode
TDI calculation and native-bin FFT per requested channel.  The older
per-carrier quadrature construction remains available through
``--inspiral-diagnostic``, while ``--validate-thread`` compares the complete
result with :func:`phenomthm_tdi_wdm.generate_thm_tdi_wdm`.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import math
from pathlib import Path
import time
import warnings

import numpy as np
from scipy.interpolate import CubicSpline

from phenomthm_tdi_wdm import (
    CHANNELS,
    TRIPLES,
    THMIntrinsicGrid,
    THMTDIGrid,
    _response_from_modes,
    _response_geometry,
    _tdi_delays_and_coefficients,
    build_thm_intrinsic_grid,
    compute_thm_tdi_grid,
    folded_projection,
    generate_thm_tdi_wdm,
)
from phenomt22_wdm_post_tdi_response import (
    Metrics,
    _interpolate_response_component,
    _metrics,
    _monochromatic_response_basis,
)
from tdi2_response import monochromatic_tdi2_transfer
from phenomt_tdi_wdm import (
    PI,
    DEFAULT_WDM_BLEND_ENDPOINT,
    DEFAULT_WDM_BLEND_HALF_WIDTH_LAYERS,
    SHORTFFT_MERGER_TAPER_MARGIN_SECONDS,
    T_CUT_FREQ,
    SourceParams,
    WDMChannel,
    WDMShape,
    _eval_cubic_spline_array,
    apply_short_fft_threshold,
    ftran_spa_only,
    make_ap_spline,
    merge_pixel_plans,
    plan_endpoint_taper_flat_time,
    phitilde,
    prune_nonincreasing_frequency_samples,
    tdi_frequency_track,
    transform_plan,
    wdm_pixels_range,
    wdm_channel_from_dense,
    wdm_pixels_add_merger_frequency_tail,
    wdm_track,
)


@dataclass(frozen=True)
class ModePackets:
    mode: tuple[int, int]
    keys: np.ndarray
    cosine: np.ndarray
    sine: np.ndarray
    plus: np.ndarray
    plus_quadrature: np.ndarray
    cross: np.ndarray
    cross_quadrature: np.ndarray
    reference: dict[str, np.ndarray]


@dataclass(frozen=True)
class TapestryResult:
    source: SourceParams
    shape: WDMShape
    maximum_frequency_hz: float
    modes: tuple[ModePackets, ...]
    keys: np.ndarray
    reference: dict[str, np.ndarray]
    tapestry: dict[str, np.ndarray]
    carrier_tapestry: dict[str, np.ndarray]
    response_nodes: np.ndarray
    response_method: str
    metrics: dict[str, Metrics]
    carrier_metrics: dict[str, Metrics]
    combined_carrier_metrics: dict[str, Metrics]
    mode_metrics: dict[tuple[str, tuple[int, int]], Metrics]
    timings: dict[str, float]


@dataclass(frozen=True)
class CompleteTapestryResult:
    """Production-like Tapestry inspiral plus exact common endpoint."""

    source: SourceParams
    shape: WDMShape
    intrinsic: THMIntrinsicGrid
    channels: dict[str, WDMChannel]
    replace_start_hz: float
    endpoint_start_s: float
    endpoint_stop_s: float
    endpoint_sample_count: int
    timings: dict[str, float]
    reference_metrics: dict[str, Metrics]


def _clip_plan(
    nmid: np.ndarray, nsize: np.ndarray, maximum_layer: int
) -> None:
    nmid[maximum_layer + 1 :] = -1
    nsize[maximum_layer + 1 :] = 0


def _extend_setup_to_frequency(
    setup: np.ndarray,
    track: np.ndarray,
    target_frequency: float,
) -> np.ndarray:
    out = np.asarray(setup, dtype=np.float64).copy()
    stop = min(max(int(out[4]), 1), track.size - 1)
    while stop + 1 < track.size and float(track[stop]) < target_frequency:
        stop += 1
    out[4] = float(stop)
    return out


def _keys(channel: WDMChannel, shape: WDMShape) -> np.ndarray:
    return channel.listn * (shape.nf + 1) + channel.listm


def _assert_packet_keys(reference: np.ndarray, channel: WDMChannel) -> None:
    if not np.array_equal(reference, channel.listn * (channel.nmid.size + 1) + channel.listm):
        # WDMShape.nf equals nmid.size.  Keep the diagnostic explicit if that
        # representation ever changes.
        raise RuntimeError("carrier quadratures produced different WDM support")


def _subset_channel(channel: WDMChannel, keep: np.ndarray) -> WDMChannel:
    return WDMChannel(
        channel.freq,
        channel.phase,
        channel.amplitude,
        channel.nmid,
        channel.nsize,
        channel.listn[keep],
        channel.listm[keep],
        channel.values[keep],
    )


def _first_frequency_crossing_time(
    times: np.ndarray, track: np.ndarray, frequency_hz: float
) -> float:
    crossings = np.flatnonzero(track >= frequency_hz)
    if not crossings.size:
        return float(times[-1])
    upper = int(crossings[0])
    lower = max(upper - 1, 0)
    if upper == lower or track[upper] == track[lower]:
        return float(times[upper])
    fraction = (frequency_hz - track[lower]) / (track[upper] - track[lower])
    return float(times[lower] + fraction * (times[upper] - times[lower]))


def _accumulate(union_keys: np.ndarray, keys: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = np.zeros(union_keys.size, dtype=np.float64)
    location = np.searchsorted(union_keys, keys)
    if np.any(location >= union_keys.size) or not np.array_equal(union_keys[location], keys):
        raise RuntimeError("mode packet lies outside the Tapestry union")
    np.add.at(out, location, values)
    return out


def _response_table(
    source: SourceParams,
    constellation,
    unique_times: np.ndarray,
    pixel_frequency: np.ndarray,
    pixel_time_columns: np.ndarray,
    spacing_hz: float,
) -> tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Build plus-cosine and cross-cosine TDI symbols on the pixel union."""

    minimum = float(np.min(pixel_frequency))
    maximum = float(np.max(pixel_frequency))
    lower = max(minimum - spacing_hz, max(1.0e-8, 0.25 * minimum))
    upper = maximum + spacing_hz
    count = max(int(math.ceil((upper - lower) / spacing_hz)) + 1, 5)
    nodes = np.linspace(lower, upper, count, dtype=np.float64)
    constellation_time, l_splines, p_splines, v_splines = constellation

    # folded_projection() has already rotated h+ and hx by psi.  The response
    # bases therefore use psi=0 and depend only on sky position and frequency.
    response_source = replace(source, polarization=0.0)
    plus_nodes = {channel: [] for channel in CHANNELS}
    cross_sine_nodes = {channel: [] for channel in CHANNELS}
    for frequency in nodes:
        plus = _monochromatic_response_basis(
            float(frequency),
            unique_times,
            response_source,
            constellation_time,
            l_splines,
            p_splines,
            v_splines,
            1.0,
            0.0,
        )
        cross_sine = _monochromatic_response_basis(
            float(frequency),
            unique_times,
            response_source,
            constellation_time,
            l_splines,
            p_splines,
            v_splines,
            0.0,
            1.0,
        )
        for channel in CHANNELS:
            plus_nodes[channel].append(plus[channel])
            cross_sine_nodes[channel].append(cross_sine[channel])

    response: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for channel in CHANNELS:
        plus_pixel = _interpolate_response_component(
            nodes,
            np.stack(plus_nodes[channel], axis=0),
            pixel_frequency,
            pixel_time_columns,
        )
        # With psi=0, across=1 generates hx=sin(phi).  Its returned analytic
        # response is -i times the response to hx=cos(phi).
        cross_pixel = 1j * _interpolate_response_component(
            nodes,
            np.stack(cross_sine_nodes[channel], axis=0),
            pixel_frequency,
            pixel_time_columns,
        )
        response[channel] = (plus_pixel, cross_pixel)
    return nodes, response


def _direct_pixel_response(
    source: SourceParams,
    constellation,
    unique_times: np.ndarray,
    pixel_frequency: np.ndarray,
    pixel_time_columns: np.ndarray,
    tdi_generation: int = 1,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Evaluate the monochromatic TDI symbol directly at every sparse pixel."""

    if tdi_generation == 2:
        grid = monochromatic_tdi2_transfer(
            unique_times, pixel_frequency, pixel_time_columns,
            source.ecliptic_colatitude,
            source.ecliptic_longitude, constellation[2],
        )
        return {
            channel: (grid[0, index, 0], grid[0, index, 1])
            for index, channel in enumerate(CHANNELS)
        }
    if tdi_generation != 1:
        raise ValueError("tdi_generation must be 1 or 2")

    arm_length, kr, app, apm, acp, acm = _response_geometry(
        unique_times, source, constellation
    )
    response: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    local_time = unique_times[pixel_time_columns]
    for channel, triple in zip(CHANNELS, TRIPLES):
        delays, coefp, coefc = _tdi_delays_and_coefficients(
            triple, unique_times, arm_length, kr, app, apm, acp, acm
        )
        local_delay = delays[pixel_time_columns] - local_time[:, np.newaxis]
        phasor = np.exp(
            1j * 2.0 * PI * pixel_frequency[:, np.newaxis] * local_delay
        )
        response[channel] = (
            np.sum(coefp[pixel_time_columns] * phasor, axis=1),
            np.sum(coefc[pixel_time_columns] * phasor, axis=1),
        )
    return response


def _mode_packet_plan(
    source: SourceParams,
    shape: WDMShape,
    intrinsic: THMIntrinsicGrid,
    tdi: THMTDIGrid,
    carrier: int,
    maximum_frequency_hz: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]], float]:
    """Return one X/Y/Z-union packet plan and each channel's SPA arrays."""

    m1, m2, _chi1, _chi2 = source.masses_seconds()
    total_mass = m1 + m2
    chirp_mass = (m1 * m2) ** (3.0 / 5.0) / total_mass ** (1.0 / 5.0)
    target = maximum_frequency_hz + 2.0 * shape.FB
    maximum_layer = min(int(math.floor(maximum_frequency_hz / shape.DF)), shape.nf - 1)
    union_mid = np.full(shape.nf, -1, dtype=np.int64)
    union_size = np.zeros(shape.nf, dtype=np.int64)
    spectra: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    channel_inputs: dict[str, tuple[int, np.ndarray, np.ndarray, np.ndarray]] = {}
    safe_time_max = float(tdi.time[-1])

    for channel_index, channel in enumerate(CHANNELS):
        phase = tdi.channel_phase[channel_index, carrier]
        track = tdi_frequency_track(
            tdi.time, phase, tdi.intrinsic_frequency[carrier]
        )
        safe_time_max = min(
            safe_time_max,
            _first_frequency_crossing_time(
                tdi.time, track, maximum_frequency_hz
            ),
        )
        setup = transform_plan(
            chirp_mass,
            total_mass,
            source.tc,
            tdi.time,
            2.0 * PI * track,
            source.tc + total_mass * T_CUT_FREQ,
            float(intrinsic.ring_frequency_hz[carrier]),
            float(intrinsic.damping_rate_hz[carrier]),
            float(tdi.time[-1]),
            shape,
            center_source_time=tdi.detector_time,
        )
        setup = _extend_setup_to_frequency(setup, track, target)
        stop = int(setup[4])
        local_mid, local_size = wdm_pixels_range(
            tdi.time,
            track,
            float(tdi.time[0]),
            float(tdi.time[stop]),
            shape,
        )
        _clip_plan(local_mid, local_size, maximum_layer)
        merge_pixel_plans(union_mid, union_size, local_mid, local_size, shape)
        channel_inputs[channel] = (channel_index, phase, track, setup)

    # Transform only after the X/Y/Z support union is complete.  Otherwise an
    # earlier channel can be emitted on a smaller plan than a later channel.
    for channel in CHANNELS:
        channel_index, phase, track, setup = channel_inputs[channel]
        frequency, fourier_phase, fourier_amplitude = ftran_spa_only(
            setup,
            tdi.time,
            tdi.signed_amplitude[channel_index, carrier],
            phase,
            shape,
        )
        frequency, fourier_phase, fourier_amplitude = (
            prune_nonincreasing_frequency_samples(
                frequency, fourier_phase, fourier_amplitude
            )
        )
        spectra[channel] = (
            wdm_track(
                frequency,
                fourier_phase,
                fourier_amplitude,
                union_mid,
                union_size,
                shape,
            ),
            track,
        )
    return union_mid, union_size, spectra, safe_time_max


def _intrinsic_mode_spectrum(
    source: SourceParams,
    shape: WDMShape,
    intrinsic: THMIntrinsicGrid,
    tdi: THMTDIGrid,
    carrier: int,
    target_frequency_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct one unprojected intrinsic carrier SPA spectrum."""

    m1, m2, _chi1, _chi2 = source.masses_seconds()
    total_mass = m1 + m2
    chirp_mass = (m1 * m2) ** (3.0 / 5.0) / total_mass ** (1.0 / 5.0)
    amplitude = np.asarray(
        make_ap_spline(intrinsic.spline_time, intrinsic.amplitude[carrier])(
            tdi.time
        ),
        dtype=np.float64,
    )
    phase = np.asarray(
        make_ap_spline(intrinsic.spline_time, intrinsic.phase[carrier])(tdi.time),
        dtype=np.float64,
    )
    track = tdi_frequency_track(
        tdi.time, phase, tdi.intrinsic_frequency[carrier]
    )
    setup = transform_plan(
        chirp_mass,
        total_mass,
        source.tc,
        tdi.time,
        2.0 * PI * track,
        source.tc + total_mass * T_CUT_FREQ,
        float(intrinsic.ring_frequency_hz[carrier]),
        float(intrinsic.damping_rate_hz[carrier]),
        float(tdi.time[-1]),
        shape,
        center_source_time=tdi.detector_time,
    )
    setup = _extend_setup_to_frequency(setup, track, target_frequency_hz)
    return _spa_spectrum_from_setup(
        setup, tdi.time, amplitude, phase, shape
    )


def _spa_spectrum_from_setup(
    setup: np.ndarray,
    times: np.ndarray,
    amplitude: np.ndarray,
    phase: np.ndarray,
    shape: WDMShape,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate and sanitize one carrier's numerical SPA spectrum."""

    frequency, fourier_phase, fourier_amplitude = ftran_spa_only(
        setup, times, amplitude, phase, shape
    )
    frequency, fourier_phase, fourier_amplitude = (
        prune_nonincreasing_frequency_samples(
            frequency, fourier_phase, fourier_amplitude
        )
    )
    return frequency, fourier_phase, fourier_amplitude


def _plan_keys(nmid: np.ndarray, nsize: np.ndarray, shape: WDMShape) -> np.ndarray:
    """Return the sorted sparse coordinates represented by a packet plan."""

    parts: list[np.ndarray] = []
    for layer in np.flatnonzero((nmid >= 0) & (nsize > 0)):
        if layer <= 0 or layer >= shape.nf:
            continue
        half = int(nsize[layer]) // 2
        rows = np.arange(int(nsize[layer]), dtype=np.int64) + int(nmid[layer]) - half
        rows = rows[(rows >= 0) & (rows < shape.nt)]
        if rows.size:
            parts.append(rows * (shape.nf + 1) + int(layer))
    if not parts:
        return np.empty(0, dtype=np.int64)
    return np.sort(np.concatenate(parts))


def _sparse_metrics(reference: WDMChannel, candidate: WDMChannel, shape: WDMShape) -> Metrics:
    """Compare two sparse channels on their coordinate union."""

    reference_keys = _keys(reference, shape)
    candidate_keys = _keys(candidate, shape)
    union = np.union1d(reference_keys, candidate_keys)
    reference_values = np.zeros(union.size, dtype=np.float64)
    candidate_values = np.zeros(union.size, dtype=np.float64)
    reference_values[np.searchsorted(union, reference_keys)] = reference.values
    candidate_values[np.searchsorted(union, candidate_keys)] = candidate.values
    return _metrics(reference_values, candidate_values)


def _intrinsic_mode_packets(
    spectrum: tuple[np.ndarray, np.ndarray, np.ndarray],
    nmid: np.ndarray,
    nsize: np.ndarray,
    shape: WDMShape,
) -> tuple[WDMChannel, WDMChannel]:
    """Transform one intrinsic carrier into separate cosine/sine packets."""

    frequency, fourier_phase, fourier_amplitude = spectrum
    cosine = wdm_track(
        frequency, fourier_phase, fourier_amplitude, nmid, nsize, shape
    )
    sine = wdm_track(
        frequency,
        fourier_phase - 0.5 * PI,
        fourier_amplitude,
        nmid,
        nsize,
        shape,
    )
    if not np.array_equal(_keys(cosine, shape), _keys(sine, shape)):
        raise RuntimeError("intrinsic cosine and sine packets do not align")
    return cosine, sine


def _combined_polarization_packets(
    spectra: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    projection: np.ndarray,
    nmid: np.ndarray,
    nsize: np.ndarray,
    requested_keys: np.ndarray,
    shape: WDMShape,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Transform the summed THM polarizations once each.

    Each complex Meyer packet retains both quadratures of a real polarization.
    If ``packet`` is the analytic packet, the ordinary real WDM coefficient is
    selected from ``Re(packet)`` or ``Im(packet)`` according to WDM parity.  The
    quadrature waveform has spectrum ``i H(f)``, so its packet follows without a
    second transform.  This reduces five cosine/sine carrier pairs to one complex
    plus packet and one complex cross packet per active frequency layer.
    """

    active_layers = [
        int(layer)
        for layer in np.flatnonzero((nmid >= 0) & (nsize > 0))
        if 0 < layer < shape.nf
    ]
    groups: dict[int, list[int]] = {}
    for layer in active_layers:
        groups.setdefault(int(nsize[layer]), []).append(layer)

    # Match wdm_track(): Akima amplitude interpolation and a natural cubic phase.
    amplitude_splines = [make_ap_spline(freq, amp) for freq, _phase, amp in spectra]
    # The Fourier phase deliberately retains the production natural-cubic
    # convention used by wdm_track().
    phase_splines = [
        CubicSpline(freq, phase, bc_type="natural")
        for freq, phase, _amp in spectra
    ]

    output_keys: list[np.ndarray] = []
    plus_parts: list[np.ndarray] = []
    plus_quadrature_parts: list[np.ndarray] = []
    cross_parts: list[np.ndarray] = []
    cross_quadrature_parts: list[np.ndarray] = []

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
        start_times = (
            nmid[layer_array].astype(np.float64) - float(half)
        ) * shape.DT
        shifted_phase = 2.0 * PI * fgrid * start_times[:, np.newaxis]
        plus_spectrum = np.zeros(fgrid.shape, dtype=np.complex128)
        cross_spectrum = np.zeros_like(plus_spectrum)
        valid_bin = base_n[np.newaxis, :] > 0

        for carrier, ((frequency, _phase, _amplitude), amp_spline, phase_spline) in enumerate(
            zip(spectra, amplitude_splines, phase_splines)
        ):
            valid = valid_bin & (fgrid > frequency[0]) & (fgrid < frequency[-1])
            if not np.any(valid):
                continue
            fvalues = fgrid[valid]
            amplitude = _eval_cubic_spline_array(amp_spline, fvalues)
            phase = _eval_cubic_spline_array(phase_spline, fvalues)
            analytic = amplitude * np.exp(1j * (phase + shifted_phase[valid]))
            coeff = projection[carrier]
            plus_spectrum[valid] += complex(coeff[0], -coeff[1]) * analytic
            cross_spectrum[valid] += complex(coeff[2], -coeff[3]) * analytic

        plus_packet = np.fft.ifft(
            plus_spectrum * window[np.newaxis, :], axis=1
        ) * packet_size
        cross_packet = np.fft.ifft(
            cross_spectrum * window[np.newaxis, :], axis=1
        ) * packet_size
        even = ((base_n[np.newaxis, :] + layer_array[:, np.newaxis]) % 2) == 0
        imag_sign = np.where((layer_array % 2) == 0, 1.0, -1.0)[:, np.newaxis]

        plus_value = scale * np.where(even, plus_packet.real, imag_sign * plus_packet.imag)
        plus_quadrature = scale * np.where(
            even, -plus_packet.imag, imag_sign * plus_packet.real
        )
        cross_value = scale * np.where(even, cross_packet.real, imag_sign * cross_packet.imag)
        cross_quadrature = scale * np.where(
            even, -cross_packet.imag, imag_sign * cross_packet.real
        )

        for row, layer in enumerate(layers):
            output_time = base_n + int(nmid[layer]) - half
            keep = (output_time >= 0) & (output_time < shape.nt)
            output_keys.append(output_time[keep] * (shape.nf + 1) + layer)
            plus_parts.append(plus_value[row, keep])
            plus_quadrature_parts.append(plus_quadrature[row, keep])
            cross_parts.append(cross_value[row, keep])
            cross_quadrature_parts.append(cross_quadrature[row, keep])

    all_keys = np.concatenate(output_keys)
    order = np.argsort(all_keys)
    all_keys = all_keys[order]
    location = np.searchsorted(all_keys, requested_keys)
    if np.any(location >= all_keys.size) or not np.array_equal(
        all_keys[location], requested_keys
    ):
        raise RuntimeError("combined polarization packets do not cover the requested union")

    def gather(parts: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(parts)[order][location]

    return (
        gather(plus_parts),
        gather(plus_quadrature_parts),
        gather(cross_parts),
        gather(cross_quadrature_parts),
    )


def generate_tapestry_diagnostic(
    source: SourceParams = SourceParams(),
    shape: WDMShape = WDMShape(),
    *,
    modes: str = "default",
    maximum_frequency_hz: float = 1.4e-2,
    response_spacing_hz: float = 5.0e-4,
    response_method: str = "direct",
    coefficient_backend: str = "python",
    tdi_generation: int = 1,
) -> TapestryResult:
    if maximum_frequency_hz <= shape.DF:
        raise ValueError("maximum frequency must exceed one WDM layer")
    if response_method not in {"direct", "table"}:
        raise ValueError("response_method must be 'direct' or 'table'")
    if tdi_generation == 2 and response_method == "table":
        raise ValueError("TDI-2 diagnostic currently requires direct pixel response")
    timings: dict[str, float] = {}

    start = time.perf_counter()
    intrinsic, constellation = build_thm_intrinsic_grid(
        source,
        shape,
        modes=modes,
        coefficient_backend=coefficient_backend,
        tdi_generation=tdi_generation,
    )
    timings["intrinsic"] = time.perf_counter() - start

    start = time.perf_counter()
    tdi = compute_thm_tdi_grid(
        source, intrinsic, constellation, tdi_generation=tdi_generation
    )
    timings["thread_tdi"] = time.perf_counter() - start
    projection = folded_projection(source, intrinsic.modes)
    packet_modes: list[ModePackets] = []
    mode_spectra: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    combined_mid = np.full(shape.nf, -1, dtype=np.int64)
    combined_size = np.zeros(shape.nf, dtype=np.int64)

    thread_reference_wdm_time = 0.0
    tapestry_spectrum_time = 0.0
    tapestry_carrier_wdm_time = 0.0
    for carrier, mode in enumerate(intrinsic.modes):
        start = time.perf_counter()
        nmid, nsize, reference_spectra, safe_time_max = _mode_packet_plan(
            source,
            shape,
            intrinsic,
            tdi,
            carrier,
            maximum_frequency_hz,
        )
        merge_pixel_plans(combined_mid, combined_size, nmid, nsize, shape)
        thread_reference_wdm_time += time.perf_counter() - start
        start = time.perf_counter()
        spectrum = _intrinsic_mode_spectrum(
            source,
            shape,
            intrinsic,
            tdi,
            carrier,
            maximum_frequency_hz + 2.0 * shape.FB,
        )
        tapestry_spectrum_time += time.perf_counter() - start
        mode_spectra.append(spectrum)
        start = time.perf_counter()
        cosine, sine = _intrinsic_mode_packets(spectrum, nmid, nsize, shape)
        tapestry_carrier_wdm_time += time.perf_counter() - start
        packet_time = cosine.listn.astype(np.float64) * shape.DT
        safe = (packet_time >= float(tdi.time[0])) & (packet_time <= safe_time_max)
        cosine = _subset_channel(cosine, safe)
        sine = _subset_channel(sine, safe)
        mode_keys = _keys(cosine, shape)
        reference: dict[str, np.ndarray] = {}
        for channel in CHANNELS:
            component = reference_spectra[channel][0]
            component = _subset_channel(component, safe)
            if not np.array_equal(mode_keys, _keys(component, shape)):
                raise RuntimeError(f"thread and intrinsic packets differ for {channel} {mode}")
            reference[channel] = component.values
        coeff = projection[carrier]
        packet_modes.append(
            ModePackets(
                mode=mode,
                keys=mode_keys,
                cosine=cosine.values,
                sine=sine.values,
                plus=coeff[0] * cosine.values + coeff[1] * sine.values,
                plus_quadrature=coeff[4] * cosine.values + coeff[5] * sine.values,
                cross=coeff[2] * cosine.values + coeff[3] * sine.values,
                cross_quadrature=coeff[6] * cosine.values + coeff[7] * sine.values,
                reference=reference,
            )
        )
    timings["thread_reference_wdm"] = thread_reference_wdm_time
    timings["tapestry_mode_spectra"] = tapestry_spectrum_time
    timings["tapestry_carrier_wdm"] = tapestry_carrier_wdm_time

    union_keys = np.unique(np.concatenate([mode.keys for mode in packet_modes]))
    plus = np.zeros(union_keys.size, dtype=np.float64)
    plus_quadrature = np.zeros_like(plus)
    cross = np.zeros_like(plus)
    cross_quadrature = np.zeros_like(plus)
    reference = {channel: np.zeros_like(plus) for channel in CHANNELS}
    for mode in packet_modes:
        location = np.searchsorted(union_keys, mode.keys)
        np.add.at(plus, location, mode.plus)
        np.add.at(plus_quadrature, location, mode.plus_quadrature)
        np.add.at(cross, location, mode.cross)
        np.add.at(cross_quadrature, location, mode.cross_quadrature)
        for channel in CHANNELS:
            np.add.at(reference[channel], location, mode.reference[channel])

    start = time.perf_counter()
    (
        combined_plus,
        combined_plus_quadrature,
        combined_cross,
        combined_cross_quadrature,
    ) = _combined_polarization_packets(
        mode_spectra,
        projection,
        combined_mid,
        combined_size,
        union_keys,
        shape,
    )
    timings["tapestry_combined_wdm"] = time.perf_counter() - start

    time_index = union_keys // (shape.nf + 1)
    frequency_layer = union_keys % (shape.nf + 1)
    unique_n, time_columns = np.unique(time_index, return_inverse=True)
    unique_times = unique_n.astype(np.float64) * shape.DT
    pixel_frequency = frequency_layer.astype(np.float64) * shape.DF

    start = time.perf_counter()
    if response_method == "direct":
        response_nodes = np.empty(0, dtype=np.float64)
        response = _direct_pixel_response(
            source,
            constellation,
            unique_times,
            pixel_frequency,
            time_columns,
            tdi_generation=tdi_generation,
        )
    else:
        response_nodes, response = _response_table(
            source,
            constellation,
            unique_times,
            pixel_frequency,
            time_columns,
            response_spacing_hz,
        )
    timings["shared_tdi_response"] = time.perf_counter() - start

    start = time.perf_counter()
    tapestry: dict[str, np.ndarray] = {}
    carrier_tapestry: dict[str, np.ndarray] = {}
    metrics: dict[str, Metrics] = {}
    carrier_metrics: dict[str, Metrics] = {}
    combined_carrier_metrics: dict[str, Metrics] = {}
    for channel in CHANNELS:
        response_plus, response_cross = response[channel]
        carrier_values = (
            response_plus.real * plus
            + response_plus.imag * plus_quadrature
            + response_cross.real * cross
            + response_cross.imag * cross_quadrature
        )
        values = (
            response_plus.real * combined_plus
            + response_plus.imag * combined_plus_quadrature
            + response_cross.real * combined_cross
            + response_cross.imag * combined_cross_quadrature
        )
        tapestry[channel] = values
        carrier_tapestry[channel] = carrier_values
        metrics[channel] = _metrics(reference[channel], values)
        carrier_metrics[channel] = _metrics(reference[channel], carrier_values)
        combined_carrier_metrics[channel] = _metrics(carrier_values, values)
    timings["shared_response_apply"] = time.perf_counter() - start

    mode_metrics: dict[tuple[str, tuple[int, int]], Metrics] = {}
    for mode in packet_modes:
        location = np.searchsorted(union_keys, mode.keys)
        for channel in CHANNELS:
            response_plus, response_cross = response[channel]
            values = (
                response_plus.real[location] * mode.plus
                + response_plus.imag[location] * mode.plus_quadrature
                + response_cross.real[location] * mode.cross
                + response_cross.imag[location] * mode.cross_quadrature
            )
            mode_metrics[(channel, mode.mode)] = _metrics(
                mode.reference[channel], values
            )

    return TapestryResult(
        source,
        shape,
        maximum_frequency_hz,
        tuple(packet_modes),
        union_keys,
        reference,
        tapestry,
        carrier_tapestry,
        response_nodes,
        response_method,
        metrics,
        carrier_metrics,
        combined_carrier_metrics,
        mode_metrics,
        timings,
    )


def generate_complete_tapestry(
    source: SourceParams = SourceParams(),
    shape: WDMShape = WDMShape(),
    *,
    modes: str = "default",
    channels: tuple[str, ...] = CHANNELS,
    coefficient_backend: str = "python",
    blend_endpoint: bool = DEFAULT_WDM_BLEND_ENDPOINT,
    blend_half_width_layers: float = DEFAULT_WDM_BLEND_HALF_WIDTH_LAYERS,
    validate_thread: bool = False,
    tdi_generation: int = 1,
) -> CompleteTapestryResult:
    """Generate the full MBH TDI+WDM signal with the Tapestry hybrid.

    The inspiral is assembled as two combined complex polarization packet
    transforms followed by one shared sparse X/Y/Z response.  Merger and
    ringdown use the exact summed-mode TDI response on the common short grid and
    its native-bin FFT packet transform.  No per-mode TDI AP extraction is done
    unless ``validate_thread`` explicitly requests the established thread-path
    comparison.
    """

    requested = tuple(channel.upper() for channel in channels)
    if tdi_generation == 2:
        warnings.warn(
            "THM Tapestry TDI-2 is experimental: use the partitioned-FFT "
            "generator for validated sparse TDI-2 WDM output",
            RuntimeWarning,
            stacklevel=2,
        )
    if not requested or any(channel not in CHANNELS for channel in requested):
        raise ValueError("channels must be selected from X, Y, Z")
    timings: dict[str, float] = {}
    total_start = time.perf_counter()

    start = time.perf_counter()
    intrinsic, constellation = build_thm_intrinsic_grid(
        source,
        shape,
        modes=modes,
        coefficient_backend=coefficient_backend,
        tdi_generation=tdi_generation,
    )
    timings["adaptive_ap"] = time.perf_counter() - start

    count = intrinsic.response_time.size
    response_time = intrinsic.response_time
    amplitude = np.asarray(intrinsic.amplitude[:, :count], dtype=np.float64)
    phase = np.asarray(intrinsic.phase[:, :count], dtype=np.float64)
    frequency = np.asarray(intrinsic.frequency[:, :count], dtype=np.float64)
    projection = folded_projection(source, intrinsic.modes)
    m1, m2, _chi1, _chi2 = source.masses_seconds()
    total_mass = m1 + m2
    chirp_mass = (m1 * m2) ** (3.0 / 5.0) / total_mass ** (1.0 / 5.0)
    transition_time = source.tc + total_mass * T_CUT_FREQ

    start = time.perf_counter()
    entries: list[dict[str, object]] = []
    driver_setup: np.ndarray | None = None
    driver_score = -1.0
    for carrier, mode in enumerate(intrinsic.modes):
        track = np.asarray(frequency[carrier], dtype=np.float64)
        setup = transform_plan(
            chirp_mass,
            total_mass,
            source.tc,
            response_time,
            2.0 * PI * track,
            transition_time,
            float(intrinsic.ring_frequency_hz[carrier]),
            float(intrinsic.damping_rate_hz[carrier]),
            float(response_time[-1]),
            shape,
            center_source_time=intrinsic.detector_time[:count],
        )
        join = float(np.interp(float(setup[5]), response_time, track))
        if not math.isfinite(join) or join <= 0.0:
            valid = track[np.isfinite(track) & (track > 0.0)]
            join = float(valid[-1]) if valid.size else shape.DF
        entries.append(
            {
                "carrier": carrier,
                "mode": mode,
                "track": track,
                "setup": setup,
                "join": join,
            }
        )
        score = float(setup[6]) / (float(setup[0]) * float(setup[1]))
        if score > driver_score:
            driver_score = score
            driver_setup = setup.copy()
    if driver_setup is None:
        raise RuntimeError("Tapestry endpoint planner did not find a valid FFT driver")

    blend_width = (
        float(blend_half_width_layers) * shape.DF if blend_endpoint else 0.0
    )
    preferred = [float(entry["join"]) for entry in entries if entry["mode"][1] >= 3]
    if not preferred:
        preferred = [float(entry["join"]) for entry in entries]
    original_flat_time = float(driver_setup[2]) + float(driver_setup[3])
    original_taper_guard = max(
        float(np.interp(original_flat_time, response_time, np.asarray(entry["track"])))
        for entry in entries
    )
    replace_start = max(min(preferred), original_taper_guard)
    response_merger = float(
        np.interp(source.tc, intrinsic.detector_time[:count], response_time)
    )
    driver_setup = plan_endpoint_taper_flat_time(
        driver_setup,
        response_time,
        ([np.asarray(entry["track"], dtype=np.float64) for entry in entries],),
        (max(0.0, replace_start - blend_width),),
        response_merger,
    )
    rise = float(driver_setup[3])
    endpoint_start = float(driver_setup[2])
    endpoint_count = int(driver_setup[1])
    endpoint_dt = float(driver_setup[0])
    endpoint_stop = endpoint_start + endpoint_dt * endpoint_count
    direct_start = endpoint_start + rise + SHORTFFT_MERGER_TAPER_MARGIN_SECONDS
    f_replace_stop = min(
        float(driver_setup[6]) / (endpoint_dt * float(endpoint_count)),
        0.5 / endpoint_dt,
    )
    flat_time = endpoint_start + rise
    taper_guard = max(
        float(np.interp(flat_time, response_time, np.asarray(entry["track"])))
        for entry in entries
    )
    if math.isfinite(taper_guard) and taper_guard > 0.0:
        replace_start = max(replace_start, taper_guard + blend_width)
    spa_stop_frequency = replace_start + shape.FB + blend_width
    timings["transform_plan"] = time.perf_counter() - start

    start = time.perf_counter()
    early_mid = np.full(shape.nf, -1, dtype=np.int64)
    early_size = np.zeros(shape.nf, dtype=np.int64)
    endpoint_mid = np.full(shape.nf, -1, dtype=np.int64)
    endpoint_size = np.zeros(shape.nf, dtype=np.int64)
    spectra: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for entry in entries:
        carrier = int(entry["carrier"])
        track = np.asarray(entry["track"], dtype=np.float64)
        setup_spa = np.asarray(entry["setup"], dtype=np.float64).copy()
        stop = min(max(int(setup_spa[4]), 1), response_time.size - 1)
        while stop + 1 < response_time.size and float(track[stop]) < spa_stop_frequency:
            stop += 1
        setup_spa[4] = float(stop)
        spectra.append(
            _spa_spectrum_from_setup(
                setup_spa,
                response_time,
                amplitude[carrier],
                phase[carrier],
                shape,
            )
        )
        local_mid, local_size = wdm_pixels_range(
            response_time,
            track,
            float(response_time[0]),
            float(response_time[stop]),
            shape,
        )
        merge_pixel_plans(early_mid, early_size, local_mid, local_size, shape)
        local_mid, local_size = wdm_pixels_range(
            response_time, track, direct_start, endpoint_stop, shape
        )
        merge_pixel_plans(endpoint_mid, endpoint_size, local_mid, local_size, shape)
    timings["mode_spectra_and_support"] = time.perf_counter() - start

    endpoint_time = endpoint_start + endpoint_dt * np.arange(
        endpoint_count, dtype=np.float64
    )
    start = time.perf_counter()
    endpoint_tdi = _response_from_modes(
        endpoint_time, source, intrinsic, constellation,
        extract_carrier_ap=False, tdi_generation=tdi_generation,
    )
    taper = np.ones(endpoint_count, dtype=np.float64)
    if rise > 0.0:
        local = endpoint_time - endpoint_start
        taper_mask = local < rise
        taper[taper_mask] = 0.5 * (1.0 - np.cos(PI * local[taper_mask] / rise))
    peak_index = int(np.argmax(np.max(np.abs(endpoint_tdi.channel_strain), axis=0)))
    wdm_pixels_add_merger_frequency_tail(
        endpoint_mid,
        endpoint_size,
        f_replace_stop,
        float(endpoint_time[peak_index]),
        shape,
    )
    timings["exact_endpoint_tdi"] = time.perf_counter() - start

    start = time.perf_counter()
    early_keys = _plan_keys(early_mid, early_size, shape)
    (
        plus,
        plus_quadrature,
        cross,
        cross_quadrature,
    ) = _combined_polarization_packets(
        spectra,
        projection,
        early_mid,
        early_size,
        early_keys,
        shape,
    )
    timings["combined_polarization_wdm"] = time.perf_counter() - start

    early_time_index = early_keys // (shape.nf + 1)
    early_frequency_layer = early_keys % (shape.nf + 1)
    unique_n, time_columns = np.unique(early_time_index, return_inverse=True)
    unique_times = unique_n.astype(np.float64) * shape.DT
    pixel_frequency = early_frequency_layer.astype(np.float64) * shape.DF
    start = time.perf_counter()
    response = _direct_pixel_response(
        source,
        constellation,
        unique_times,
        pixel_frequency,
        time_columns,
        tdi_generation=tdi_generation,
    )
    timings["shared_tdi_response"] = time.perf_counter() - start

    total_mid = early_mid.copy()
    total_size = early_size.copy()
    merge_pixel_plans(total_mid, total_size, endpoint_mid, endpoint_size, shape)
    all_frequency = np.concatenate([item[0] for item in spectra])
    all_phase = np.concatenate([item[1] for item in spectra])
    all_amplitude = np.concatenate([item[2] for item in spectra])
    channels_out: dict[str, WDMChannel] = {}
    response_apply_time = 0.0
    endpoint_wdm_time = 0.0
    for channel in requested:
        channel_index = CHANNELS.index(channel)
        response_plus, response_cross = response[channel]
        start = time.perf_counter()
        early_values = (
            response_plus.real * plus
            + response_plus.imag * plus_quadrature
            + response_cross.real * cross
            + response_cross.imag * cross_quadrature
        )
        response_apply_time += time.perf_counter() - start

        start = time.perf_counter()
        dense = np.zeros((shape.nt, shape.nf + 1), dtype=np.float64)
        dense[early_time_index, early_frequency_layer] = early_values
        apply_short_fft_threshold(
            dense,
            endpoint_mid,
            endpoint_size,
            endpoint_tdi.channel_strain[channel_index] * taper,
            driver_setup,
            replace_start,
            f_replace_stop,
            shape,
            blend_half_width=blend_width,
        )
        channels_out[channel] = wdm_channel_from_dense(
            all_frequency,
            all_phase,
            all_amplitude,
            total_mid,
            total_size,
            dense,
            shape,
        )
        endpoint_wdm_time += time.perf_counter() - start
    timings["shared_response_apply"] = response_apply_time
    timings["endpoint_fft_wdm"] = endpoint_wdm_time

    reference_metrics: dict[str, Metrics] = {}
    if validate_thread:
        start = time.perf_counter()
        reference = generate_thm_tdi_wdm(
            source,
            shape,
            requested,
            modes=modes,
            coefficient_backend=coefficient_backend,
            wdm_method="partitioned-fft" if tdi_generation == 2 else "spa",
            tdi_generation=tdi_generation,
            blend_endpoint=blend_endpoint,
            blend_half_width_layers=blend_half_width_layers,
        )
        timings["thread_validation"] = time.perf_counter() - start
        reference_metrics = {
            channel: _sparse_metrics(reference.channels[channel], channels_out[channel], shape)
            for channel in requested
        }

    timings["tapestry_total"] = time.perf_counter() - total_start
    return CompleteTapestryResult(
        source=source,
        shape=shape,
        intrinsic=intrinsic,
        channels=channels_out,
        replace_start_hz=replace_start,
        endpoint_start_s=endpoint_start,
        endpoint_stop_s=endpoint_stop,
        endpoint_sample_count=endpoint_count,
        timings=timings,
        reference_metrics=reference_metrics,
    )


def write_result(result: TapestryResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "channel",
                "reference_combined_match",
                "reference_combined_mismatch",
                "reference_combined_power_ratio",
                "reference_combined_relative_l2",
                "reference_carrier_match",
                "reference_carrier_mismatch",
                "carrier_combined_match",
                "carrier_combined_mismatch",
            )
        )
        for channel in CHANNELS:
            metric = result.metrics[channel]
            carrier_metric = result.carrier_metrics[channel]
            representation_metric = result.combined_carrier_metrics[channel]
            writer.writerow(
                (
                    channel,
                    metric.match,
                    metric.mismatch,
                    metric.power_ratio,
                    metric.relative_l2,
                    carrier_metric.match,
                    carrier_metric.mismatch,
                    representation_metric.match,
                    representation_metric.mismatch,
                )
            )
    with (output_dir / "modes.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "channel",
                "ell",
                "abs_m",
                "pixels",
                "reference_power",
                "incoherent_power_fraction",
                "match",
                "mismatch",
                "power_ratio",
                "relative_l2",
            )
        )
        channel_power = {
            channel: sum(
                float(np.dot(mode.reference[channel], mode.reference[channel]))
                for mode in result.modes
            )
            for channel in CHANNELS
        }
        for mode in result.modes:
            for channel in CHANNELS:
                metric = result.mode_metrics[(channel, mode.mode)]
                reference_power = float(
                    np.dot(mode.reference[channel], mode.reference[channel])
                )
                writer.writerow(
                    (
                        channel,
                        mode.mode[0],
                        mode.mode[1],
                        mode.keys.size,
                        reference_power,
                        reference_power / channel_power[channel],
                        metric.match,
                        metric.mismatch,
                        metric.power_ratio,
                        metric.relative_l2,
                    )
                )

    time_index = result.keys // (result.shape.nf + 1)
    frequency_layer = result.keys % (result.shape.nf + 1)
    columns = [
        time_index,
        frequency_layer,
        time_index * result.shape.DT,
        frequency_layer * result.shape.DF,
    ]
    header = ["time_index", "frequency_layer", "time_s", "frequency_hz"]
    for channel in CHANNELS:
        columns.extend(
            (
                result.reference[channel],
                result.carrier_tapestry[channel],
                result.tapestry[channel],
            )
        )
        header.extend(
            (
                f"reference_{channel}",
                f"carrier_tapestry_{channel}",
                f"combined_tapestry_{channel}",
            )
        )
    np.savetxt(
        output_dir / "pixels.dat",
        np.column_stack(columns),
        fmt="%.15e",
        header=" ".join(header),
    )
    np.savetxt(output_dir / "response_frequency_nodes.dat", result.response_nodes)


def write_complete_result(result: CompleteTapestryResult, output_dir: Path) -> None:
    """Write only the complete generator's sparse live pixels and summary."""

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("quantity", "value"))
        writer.writerow(("replace_start_hz", result.replace_start_hz))
        writer.writerow(("endpoint_start_s", result.endpoint_start_s))
        writer.writerow(("endpoint_stop_s", result.endpoint_stop_s))
        writer.writerow(("endpoint_sample_count", result.endpoint_sample_count))
        for name, value in result.timings.items():
            writer.writerow((f"timing_{name}_s", value))
        for channel, metric in result.reference_metrics.items():
            writer.writerow((f"validation_{channel}_match", metric.match))
            writer.writerow((f"validation_{channel}_mismatch", metric.mismatch))
            writer.writerow((f"validation_{channel}_power_ratio", metric.power_ratio))
            writer.writerow((f"validation_{channel}_relative_l2", metric.relative_l2))

    for channel, packet in result.channels.items():
        np.savetxt(
            output_dir / f"{channel}_pixels.dat",
            np.column_stack(
                (
                    packet.listn,
                    packet.listm,
                    packet.listn.astype(np.float64) * result.shape.DT,
                    packet.listm.astype(np.float64) * result.shape.DF,
                    packet.values,
                )
            ),
            fmt="%.15e",
            header="time_index frequency_layer time_s frequency_hz value",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", default="default")
    parser.add_argument(
        "--inspiral-diagnostic",
        action="store_true",
        help="run the earlier fixed-frequency validation instead of the complete endpoint hybrid",
    )
    parser.add_argument(
        "--validate-thread",
        action="store_true",
        help="also run the established per-carrier TDI+WDM path and report sparse matches",
    )
    parser.add_argument(
        "--channels",
        default="XYZ",
        help="TDI channels to generate, as any combination of X, Y, and Z",
    )
    parser.add_argument("--maximum-frequency", type=float, default=1.4e-2)
    parser.add_argument("--response-spacing", type=float, default=5.0e-4)
    parser.add_argument(
        "--response-method", choices=("direct", "table"), default="direct"
    )
    parser.add_argument("--m1-solar", type=float, default=2.0e5)
    parser.add_argument("--m2-solar", type=float, default=1.0e5)
    parser.add_argument("--chi1", type=float, default=0.42)
    parser.add_argument("--chi2", type=float, default=0.85)
    parser.add_argument("--phic", type=float, default=0.0)
    parser.add_argument("--tc", type=float, default=3.0e7)
    parser.add_argument("--distance-gpc", type=float, default=1.0)
    parser.add_argument("--theta", type=float, default=2.31)
    parser.add_argument("--lambda", dest="longitude", type=float, default=0.57)
    parser.add_argument("--psi", type=float, default=0.4)
    parser.add_argument("--cosi", type=float, default=0.3)
    parser.add_argument("--nf", type=int, default=4096)
    parser.add_argument("--nt", type=int, default=4096)
    parser.add_argument("--dt", type=float, default=1.875)
    parser.add_argument("--tdi-generation", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--coefficient-backend",
        choices=("auto", "native", "python", "reference"),
        default="python",
    )
    parser.add_argument(
        "--wdm-blend-endpoint",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_WDM_BLEND_ENDPOINT,
    )
    parser.add_argument(
        "--wdm-blend-half-width-layers",
        type=float,
        default=DEFAULT_WDM_BLEND_HALF_WIDTH_LAYERS,
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("phenomthm_tdi_tapestry_full")
    )
    parser.add_argument("--no-write", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
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
    start = time.perf_counter()
    if not args.inspiral_diagnostic:
        requested = tuple(dict.fromkeys(args.channels.upper()))
        result = generate_complete_tapestry(
            source,
            shape,
            modes=args.modes,
            channels=requested,
            coefficient_backend=args.coefficient_backend,
            blend_endpoint=args.wdm_blend_endpoint,
            blend_half_width_layers=args.wdm_blend_half_width_layers,
            validate_thread=args.validate_thread,
            tdi_generation=args.tdi_generation,
        )
        elapsed = time.perf_counter() - start
        print(
            "modes "
            + " ".join(f"{ell}{emm}" for ell, emm in result.intrinsic.modes)
        )
        print("channels " + " ".join(result.channels))
        print(f"replace_start_hz {result.replace_start_hz:.12e}")
        print(
            f"endpoint start_s {result.endpoint_start_s:.9f} "
            f"stop_s {result.endpoint_stop_s:.9f} "
            f"samples {result.endpoint_sample_count}"
        )
        for channel, packet in result.channels.items():
            line = f"channel {channel} active_pixels {packet.values.size}"
            if channel in result.reference_metrics:
                metric = result.reference_metrics[channel]
                line += (
                    f" match {metric.match:.12f} mismatch {metric.mismatch:.12e} "
                    f"power_ratio {metric.power_ratio:.12e} "
                    f"relative_l2 {metric.relative_l2:.12e}"
                )
            print(line)
        print("timing_seconds")
        for name, value in result.timings.items():
            print(f"{name} {value:.6f}")
        print(f"elapsed_with_output_disabled {elapsed:.6f}")
        if not args.no_write:
            write_complete_result(result, args.output_dir)
            print(f"output_dir {args.output_dir}")
        return

    result = generate_tapestry_diagnostic(
        source,
        shape,
        modes=args.modes,
        maximum_frequency_hz=args.maximum_frequency,
        response_spacing_hz=args.response_spacing,
        response_method=args.response_method,
        coefficient_backend=args.coefficient_backend,
        tdi_generation=args.tdi_generation,
    )
    elapsed = time.perf_counter() - start
    print("modes " + " ".join(f"{ell}{emm}" for ell, emm in (item.mode for item in result.modes)))
    print(f"maximum_frequency_hz {result.maximum_frequency_hz:.12e}")
    print(f"union_pixels {result.keys.size}")
    print(f"response_method {result.response_method}")
    print(f"response_nodes {result.response_nodes.size}")
    for channel in CHANNELS:
        metric = result.metrics[channel]
        carrier_metric = result.carrier_metrics[channel]
        representation_metric = result.combined_carrier_metrics[channel]
        print(
            f"channel {channel} reference_combined_match {metric.match:.12f} "
            f"reference_combined_mismatch {metric.mismatch:.12e} "
            f"power_ratio {metric.power_ratio:.12e} "
            f"relative_l2 {metric.relative_l2:.12e} "
            f"reference_carrier_mismatch {carrier_metric.mismatch:.12e} "
            f"carrier_combined_mismatch {representation_metric.mismatch:.12e}"
        )
    print("mode_mismatches")
    channel_power = {
        channel: sum(
            float(np.dot(item.reference[channel], item.reference[channel]))
            for item in result.modes
        )
        for channel in CHANNELS
    }
    for mode in result.modes:
        code = f"{mode.mode[0]}{mode.mode[1]}"
        values = " ".join(
            f"{channel}={result.mode_metrics[(channel, mode.mode)].mismatch:.6e}"
            f"(power={float(np.dot(mode.reference[channel], mode.reference[channel])) / channel_power[channel]:.3e})"
            for channel in CHANNELS
        )
        print(f"mode {code} pixels {mode.keys.size} {values}")
    print("timing_seconds")
    for name, value in result.timings.items():
        print(f"{name} {value:.6f}")
    print(f"elapsed {elapsed:.6f}")
    if not args.no_write:
        write_result(result, args.output_dir)
        print(f"output_dir {args.output_dir}")


if __name__ == "__main__":
    main()
