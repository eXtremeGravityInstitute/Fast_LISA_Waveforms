#!/usr/bin/env python3
# FEW-driven LISA response and WDM implementation by Neil Cornish.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Polarization-resolved TDI Tapestry prototype for FEW EMRI waveforms.

The established FEW path applies TDI and a partitioned heterodyned WDM
transform to every exact phase carrier.  This prototype transforms the source
plus/cross polarizations before the detector response, using either each
carrier's bandwidth-limited FFT plan or a local chirplet lookup table.  The two
real WDM quadratures of each polarization are accumulated on a common sparse
pixel union, and one complex TDI symbol per polarization is then applied to
produce X, Y, and Z.  The physical plunge and QNM completion retain the
established exact, summed-mode short-FFT treatment.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Iterable

import numpy as np
from few.utils.constants import MTSUN_SI

from few_tdi_wdm import (
    CHIRPLET_LOOKUP_FREQUENCY_STEP,
    CHIRPLET_LOOKUP_RATE_MAX,
    CHIRPLET_LOOKUP_RATE_STEP,
    CHANNEL_TRIPLES,
    EARLY_BLOCK_FFT_ROLL_SECONDS,
    FEWEarlyBlockFFTDiagnostic,
    FEWModeCarrier,
    FEWModeCollection,
    FEWSourceParams,
    TDI_ENDPOINT_DELAY_MARGIN_SECONDS,
    _channel_terms,
    _geometry_at_times,
    build_few_mode_collection,
    build_shared_tdi_time_grid,
    chirplet_lookup_wdm,
    common_plunge_wdm,
    edge_window,
    kerr_qnm_fundamental,
    normalized_match,
    partitioned_band_fft_wdm,
    plunge_partition,
    sparse_wdm_from_global_rfft,
    tdi_source_event_output_bounds,
    tdi_track_frequency,
    validate_modes,
)
from phenomt_tdi_wdm import (
    PI,
    TDIGrid,
    WDMChannel,
    WDMShape,
    build_constellation_splines,
    combine_sparse_wdm,
    extract_ap,
    merge_pixel_plans,
    wdm_pixels,
)


POLARIZATION_KEY = "P"
DEFAULT_SKY_COSTHETA = 0.45
DEFAULT_SKY_LONGITUDE = 2.20
DEFAULT_POLARIZATION = 0.70
DEFAULT_TDI_MAXIMUM_STEP = 8.0e4
DEFAULT_RESPONSE_FREQUENCY_ORDER = 1


@dataclass(frozen=True)
class FEWTapestryResult:
    source: FEWSourceParams
    shape: WDMShape
    collection: FEWModeCollection
    channels: dict[str, WDMChannel]
    response_order_channels: dict[int, dict[str, WDMChannel]]
    plus: WDMChannel
    plus_quadrature: WDMChannel
    cross: WDMChannel
    cross_quadrature: WDMChannel
    partition_start: float
    partition_rise: float
    timings: dict[str, float]
    metrics: dict[str, float]
    block_diagnostics: tuple[FEWEarlyBlockFFTDiagnostic, ...]


def _empty_channel(shape: WDMShape) -> WDMChannel:
    return WDMChannel(
        np.empty(0),
        np.empty(0),
        np.empty(0),
        np.full(shape.nf, -1, dtype=np.int64),
        np.zeros(shape.nf, dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.empty(0),
    )


def _synthetic_polarization_grid(
    carrier: FEWModeCarrier,
    times: np.ndarray,
    polarization: str,
) -> TDIGrid:
    """Represent one intrinsic polarization in the existing block-FFT AP API."""

    phase = carrier.phase(times)
    plus, cross = carrier.polarization_coefficients(times)
    coefficient = plus if polarization == "plus" else cross
    cosine = np.cos(phase)
    sine = np.sin(phase)
    value = coefficient.real * cosine - coefficient.imag * sine
    quadrature = -coefficient.real * sine - coefficient.imag * cosine
    signed_amplitude, phase_offset = extract_ap(value, quadrature, phase)
    return TDIGrid(
        np.asarray(times, dtype=np.float64),
        np.asarray(times, dtype=np.float64),
        phase,
        phase.copy(),
        {POLARIZATION_KEY: signed_amplitude},
        {POLARIZATION_KEY: phase_offset},
    )


def _polarization_support_plan(
    collection: FEWModeCollection,
    times: np.ndarray,
    shape: WDMShape,
    time_weight,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plan the union support without constructing any carrier FFT blocks."""

    nmid = np.full(shape.nf, -1, dtype=np.int64)
    nsize = np.zeros(shape.nf, dtype=np.int64)
    key_parts: list[np.ndarray] = []
    sparse_weight = np.asarray(time_weight(times), dtype=np.float64)
    for carrier in collection.carriers:
        for polarization in ("plus", "cross"):
            grid = _synthetic_polarization_grid(carrier, times, polarization)
            frequency, _ = tdi_track_frequency(carrier, grid, POLARIZATION_KEY)
            weighted_amplitude = grid.amplitude[POLARIZATION_KEY] * sparse_weight
            amplitude_scale = max(
                float(np.max(np.abs(weighted_amplitude))), np.finfo(float).tiny
            )
            active = (
                np.isfinite(frequency)
                & (frequency > 0.0)
                & (sparse_weight > 1.0e-10)
                & (np.abs(weighted_amplitude) > 1.0e-13 * amplitude_scale)
            )
            if np.count_nonzero(active) < 4:
                continue
            local_mid, local_size = wdm_pixels(
                times[active], np.abs(frequency[active]), shape
            )
            for layer in np.flatnonzero(local_size > 0):
                packet_size = int(local_size[layer])
                output_time = (
                    np.arange(packet_size, dtype=np.int64)
                    + int(local_mid[layer])
                    - packet_size // 2
                )
                output_time = output_time[
                    (output_time >= 0) & (output_time < shape.nt)
                ]
                key_parts.append(output_time * (shape.nf + 1) + int(layer))
            merge_pixel_plans(nmid, nsize, local_mid, local_size, shape)
    keys = (
        np.unique(np.concatenate(key_parts))
        if key_parts
        else np.empty(0, dtype=np.int64)
    )
    return nmid, nsize, keys


def _restrict_channel_keys(
    channel: WDMChannel,
    requested_keys: np.ndarray,
    shape: WDMShape,
) -> WDMChannel:
    channel_keys = _keys(channel, shape)
    order = np.argsort(channel_keys)
    channel_keys = channel_keys[order]
    location = np.searchsorted(channel_keys, requested_keys)
    if np.any(location >= channel_keys.size) or not np.array_equal(
        channel_keys[location], requested_keys
    ):
        raise RuntimeError("global polarization packet misses requested support")
    values = channel.values[order][location]
    return WDMChannel(
        channel.freq,
        channel.phase,
        channel.amplitude,
        channel.nmid.copy(),
        channel.nsize.copy(),
        requested_keys // (shape.nf + 1),
        requested_keys % (shape.nf + 1),
        values,
    )


def _global_polarization_packets(
    collection: FEWModeCollection,
    shape: WDMShape,
    nmid: np.ndarray,
    nsize: np.ndarray,
    time_weight,
    taper_width: float,
    *,
    live_stop_time: float | None = None,
    dense_chunk: int = 65536,
) -> tuple[WDMChannel, WDMChannel, WDMChannel, WDMChannel, dict[str, float]]:
    """Build summed intrinsic polarization packets from two global real FFTs."""

    if dense_chunk < 1:
        raise ValueError("dense_chunk must be positive")
    timing: dict[str, float] = {}
    plus_time = np.zeros(shape.n, dtype=np.float64)
    cross_time = np.zeros(shape.n, dtype=np.float64)
    # The early partition is identically zero after this point.  Avoid asking
    # FEW for a final chunk of source samples that cannot enter the result.
    live_stop = shape.n
    if live_stop_time is not None:
        live_stop = min(
            live_stop,
            max(0, int(math.ceil(float(live_stop_time) / shape.dt)) + 1),
        )
    start_time = time.perf_counter()
    for start in range(0, live_stop, dense_chunk):
        stop = min(start + dense_chunk, live_stop)
        times = shape.dt * np.arange(start, stop, dtype=np.float64)
        plus_modes, cross_modes = collection.polarization_values(times)
        weight = np.asarray(time_weight(times), dtype=np.float64)
        weight *= edge_window(times, shape.Tobs, taper_width)
        plus_time[start:stop] = np.sum(plus_modes, axis=1) * weight
        cross_time[start:stop] = np.sum(cross_modes, axis=1) * weight
    timing["global_dense_reconstruction"] = time.perf_counter() - start_time

    start_time = time.perf_counter()
    plus_spectrum = np.fft.rfft(plus_time) * (2.0 * shape.dt)
    timing["global_plus_rfft"] = time.perf_counter() - start_time
    plus_quadrature_out: list[WDMChannel] = []
    start_time = time.perf_counter()
    plus = sparse_wdm_from_global_rfft(
        plus_spectrum, nmid, nsize, shape, plus_quadrature_out
    )
    timing["global_plus_sparse_wdm"] = time.perf_counter() - start_time
    del plus_spectrum, plus_time

    start_time = time.perf_counter()
    cross_spectrum = np.fft.rfft(cross_time) * (2.0 * shape.dt)
    timing["global_cross_rfft"] = time.perf_counter() - start_time
    cross_quadrature_out: list[WDMChannel] = []
    start_time = time.perf_counter()
    cross = sparse_wdm_from_global_rfft(
        cross_spectrum, nmid, nsize, shape, cross_quadrature_out
    )
    timing["global_cross_sparse_wdm"] = time.perf_counter() - start_time
    if len(plus_quadrature_out) != 1 or len(cross_quadrature_out) != 1:
        raise RuntimeError("global FFT did not return both polarization quadratures")
    return (
        plus,
        plus_quadrature_out[0],
        cross,
        cross_quadrature_out[0],
        timing,
    )


def _sum_sparse_parts(parts: list[WDMChannel], shape: WDMShape) -> WDMChannel:
    if not parts:
        return _empty_channel(shape)
    listn, listm, values = combine_sparse_wdm(
        shape,
        [part.listn for part in parts],
        [part.listm for part in parts],
        [part.values for part in parts],
    )
    nmid = np.full(shape.nf, -1, dtype=np.int64)
    nsize = np.zeros(shape.nf, dtype=np.int64)
    for part in parts:
        merge_pixel_plans(nmid, nsize, part.nmid, part.nsize, shape)
    return WDMChannel(
        np.empty(0), np.empty(0), np.empty(0), nmid, nsize, listn, listm, values
    )


def _keys(channel: WDMChannel, shape: WDMShape) -> np.ndarray:
    return channel.listn * (shape.nf + 1) + channel.listm


def _values_on_union(
    channel: WDMChannel, union: np.ndarray, shape: WDMShape
) -> np.ndarray:
    out = np.zeros(union.size, dtype=np.float64)
    if channel.values.size:
        keys = _keys(channel, shape)
        location = np.searchsorted(union, keys)
        if np.any(location >= union.size) or not np.array_equal(union[location], keys):
            raise RuntimeError("polarization packet lies outside the Tapestry union")
        np.add.at(out, location, channel.values)
    return out


def _direct_sparse_response(
    source: FEWSourceParams,
    times: np.ndarray,
    pixel_frequency: np.ndarray,
    pixel_time_columns: np.ndarray,
    channels: Iterable[str],
    l_splines,
    p_splines,
    v_splines,
    frequency_derivative_output: (
        dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] | None
    ) = None,
    tdi_generation: int = 1,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Evaluate complex plus/cross TDI symbols at sparse pixel centers.

    Requested frequency derivatives are accumulated analytically from the
    same delay phasors.  The derivative output is indexed first by order and
    then by channel, while the ordinary return value remains unchanged.
    """

    derivative_orders = tuple(
        sorted(int(order) for order in (frequency_derivative_output or {}))
    )
    if any(order < 1 for order in derivative_orders):
        raise ValueError("response frequency derivative orders must be positive")

    if tdi_generation == 2:
        from tdi2_response import monochromatic_tdi2_transfer

        response_grid = monochromatic_tdi2_transfer(
            times, pixel_frequency, pixel_time_columns,
            math.asin(float(source.ecliptic_costheta)),
            source.ecliptic_longitude, p_splines,
            maximum_derivative=max(derivative_orders, default=0),
        )
        response = {
            channel: (response_grid[0, "XYZ".index(channel), 0],
                      response_grid[0, "XYZ".index(channel), 1])
            for channel in channels
        }
        if frequency_derivative_output is not None:
            for order in derivative_orders:
                for channel in channels:
                    frequency_derivative_output[order][channel] = (
                        response_grid[order, "XYZ".index(channel), 0],
                        response_grid[order, "XYZ".index(channel), 1],
                    )
        return response
    if tdi_generation != 1:
        raise ValueError("tdi_generation must be 1 or 2")

    larm, kr, app, apm, acp, acm = _geometry_at_times(
        times, source, l_splines, p_splines, v_splines
    )
    local_time = times[pixel_time_columns]
    response: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for channel in channels:
        response_plus = np.zeros(pixel_frequency.size, dtype=np.complex128)
        response_cross = np.zeros_like(response_plus)
        derivative_plus = {
            order: np.zeros_like(response_plus) for order in derivative_orders
        }
        derivative_cross = {
            order: np.zeros_like(response_cross) for order in derivative_orders
        }
        for delayed, coefficient_plus, coefficient_cross in _channel_terms(
            channel, times, larm, kr, app, apm, acp, acm
        ):
            delay = delayed[pixel_time_columns] - local_time
            phasor = np.exp(1j * 2.0 * PI * pixel_frequency * delay)
            response_plus += coefficient_plus[pixel_time_columns] * phasor
            response_cross += coefficient_cross[pixel_time_columns] * phasor
            for order in derivative_orders:
                factor = (1j * 2.0 * PI * delay) ** order
                derivative_plus[order] += (
                    coefficient_plus[pixel_time_columns] * phasor * factor
                )
                derivative_cross[order] += (
                    coefficient_cross[pixel_time_columns] * phasor * factor
                )
        response[channel] = (response_plus, response_cross)
        if frequency_derivative_output is not None:
            for order in derivative_orders:
                frequency_derivative_output[order][channel] = (
                    derivative_plus[order],
                    derivative_cross[order],
                )
    return response


def _sparse_metrics(
    reference: WDMChannel, candidate: WDMChannel, shape: WDMShape
) -> dict[str, float]:
    reference_keys = _keys(reference, shape)
    candidate_keys = _keys(candidate, shape)
    union = np.union1d(reference_keys, candidate_keys)
    left = _values_on_union(reference, union, shape)
    right = _values_on_union(candidate, union, shape)
    left_power = float(np.dot(left, left))
    right_power = float(np.dot(right, right))
    return {
        "match": normalized_match(left, right),
        "power_ratio": right_power / max(left_power, np.finfo(float).tiny),
        "relative_l2": float(np.linalg.norm(right - left))
        / max(math.sqrt(left_power), np.finfo(float).tiny),
    }


def _comparison_union(
    reference: WDMChannel,
    candidate: WDMChannel,
    shape: WDMShape,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    union = np.union1d(_keys(reference, shape), _keys(candidate, shape))
    return (
        union,
        _values_on_union(reference, union, shape),
        _values_on_union(candidate, union, shape),
    )


def _aggregate_validation(
    index: np.ndarray,
    size: int,
    reference: np.ndarray,
    candidate: np.ndarray,
) -> np.ndarray:
    count = np.bincount(index, minlength=size).astype(np.int64)
    reference_power = np.bincount(
        index, weights=reference * reference, minlength=size
    )
    candidate_power = np.bincount(
        index, weights=candidate * candidate, minlength=size
    )
    residual_power = np.bincount(
        index, weights=(candidate - reference) ** 2, minlength=size
    )
    cross = np.bincount(index, weights=reference * candidate, minlength=size)
    denominator = np.sqrt(reference_power * candidate_power)
    match = np.full(size, np.nan)
    valid = denominator > 0.0
    match[valid] = np.abs(cross[valid]) / denominator[valid]
    relative_l2 = np.full(size, np.nan)
    valid_reference = reference_power > 0.0
    relative_l2[valid_reference] = np.sqrt(
        residual_power[valid_reference] / reference_power[valid_reference]
    )
    total_reference = max(float(np.sum(reference_power)), np.finfo(float).tiny)
    total_residual = max(float(np.sum(residual_power)), np.finfo(float).tiny)
    return np.column_stack(
        (
            np.arange(size),
            count,
            reference_power,
            candidate_power,
            residual_power,
            reference_power / total_reference,
            residual_power / total_residual,
            match,
            1.0 - match,
            relative_l2,
        )
    )


def write_validation_diagnostics(
    reference: WDMChannel,
    candidate: WDMChannel,
    shape: WDMShape,
    channel: str,
    output: Path,
) -> None:
    """Write and plot the Tapestry residual resolved in WDM time/frequency."""

    output.mkdir(parents=True, exist_ok=True)
    union, left, right = _comparison_union(reference, candidate, shape)
    time_index = union // (shape.nf + 1)
    frequency_layer = union % (shape.nf + 1)
    by_time = _aggregate_validation(time_index, shape.nt, left, right)
    by_layer = _aggregate_validation(
        frequency_layer, shape.nf + 1, left, right
    )
    by_time[:, 0] *= shape.DT
    by_layer[:, 0] *= shape.DF
    np.savetxt(
        output / f"validation_{channel}_by_time.dat",
        by_time[by_time[:, 1] > 0.0],
        fmt="%.15e",
        header=(
            "time_s pixels reference_power candidate_power residual_power "
            "reference_power_fraction residual_power_fraction match mismatch "
            "relative_l2"
        ),
    )
    np.savetxt(
        output / f"validation_{channel}_by_frequency.dat",
        by_layer[by_layer[:, 1] > 0.0],
        fmt="%.15e",
        header=(
            "frequency_hz pixels reference_power candidate_power residual_power "
            "reference_power_fraction residual_power_fraction match mismatch "
            "relative_l2"
        ),
    )

    residual = right - left
    total_reference = max(float(np.dot(left, left)), np.finfo(float).tiny)
    order = np.argsort(residual * residual)[::-1]
    keep = order[: min(10000, order.size)]
    np.savetxt(
        output / f"validation_{channel}_largest_residual_pixels.dat",
        np.column_stack(
            (
                time_index[keep],
                frequency_layer[keep],
                time_index[keep].astype(np.float64) * shape.DT,
                frequency_layer[keep].astype(np.float64) * shape.DF,
                left[keep],
                right[keep],
                residual[keep],
                residual[keep] ** 2 / total_reference,
                left[keep] ** 2 / total_reference,
            )
        ),
        fmt=["%d", "%d", "%.15e", "%.15e", "%.15e", "%.15e", "%.15e", "%.15e", "%.15e"],
        header=(
            "time_index frequency_layer time_s frequency_hz reference tapestry "
            "residual residual_power_over_total_reference "
            "pixel_reference_power_fraction"
        ),
    )

    import matplotlib.pyplot as plt

    live_time = by_time[:, 1] > 0.0
    live_layer = by_layer[:, 1] > 0.0
    time_days = by_time[live_time, 0] / 86400.0
    frequency_mhz = by_layer[live_layer, 0] * 1.0e3
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.5), constrained_layout=True)
    axes[0, 0].semilogy(
        frequency_mhz,
        np.maximum(by_layer[live_layer, 5], 1.0e-20),
        label="reference power",
    )
    axes[0, 0].semilogy(
        frequency_mhz,
        np.maximum(by_layer[live_layer, 6], 1.0e-20),
        label="residual power",
    )
    axes[0, 0].set_xlabel("frequency (mHz)")
    axes[0, 0].set_ylabel("fraction per layer")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].semilogy(
        frequency_mhz,
        np.maximum(by_layer[live_layer, 9], 1.0e-12),
    )
    axes[0, 1].set_xlabel("frequency (mHz)")
    axes[0, 1].set_ylabel("relative L2 by layer")
    axes[1, 0].semilogy(
        time_days,
        np.maximum(by_time[live_time, 5], 1.0e-20),
        label="reference power",
    )
    axes[1, 0].semilogy(
        time_days,
        np.maximum(by_time[live_time, 6], 1.0e-20),
        label="residual power",
    )
    axes[1, 0].set_xlabel("time (days)")
    axes[1, 0].set_ylabel("fraction per time pixel")
    axes[1, 0].legend(frameon=False)
    axes[1, 1].semilogy(
        time_days,
        np.maximum(by_time[live_time, 9], 1.0e-12),
    )
    axes[1, 1].set_xlabel("time (days)")
    axes[1, 1].set_ylabel("relative L2 by time pixel")
    figure.savefig(output / f"validation_{channel}_marginals.png", dpi=180)
    figure.savefig(output / f"validation_{channel}_marginals.pdf")
    plt.close(figure)

    residual_fraction = residual * residual / total_reference
    positive = residual_fraction > 0.0
    if np.any(positive):
        log_residual = np.log10(residual_fraction[positive])
        floor = max(float(np.max(log_residual)) - 8.0, -20.0)
        figure, axis = plt.subplots(figsize=(10.5, 5.5), constrained_layout=True)
        scatter = axis.scatter(
            time_index[positive].astype(np.float64) * shape.DT / 86400.0,
            frequency_layer[positive].astype(np.float64) * shape.DF * 1.0e3,
            c=np.maximum(log_residual, floor),
            s=1.0,
            linewidths=0.0,
            cmap="magma",
            rasterized=True,
        )
        axis.set_xlabel("time (days)")
        axis.set_ylabel("frequency (mHz)")
        colorbar = figure.colorbar(scatter, ax=axis)
        colorbar.set_label("log10 pixel residual power / total reference power")
        figure.savefig(output / f"validation_{channel}_residual_map.png", dpi=180)
        figure.savefig(output / f"validation_{channel}_residual_map.pdf")
        plt.close(figure)


def write_response_order_diagnostics(
    reference: WDMChannel,
    candidates: dict[int, WDMChannel],
    shape: WDMShape,
    channel: str,
    output: Path,
) -> None:
    """Compare local-symbol frequency orders layer by layer."""

    output.mkdir(parents=True, exist_ok=True)
    order_values = sorted(candidates)
    aggregates: dict[int, np.ndarray] = {}
    for order in order_values:
        union, left, right = _comparison_union(reference, candidates[order], shape)
        frequency_layer = union % (shape.nf + 1)
        aggregates[order] = _aggregate_validation(
            frequency_layer, shape.nf + 1, left, right
        )

    live = np.zeros(shape.nf + 1, dtype=bool)
    for aggregate in aggregates.values():
        live |= aggregate[:, 1] > 0.0
    frequencies = np.arange(shape.nf + 1, dtype=np.float64) * shape.DF
    columns = [frequencies[live]]
    header = ["frequency_hz"]
    for order in order_values:
        aggregate = aggregates[order]
        columns.extend((aggregate[live, 9], aggregate[live, 4]))
        header.extend(
            (
                f"relative_l2_order_{order}",
                f"residual_power_order_{order}",
            )
        )
    np.savetxt(
        output / f"validation_{channel}_response_orders_by_frequency.dat",
        np.column_stack(columns),
        fmt="%.15e",
        header=" ".join(header),
    )

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(9.0, 7.0), constrained_layout=True)
    frequency_mhz = frequencies[live] * 1.0e3
    for order in order_values:
        aggregate = aggregates[order]
        axes[0].semilogy(
            frequency_mhz,
            np.maximum(aggregate[live, 9], 1.0e-12),
            label=f"response order {order}",
        )
        total_reference = max(
            float(np.sum(aggregate[:, 2])), np.finfo(float).tiny
        )
        axes[1].semilogy(
            frequency_mhz,
            np.maximum(aggregate[live, 4] / total_reference, 1.0e-20),
            label=f"response order {order}",
        )
    baseline_residual = aggregates[order_values[0]][:, 4]
    significant = baseline_residual > float(np.max(baseline_residual)) * 1.0e-12
    if np.any(significant):
        frequency_limit = 1.05 * frequencies[significant][-1] * 1.0e3
        axes[0].set_xlim(0.0, frequency_limit)
        axes[1].set_xlim(0.0, frequency_limit)
    axes[0].set_ylabel("relative L2 by layer")
    axes[0].legend(frameon=False)
    axes[1].set_xlabel("frequency (mHz)")
    axes[1].set_ylabel("residual power / total reference power")
    axes[1].legend(frameon=False)
    figure.savefig(
        output / f"validation_{channel}_response_orders_by_frequency.png",
        dpi=180,
    )
    figure.savefig(output / f"validation_{channel}_response_orders_by_frequency.pdf")
    plt.close(figure)


def _sparse_band_metrics(
    reference: WDMChannel,
    candidate: WDMChannel,
    shape: WDMShape,
    edges_hz: tuple[float, ...] = (0.0, 1.0e-3, 5.0e-3, 2.0e-2, math.inf),
) -> list[tuple[float, float, dict[str, float], float]]:
    """Return unweighted validation metrics and reference-power fractions."""

    reference_keys = _keys(reference, shape)
    candidate_keys = _keys(candidate, shape)
    union = np.union1d(reference_keys, candidate_keys)
    left = _values_on_union(reference, union, shape)
    right = _values_on_union(candidate, union, shape)
    frequency = (union % (shape.nf + 1)).astype(np.float64) * shape.DF
    total_power = max(float(np.dot(left, left)), np.finfo(float).tiny)
    output: list[tuple[float, float, dict[str, float], float]] = []
    for lower, upper in zip(edges_hz[:-1], edges_hz[1:]):
        select = (frequency >= lower) & (frequency < upper)
        if not np.any(select):
            continue
        left_band = left[select]
        right_band = right[select]
        left_power = float(np.dot(left_band, left_band))
        right_power = float(np.dot(right_band, right_band))
        metrics = {
            "match": normalized_match(left_band, right_band),
            "power_ratio": right_power / max(left_power, np.finfo(float).tiny),
            "relative_l2": float(np.linalg.norm(right_band - left_band))
            / max(math.sqrt(left_power), np.finfo(float).tiny),
        }
        output.append((lower, upper, metrics, left_power / total_power))
    return output


def generate_few_tapestry(
    source: FEWSourceParams,
    shape: WDMShape,
    *,
    channels: tuple[str, ...] = ("X", "Y", "Z"),
    mode_selection_threshold: float = 1.0e-3,
    taper_pixels: float = 8.0,
    tdi_maximum_step: float = DEFAULT_TDI_MAXIMUM_STEP,
    endpoint_model: str = "qnm",
    plunge_pre_m: float = 200.0,
    plunge_rise_m: float = 50.0,
    qnm_transition_m: float = 10.0,
    qnm_efolds: float = 8.0,
    qnm_fade_efolds: float = 2.0,
    endpoint_spectral_power_tolerance: float = 1.0e-10,
    polarization_method: str = "block-fft",
    global_dense_chunk: int = 65536,
    early_fft_bandwidth_hz: float = 1.0e-3,
    early_fft_roll_seconds: float = EARLY_BLOCK_FFT_ROLL_SECONDS,
    early_fft_tile_strategy: str = "dyadic",
    lookup_chirp_rate_max: float = CHIRPLET_LOOKUP_RATE_MAX,
    lookup_chirp_rate_step: float = CHIRPLET_LOOKUP_RATE_STEP,
    lookup_frequency_step: float = CHIRPLET_LOOKUP_FREQUENCY_STEP,
    lookup_amplitude_order: int = 2,
    response_frequency_order: int = DEFAULT_RESPONSE_FREQUENCY_ORDER,
    tdi_generation: int = 1,
) -> FEWTapestryResult:
    requested = tuple(dict.fromkeys(channel.upper() for channel in channels))
    if not requested or any(channel not in CHANNEL_TRIPLES for channel in requested):
        raise ValueError("channels must be selected from X, Y, and Z")
    if shape.nt < 2 or shape.nt & (shape.nt - 1):
        raise ValueError("WDM nt must be a power of two")
    if shape.nf < 2 or shape.nf & (shape.nf - 1):
        raise ValueError("WDM nf must be a power of two")
    if polarization_method not in {"block-fft", "lookup", "global-fft"}:
        raise ValueError(
            "polarization_method must be 'block-fft', 'lookup', or 'global-fft'"
        )
    if response_frequency_order not in {0, 1, 2}:
        raise ValueError("response_frequency_order must be 0, 1, or 2")
    if tdi_generation not in (1, 2):
        raise ValueError("tdi_generation must be 1 or 2")
    if response_frequency_order and polarization_method == "global-fft":
        raise ValueError(
            "response frequency moments require block-fft or lookup polarizations"
        )

    total_start = time.perf_counter()
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

    stage = time.perf_counter()
    collection = build_few_mode_collection(
        source,
        shape,
        mode_selection="threshold",
        mode_selection_threshold=mode_selection_threshold,
        endpoint_padding=endpoint_padding,
        endpoint_model=endpoint_model,
        qnm_transition_m=qnm_transition_m,
        qnm_efolds=qnm_efolds,
        qnm_fade_efolds=qnm_fade_efolds,
    )
    timings["few_collection"] = time.perf_counter() - stage

    stage = time.perf_counter()
    _, l_splines, p_splines, v_splines = build_constellation_splines(shape)
    timings["constellation"] = time.perf_counter() - stage

    endpoint_bounds = [
        tdi_source_event_output_bounds(
            collection.plunge_time,
            collection.source,
            channel,
            l_splines,
            p_splines,
            v_splines,
            tdi_generation=tdi_generation,
        )
        for channel in requested
    ]
    response_endpoint_start = min(bounds[0] for bounds in endpoint_bounds)
    endpoint_partition_minimum = 2.0 * float(shape.mult) * shape.DT
    plunge_pre = max(plunge_pre_m * collection.mass_seconds, endpoint_partition_minimum)
    plunge_rise = max(
        plunge_rise_m * collection.mass_seconds, endpoint_partition_minimum
    )
    partition_start = max(0.0, response_endpoint_start - plunge_pre)
    partition_rise = min(
        plunge_rise, max(response_endpoint_start - partition_start, 0.0)
    )
    partition_stop = partition_start + partition_rise

    stage = time.perf_counter()
    planner_diagnostics: dict[str, float] = {}
    planning_times = build_shared_tdi_time_grid(
        collection,
        shape,
        p_splines,
        maximum_step=tdi_maximum_step,
        taper_width=taper_width,
        planning_stop=partition_stop,
        diagnostics=planner_diagnostics,
        l_splines=l_splines,
        channels=requested,
        tdi_generation=tdi_generation,
    )
    if partition_rise > 0.0:
        transition = np.linspace(partition_start, partition_stop, 65)
        planning_times = np.unique(
            np.concatenate((planning_times, transition, np.array([partition_stop])))
        )
    timings["shared_grid_plan"] = time.perf_counter() - stage

    def early_weight(values: np.ndarray) -> np.ndarray:
        return 1.0 - plunge_partition(values, partition_start, partition_rise)

    diagnostics: list[FEWEarlyBlockFFTDiagnostic] = []
    fft_samples = 0
    polarization_moments: dict[
        int, dict[str, tuple[WDMChannel, WDMChannel]]
    ] = {}
    if polarization_method in {"block-fft", "lookup"}:
        plus_parts: list[WDMChannel] = []
        plus_quadrature_parts: list[WDMChannel] = []
        cross_parts: list[WDMChannel] = []
        cross_quadrature_parts: list[WDMChannel] = []
        moment_parts: dict[
            int,
            dict[str, tuple[list[WDMChannel], list[WDMChannel]]],
        ] = {
            order: {
                "plus": ([], []),
                "cross": ([], []),
            }
            for order in range(1, response_frequency_order + 1)
        }
        block_plan_seconds = 0.0
        block_build_seconds = 0.0
        block_packet_seconds = 0.0
        lookup_table_seconds = 0.0
        lookup_evaluate_seconds = 0.0
        lookup_pixel_evaluations = 0.0
        lookup_table_entries = 0.0
        lookup_table_rate_limit = 0.0
        lookup_rate_min = math.inf
        lookup_rate_max = -math.inf
        stage = time.perf_counter()
        for carrier_index, carrier in enumerate(collection.carriers):
            for polarization, parts, quadrature_parts in (
                ("plus", plus_parts, plus_quadrature_parts),
                ("cross", cross_parts, cross_quadrature_parts),
            ):
                grid = _synthetic_polarization_grid(
                    carrier, planning_times, polarization
                )
                track_frequency, _ = tdi_track_frequency(
                    carrier, grid, POLARIZATION_KEY
                )
                quadrature: list[WDMChannel] = []
                frequency_moments = {
                    order: []
                    for order in range(1, response_frequency_order + 1)
                }
                block_timing: dict[str, float] = {}
                try:
                    if polarization_method == "block-fft":
                        packet, samples, block_diagnostics = partitioned_band_fft_wdm(
                            carrier,
                            grid,
                            POLARIZATION_KEY,
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
                            timing_diagnostics=block_timing,
                            quadrature_output=quadrature,
                            frequency_moment_output=frequency_moments,
                        )
                    else:
                        packet, samples, block_diagnostics = chirplet_lookup_wdm(
                            carrier,
                            grid,
                            POLARIZATION_KEY,
                            shape,
                            taper_width,
                            early_weight,
                            track_frequency=track_frequency,
                            maximum_moment_order=response_frequency_order,
                            chirp_rate_max=lookup_chirp_rate_max,
                            chirp_rate_step=lookup_chirp_rate_step,
                            frequency_step=lookup_frequency_step,
                            amplitude_order=lookup_amplitude_order,
                            timing_diagnostics=block_timing,
                            quadrature_output=quadrature,
                            frequency_moment_output=frequency_moments,
                        )
                except RuntimeError as exc:
                    if "insufficient" in str(exc):
                        continue
                    raise
                if len(quadrature) != 1:
                    raise RuntimeError(
                        "block FFT did not return one analytic quadrature"
                    )
                parts.append(packet)
                quadrature_parts.append(quadrature[0])
                for order, moment_output in frequency_moments.items():
                    if len(moment_output) != 2:
                        raise RuntimeError(
                            f"block FFT did not return frequency moment {order}"
                        )
                    value_parts, quadrature_moment_parts = moment_parts[order][
                        polarization
                    ]
                    value_parts.append(moment_output[0])
                    quadrature_moment_parts.append(moment_output[1])
                diagnostics.extend(block_diagnostics)
                fft_samples += samples
                block_plan_seconds += block_timing.get("plan", 0.0)
                block_build_seconds += block_timing.get("fft_build", 0.0)
                block_packet_seconds += block_timing.get("packet_wdm", 0.0)
                lookup_table_seconds += block_timing.get("table_access", 0.0)
                lookup_evaluate_seconds += block_timing.get("evaluate", 0.0)
                lookup_pixel_evaluations += block_timing.get(
                    "pixel_evaluations", 0.0
                )
                lookup_table_entries = max(
                    lookup_table_entries, block_timing.get("table_entries", 0.0)
                )
                lookup_table_rate_limit = max(
                    lookup_table_rate_limit,
                    block_timing.get("table_chirp_rate_limit", 0.0),
                )
                lookup_rate_min = min(
                    lookup_rate_min, block_timing.get("chirp_rate_min", math.inf)
                )
                lookup_rate_max = max(
                    lookup_rate_max, block_timing.get("chirp_rate_max", -math.inf)
                )
        timings["polarization_transform_total"] = time.perf_counter() - stage
        timings["polarization_block_plan"] = block_plan_seconds
        timings["polarization_block_build"] = block_build_seconds
        timings["polarization_packet_wdm"] = block_packet_seconds
        if polarization_method == "lookup":
            timings["polarization_lookup_table_access"] = lookup_table_seconds
            timings["polarization_lookup_evaluate"] = lookup_evaluate_seconds

        stage = time.perf_counter()
        plus = _sum_sparse_parts(plus_parts, shape)
        plus_quadrature = _sum_sparse_parts(plus_quadrature_parts, shape)
        cross = _sum_sparse_parts(cross_parts, shape)
        cross_quadrature = _sum_sparse_parts(cross_quadrature_parts, shape)
        for order, by_polarization in moment_parts.items():
            polarization_moments[order] = {}
            for polarization, (value_parts, quadrature_parts) in (
                by_polarization.items()
            ):
                polarization_moments[order][polarization] = (
                    _sum_sparse_parts(value_parts, shape),
                    _sum_sparse_parts(quadrature_parts, shape),
                )
        timings["polarization_accumulation"] = time.perf_counter() - stage
    else:
        stage = time.perf_counter()
        global_mid, global_size, global_keys = _polarization_support_plan(
            collection, planning_times, shape, early_weight
        )
        timings["polarization_global_support_plan"] = time.perf_counter() - stage
        stage = time.perf_counter()
        (
            plus,
            plus_quadrature,
            cross,
            cross_quadrature,
            global_timing,
        ) = _global_polarization_packets(
            collection,
            shape,
            global_mid,
            global_size,
            early_weight,
            taper_width,
            live_stop_time=partition_stop,
            dense_chunk=global_dense_chunk,
        )
        plus = _restrict_channel_keys(plus, global_keys, shape)
        plus_quadrature = _restrict_channel_keys(
            plus_quadrature, global_keys, shape
        )
        cross = _restrict_channel_keys(cross, global_keys, shape)
        cross_quadrature = _restrict_channel_keys(
            cross_quadrature, global_keys, shape
        )
        timings["polarization_transform_total"] = time.perf_counter() - stage
        timings.update(global_timing)
        fft_samples = 2 * shape.n
    nonempty = [
        item
        for item in (plus, plus_quadrature, cross, cross_quadrature)
        if item.values.size
    ]
    if not nonempty:
        raise RuntimeError("no early polarization packets were generated")
    union_keys = np.unique(np.concatenate([_keys(item, shape) for item in nonempty]))
    plus_values = _values_on_union(plus, union_keys, shape)
    plus_quadrature_values = _values_on_union(plus_quadrature, union_keys, shape)
    cross_values = _values_on_union(cross, union_keys, shape)
    cross_quadrature_values = _values_on_union(
        cross_quadrature, union_keys, shape
    )
    moment_values: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for order, by_polarization in polarization_moments.items():
        moment_values[order] = {}
        for polarization, (value_packet, quadrature_packet) in (
            by_polarization.items()
        ):
            moment_values[order][polarization] = (
                _values_on_union(value_packet, union_keys, shape),
                _values_on_union(quadrature_packet, union_keys, shape),
            )

    time_index = union_keys // (shape.nf + 1)
    frequency_layer = union_keys % (shape.nf + 1)
    unique_time_index, pixel_time_columns = np.unique(
        time_index, return_inverse=True
    )
    response_times = unique_time_index.astype(np.float64) * shape.DT
    pixel_frequency = frequency_layer.astype(np.float64) * shape.DF
    stage = time.perf_counter()
    response_derivatives: dict[
        int, dict[str, tuple[np.ndarray, np.ndarray]]
    ] = {
        order: {} for order in range(1, response_frequency_order + 1)
    }
    response = _direct_sparse_response(
        source,
        response_times,
        pixel_frequency,
        pixel_time_columns,
        requested,
        l_splines,
        p_splines,
        v_splines,
        frequency_derivative_output=response_derivatives,
        tdi_generation=tdi_generation,
    )
    timings["shared_sparse_response"] = time.perf_counter() - stage

    early_channels_by_order: dict[int, dict[str, WDMChannel]] = {
        order: {} for order in range(response_frequency_order + 1)
    }
    early_nmid = np.full(shape.nf, -1, dtype=np.int64)
    early_nsize = np.zeros(shape.nf, dtype=np.int64)
    for packet in (plus, cross):
        merge_pixel_plans(early_nmid, early_nsize, packet.nmid, packet.nsize, shape)
    stage = time.perf_counter()
    for channel in requested:
        response_plus, response_cross = response[channel]
        cumulative_values = (
            response_plus.real * plus_values
            + response_plus.imag * plus_quadrature_values
            + response_cross.real * cross_values
            + response_cross.imag * cross_quadrature_values
        )
        for order in range(response_frequency_order + 1):
            if order > 0:
                derivative_plus, derivative_cross = response_derivatives[order][
                    channel
                ]
                plus_moment, plus_quadrature_moment = moment_values[order]["plus"]
                cross_moment, cross_quadrature_moment = moment_values[order][
                    "cross"
                ]
                scale = shape.DF**order / float(math.factorial(order))
                cumulative_values = cumulative_values + scale * (
                    derivative_plus.real * plus_moment
                    + derivative_plus.imag * plus_quadrature_moment
                    + derivative_cross.real * cross_moment
                    + derivative_cross.imag * cross_quadrature_moment
                )
            early_channels_by_order[order][channel] = WDMChannel(
                np.empty(0),
                np.empty(0),
                np.empty(0),
                early_nmid.copy(),
                early_nsize.copy(),
                time_index.copy(),
                frequency_layer.copy(),
                np.asarray(cumulative_values, dtype=np.float64).copy(),
            )
    timings["shared_response_apply"] = time.perf_counter() - stage

    source_probe_start = max(
        0.0,
        collection.plunge_time - plunge_pre - TDI_ENDPOINT_DELAY_MARGIN_SECONDS,
    )
    plunge_probe = np.linspace(source_probe_start, collection.plunge_time, 257)
    plunge_frequency = float(
        np.max(np.abs(collection.frequency_matrix(plunge_probe)))
    )
    maximum_endpoint_frequency = max(
        plunge_frequency, collection.maximum_qnm_frequency
    )
    outputs_by_order: dict[int, dict[str, WDMChannel]] = {
        order: {} for order in range(response_frequency_order + 1)
    }
    endpoint_pixels = 0
    stage = time.perf_counter()
    for channel in requested:
        endpoint, _, _, _ = common_plunge_wdm(
            collection,
            channel,
            shape,
            l_splines,
            p_splines,
            v_splines,
            partition_start,
            partition_rise,
            taper_width,
            maximum_endpoint_frequency,
            spectral_power_tolerance=endpoint_spectral_power_tolerance,
            tdi_generation=tdi_generation,
        )
        nmid = early_nmid.copy()
        nsize = early_nsize.copy()
        merge_pixel_plans(nmid, nsize, endpoint.nmid, endpoint.nsize, shape)
        for order in range(response_frequency_order + 1):
            early_channel = early_channels_by_order[order][channel]
            listn, listm, values = combine_sparse_wdm(
                shape,
                [early_channel.listn, endpoint.listn],
                [early_channel.listm, endpoint.listm],
                [early_channel.values, endpoint.values],
            )
            outputs_by_order[order][channel] = WDMChannel(
                np.empty(0),
                np.empty(0),
                np.empty(0),
                nmid.copy(),
                nsize.copy(),
                listn,
                listm,
                values,
            )
        endpoint_pixels = max(endpoint_pixels, endpoint.values.size)
    timings["exact_endpoint_wdm"] = time.perf_counter() - stage
    timings["total"] = time.perf_counter() - total_start
    outputs = outputs_by_order[response_frequency_order]

    metrics = {
        "global_polarization_fft": float(polarization_method == "global-fft"),
        "lookup_polarization_transform": float(polarization_method == "lookup"),
        "selected_modes": float(len(collection.selected_modes)),
        "exact_phase_carriers": float(collection.size),
        "few_sparse_points": float(collection.sparse_time.size),
        "planning_grid_points": float(planning_times.size),
        "polarization_union_pixels": float(union_keys.size),
        "polarization_block_count": float(len(diagnostics)),
        "polarization_fft_samples": float(fft_samples),
        "endpoint_pixels_per_channel": float(endpoint_pixels),
        "partition_start_seconds": partition_start,
        "partition_rise_seconds": partition_rise,
        "sky_costheta": source.ecliptic_costheta,
        "sky_longitude": source.ecliptic_longitude,
        "response_frequency_order": float(response_frequency_order),
        "tdi_generation": float(tdi_generation),
    }
    if polarization_method == "lookup":
        metrics.update(
            {
                "lookup_table_entries": lookup_table_entries,
                "lookup_table_chirp_rate_limit": lookup_table_rate_limit,
                "lookup_pixel_evaluations": lookup_pixel_evaluations,
                "lookup_chirp_rate_min": (
                    lookup_rate_min if math.isfinite(lookup_rate_min) else 0.0
                ),
                "lookup_chirp_rate_max": (
                    lookup_rate_max if math.isfinite(lookup_rate_max) else 0.0
                ),
            }
        )
    for channel, packet in outputs.items():
        metrics[f"active_pixels_{channel}"] = float(packet.values.size)
    return FEWTapestryResult(
        source,
        shape,
        collection,
        outputs,
        outputs_by_order,
        plus,
        plus_quadrature,
        cross,
        cross_quadrature,
        partition_start,
        partition_rise,
        timings,
        metrics,
        tuple(diagnostics),
    )


def write_result(result: FEWTapestryResult, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("quantity", "value"))
        for key, value in result.metrics.items():
            writer.writerow((key, value))
        for key, value in result.timings.items():
            writer.writerow((f"seconds_{key}", value))
    for channel, packet in result.channels.items():
        np.savetxt(
            output / f"{channel}_pixels.dat",
            np.column_stack(
                (
                    packet.listn,
                    packet.listm,
                    packet.listn.astype(np.float64) * result.shape.DT,
                    packet.listm.astype(np.float64) * result.shape.DF,
                    packet.values,
                )
            ),
            fmt=["%d", "%d", "%.15e", "%.15e", "%.15e"],
            header="time_index frequency_layer time_s frequency_hz value",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mass", type=float, default=1.0e6)
    parser.add_argument("--mu", type=float, default=50.0)
    parser.add_argument("--spin", type=float, default=0.5)
    parser.add_argument("--p0", type=float, default=12.510272236947417)
    parser.add_argument("--e0", type=float, default=0.4)
    parser.add_argument("--sky-costheta", type=float, default=DEFAULT_SKY_COSTHETA)
    parser.add_argument("--sky-longitude", type=float, default=DEFAULT_SKY_LONGITUDE)
    parser.add_argument("--polarization", type=float, default=DEFAULT_POLARIZATION)
    parser.add_argument("--channels", default="XYZ")
    parser.add_argument("--nt", type=int, default=4096)
    parser.add_argument("--nf", type=int, default=4096)
    parser.add_argument("--dt", type=float, default=1.875)
    parser.add_argument("--tdi-generation", type=int, choices=(1, 2), default=1)
    parser.add_argument("--mode-selection-threshold", type=float, default=1.0e-3)
    parser.add_argument("--taper-pixels", type=float, default=8.0)
    parser.add_argument(
        "--tdi-maximum-step", type=float, default=DEFAULT_TDI_MAXIMUM_STEP
    )
    parser.add_argument(
        "--polarization-method",
        choices=("block-fft", "lookup", "global-fft"),
        default="block-fft",
        help="early intrinsic polarization transform used before sparse TDI",
    )
    parser.add_argument(
        "--global-dense-chunk",
        type=int,
        default=65536,
        help="number of full-cadence samples evaluated per FEW global-FFT chunk",
    )
    parser.add_argument("--early-fft-bandwidth-hz", type=float, default=1.0e-3)
    parser.add_argument(
        "--early-fft-roll-seconds",
        type=float,
        default=EARLY_BLOCK_FFT_ROLL_SECONDS,
    )
    parser.add_argument(
        "--early-fft-tile-strategy", choices=("dyadic", "maximal"), default="dyadic"
    )
    parser.add_argument(
        "--lookup-chirp-rate-max",
        type=float,
        default=CHIRPLET_LOOKUP_RATE_MAX,
        help="symmetric limit on fdot*Tfilt/DF in the chirplet table",
    )
    parser.add_argument(
        "--lookup-chirp-rate-step",
        type=float,
        default=CHIRPLET_LOOKUP_RATE_STEP,
    )
    parser.add_argument(
        "--lookup-frequency-step",
        type=float,
        default=CHIRPLET_LOOKUP_FREQUENCY_STEP,
        help="target table spacing in units of DF",
    )
    parser.add_argument(
        "--lookup-amplitude-order",
        type=int,
        choices=(0, 1, 2),
        default=2,
        help="local Taylor order retained for the slowly varying amplitude",
    )
    parser.add_argument(
        "--response-frequency-order",
        type=int,
        choices=(0, 1, 2),
        default=DEFAULT_RESPONSE_FREQUENCY_ORDER,
        help=(
            "include frequency-gradient response terms through this order; "
            "zero recovers the pixel-center local-symbol approximation"
        ),
    )
    parser.add_argument(
        "--validate-thread",
        action="store_true",
        help="also generate the established per-carrier block path per channel",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=0,
        help="discard this many complete calls before reporting the measured call",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("few_tdi_tapestry"))
    parser.add_argument("--no-write", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.warmup_runs < 0:
        raise SystemExit("--warmup-runs must be non-negative")
    channels = tuple(dict.fromkeys(args.channels.upper()))
    source = FEWSourceParams(
        mass_solar=args.mass,
        mu_solar=args.mu,
        spin=args.spin,
        p0=args.p0,
        eccentricity0=args.e0,
        ecliptic_costheta=args.sky_costheta,
        ecliptic_longitude=args.sky_longitude,
        polarization=args.polarization,
    )
    shape = WDMShape(nf=args.nf, nt=args.nt, dt=args.dt)
    def generate() -> FEWTapestryResult:
        return generate_few_tapestry(
            source,
            shape,
            channels=channels,
            mode_selection_threshold=args.mode_selection_threshold,
            taper_pixels=args.taper_pixels,
            tdi_maximum_step=args.tdi_maximum_step,
            polarization_method=args.polarization_method,
            global_dense_chunk=args.global_dense_chunk,
            early_fft_bandwidth_hz=args.early_fft_bandwidth_hz,
            early_fft_roll_seconds=args.early_fft_roll_seconds,
            early_fft_tile_strategy=args.early_fft_tile_strategy,
            lookup_chirp_rate_max=args.lookup_chirp_rate_max,
            lookup_chirp_rate_step=args.lookup_chirp_rate_step,
            lookup_frequency_step=args.lookup_frequency_step,
            lookup_amplitude_order=args.lookup_amplitude_order,
            response_frequency_order=args.response_frequency_order,
            tdi_generation=args.tdi_generation,
        )

    for _ in range(args.warmup_runs):
        generate()
    result = generate()
    print(
        f"source mass {source.mass_solar:.9e} mu {source.mu_solar:.9e} "
        f"spin {source.spin:.6f} e0 {source.eccentricity0:.6f} "
        f"sky_costheta {source.ecliptic_costheta:.6f} "
        f"sky_longitude {source.ecliptic_longitude:.6f}"
    )
    print(
        f"wdm_grid nt {shape.nt} nf {shape.nf} dt {shape.dt:.9e} "
        f"Tobs {shape.Tobs:.9e}"
    )
    for key, value in result.metrics.items():
        print(f"{key} {value:.12e}")
    for key, value in result.timings.items():
        print(f"seconds_{key} {value:.6f}")

    if args.validate_thread:
        validation_start = time.perf_counter()
        for channel in channels:
            start = time.perf_counter()
            reference = validate_modes(
                source,
                shape,
                channel=channel,
                mode_selection="threshold",
                mode_selection_threshold=args.mode_selection_threshold,
                taper_pixels=args.taper_pixels,
                tdi_maximum_step=args.tdi_maximum_step,
                fast_method="block-fft",
                early_fft_bandwidth_hz=args.early_fft_bandwidth_hz,
                early_fft_roll_seconds=args.early_fft_roll_seconds,
                early_fft_tile_strategy=args.early_fft_tile_strategy,
                direct_validation=False,
                tdi_generation=args.tdi_generation,
            )
            comparison = _sparse_metrics(
                reference.fast_wdm, result.channels[channel], shape
            )
            result.metrics[f"thread_{channel}_match"] = comparison["match"]
            result.metrics[f"thread_{channel}_mismatch"] = (
                1.0 - comparison["match"]
            )
            result.metrics[f"thread_{channel}_power_ratio"] = comparison[
                "power_ratio"
            ]
            result.metrics[f"thread_{channel}_relative_l2"] = comparison[
                "relative_l2"
            ]
            print(f"thread_{channel}_seconds {time.perf_counter() - start:.6f}")
            print(f"thread_{channel}_match {comparison['match']:.12f}")
            print(f"thread_{channel}_mismatch {1.0 - comparison['match']:.12e}")
            print(
                f"thread_{channel}_power_ratio "
                f"{comparison['power_ratio']:.12e}"
            )
            print(
                f"thread_{channel}_relative_l2 "
                f"{comparison['relative_l2']:.12e}"
            )
            for order, candidates in result.response_order_channels.items():
                order_comparison = _sparse_metrics(
                    reference.fast_wdm, candidates[channel], shape
                )
                prefix = f"thread_{channel}_response_order_{order}"
                result.metrics[f"{prefix}_match"] = order_comparison["match"]
                result.metrics[f"{prefix}_mismatch"] = (
                    1.0 - order_comparison["match"]
                )
                result.metrics[f"{prefix}_power_ratio"] = order_comparison[
                    "power_ratio"
                ]
                result.metrics[f"{prefix}_relative_l2"] = order_comparison[
                    "relative_l2"
                ]
                print(f"{prefix}_match {order_comparison['match']:.12f}")
                print(
                    f"{prefix}_mismatch "
                    f"{1.0 - order_comparison['match']:.12e}"
                )
                print(
                    f"{prefix}_power_ratio "
                    f"{order_comparison['power_ratio']:.12e}"
                )
                print(
                    f"{prefix}_relative_l2 "
                    f"{order_comparison['relative_l2']:.12e}"
                )
                for lower, upper, band, power_fraction in _sparse_band_metrics(
                    reference.fast_wdm, candidates[channel], shape
                ):
                    upper_label = "inf" if math.isinf(upper) else f"{upper:.3e}"
                    print(
                        f"{prefix}_band {lower:.3e} {upper_label} "
                        f"match {band['match']:.12f} "
                        f"mismatch {1.0 - band['match']:.12e} "
                        f"power_fraction {power_fraction:.12e}"
                    )
            if not args.no_write:
                write_validation_diagnostics(
                    reference.fast_wdm,
                    result.channels[channel],
                    shape,
                    channel,
                    args.output_dir,
                )
                if len(result.response_order_channels) > 1:
                    write_response_order_diagnostics(
                        reference.fast_wdm,
                        {
                            order: candidates[channel]
                            for order, candidates in (
                                result.response_order_channels.items()
                            )
                        },
                        shape,
                        channel,
                        args.output_dir,
                    )
        print(
            f"thread_validation_seconds "
            f"{time.perf_counter() - validation_start:.6f}"
        )

    if not args.no_write:
        write_result(result, args.output_dir)
        print(f"output_dir {args.output_dir}")


if __name__ == "__main__":
    main()
