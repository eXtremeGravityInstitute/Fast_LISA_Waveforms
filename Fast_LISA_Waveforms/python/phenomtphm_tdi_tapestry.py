#!/usr/bin/env python3
# Python LISA response and WDM port: Copyright (C) 2026 Neil Cornish.
# SPDX-License-Identifier: GPL-3.0-or-later
# Distributed without warranty; see the GNU GPL for details.
# The imported LAL-derived waveform model retains its upstream notices.
"""TDI Tapestry inspiral packets for the precessing TPHM waveform.

The endpoint remains an exact, summed-carrier TDI and FFT in
``phenomtphm_tdi_wdm``. Here the folded intrinsic carriers are heterodyned
separately, but their plus/cross Fourier bins are combined before the Meyer
packet IFFT. A single monochromatic TDI symbol is then applied per pixel.
"""

from __future__ import annotations

import time

import numpy as np

from phenomthm_tdi_wdm import (
    CHANNELS,
    TRIPLES,
    _partition_blocks_to_wdm,
    _response_geometry,
    _tdi_delays_and_coefficients,
)
from phenomt_tdi_wdm import WDMChannel, WDMShape, make_ap_spline
from phenomtphm_tdi_wdm import (
    TPHMIntrinsicGrid,
    TPHMPlanningBands,
    TPHMSourceParams,
    _build_early_blocks,
)


def _polarization_envelope_splines(intrinsic: TPHMIntrinsicGrid):
    real_splines = []
    imag_splines = []
    phase_splines = []
    for carrier in range(len(intrinsic.modes)):
        phase_splines.append(
            make_ap_spline(intrinsic.waveform_time, intrinsic.model_phase[carrier])
        )
        for envelope in (intrinsic.plus_envelope[carrier], intrinsic.cross_envelope[carrier]):
            real_splines.append(make_ap_spline(intrinsic.waveform_time, envelope.real))
            imag_splines.append(make_ap_spline(intrinsic.waveform_time, envelope.imag))
    return real_splines, imag_splines, phase_splines


def _pixel_response(
    source: TPHMSourceParams,
    constellation,
    unique_times: np.ndarray,
    pixel_frequency: np.ndarray,
    time_columns: np.ndarray,
    order: int,
    tdi_generation: int = 1,
):
    if tdi_generation == 2:
        from tdi2_response import monochromatic_tdi2_transfer

        symbols = monochromatic_tdi2_transfer(
            unique_times, pixel_frequency, time_columns,
            source.ecliptic_colatitude, source.ecliptic_longitude,
            constellation[2], maximum_derivative=order,
        )
        return {
            channel: tuple(
                symbols[derivative, index, polarization]
                for derivative in range(order+1)
                for polarization in range(2)
            )
            for index, channel in enumerate(CHANNELS)
        }
    if tdi_generation != 1:
        raise ValueError("tdi_generation must be 1 or 2")
    arm_length, kr, app, apm, acp, acm = _response_geometry(
        unique_times, source, constellation
    )
    local_time = unique_times[time_columns]
    response = {}
    for channel, triple in zip(CHANNELS, TRIPLES):
        delays, coefp, coefc = _tdi_delays_and_coefficients(
            triple, unique_times, arm_length, kr, app, apm, acp, acm
        )
        local_delay = delays[time_columns] - local_time[:, np.newaxis]
        phasor = np.exp(
            2j * np.pi * pixel_frequency[:, np.newaxis] * local_delay
        )
        response_plus = np.sum(coefp[time_columns] * phasor, axis=1)
        response_cross = np.sum(coefc[time_columns] * phasor, axis=1)
        if order:
            derivative = (2j * np.pi * local_delay) * phasor
            derivative_plus = np.sum(coefp[time_columns] * derivative, axis=1)
            derivative_cross = np.sum(coefc[time_columns] * derivative, axis=1)
            response[channel] = (
                response_plus,
                response_cross,
                derivative_plus,
                derivative_cross,
            )
        else:
            response[channel] = (response_plus, response_cross)
    return response


def build_tapestry_early_wdm(
    source: TPHMSourceParams,
    intrinsic: TPHMIntrinsicGrid,
    tdi,
    bands: TPHMPlanningBands,
    constellation,
    requested: tuple[str, ...],
    early_mid: np.ndarray,
    early_size: np.ndarray,
    endpoint_start: float,
    endpoint_rise: float,
    shape: WDMShape,
    *,
    bandwidth_hz: float,
    roll_seconds: float,
    response_order: int,
    tdi_generation: int = 1,
) -> tuple[dict[str, WDMChannel], tuple, int, dict[str, float]]:
    if response_order not in (0, 1):
        raise ValueError("Tapestry response order must be 0 or 1")
    timings: dict[str, float] = {}
    start = time.perf_counter()
    blocks, diagnostics, samples = _build_early_blocks(
        tdi,
        intrinsic,
        bands,
        (0, 1),
        early_mid,
        early_size,
        endpoint_start,
        endpoint_rise,
        shape,
        bandwidth_hz=bandwidth_hz,
        roll_seconds=roll_seconds,
        envelope_splines=_polarization_envelope_splines(intrinsic),
    )
    timings["tapestry_polarization_fft"] = time.perf_counter() - start

    start = time.perf_counter()
    packets = _partition_blocks_to_wdm(
        blocks,
        early_mid,
        early_size,
        shape,
        2,
        quadratures=True,
        frequency_moment_order=response_order,
    )
    polarizations, quadratures = packets[:2]
    moments = packets[2:] if response_order else None
    timings["tapestry_polarization_wdm"] = time.perf_counter() - start
    plus, cross = polarizations
    plus_quadrature, cross_quadrature = quadratures
    for companion in (cross, plus_quadrature, cross_quadrature):
        if not (
            np.array_equal(plus.listn, companion.listn)
            and np.array_equal(plus.listm, companion.listm)
        ):
            raise RuntimeError("TPHM Tapestry polarization packets do not share a pixel grid")

    if plus.listn.size == 0:
        raise RuntimeError("TPHM Tapestry has no early polarization pixels")
    unique_n, time_columns = np.unique(plus.listn, return_inverse=True)
    start = time.perf_counter()
    response = _pixel_response(
        source,
        constellation,
        unique_n.astype(np.float64) * shape.DT,
        plus.listm.astype(np.float64) * shape.DF,
        time_columns,
        response_order,
        tdi_generation=tdi_generation,
    )
    timings["tapestry_response_symbols"] = time.perf_counter() - start

    start = time.perf_counter()
    empty = np.empty(0, dtype=np.float64)
    output: dict[str, WDMChannel] = {}
    for channel in requested:
        if channel not in CHANNELS:
            raise ValueError(f"unknown TDI channel {channel}")
        response_plus, response_cross = response[channel][:2]
        values = (
            response_plus.real * plus.values
            + response_plus.imag * plus_quadrature.values
            + response_cross.real * cross.values
            + response_cross.imag * cross_quadrature.values
        )
        if response_order:
            plus_moment, cross_moment = moments[0]
            plus_quadrature_moment, cross_quadrature_moment = moments[1]
            derivative_plus, derivative_cross = response[channel][2:]
            values += (
                derivative_plus.real * plus_moment.values
                + derivative_plus.imag * plus_quadrature_moment.values
                + derivative_cross.real * cross_moment.values
                + derivative_cross.imag * cross_quadrature_moment.values
            )
        output[channel] = WDMChannel(
            empty,
            empty,
            empty,
            plus.nmid.copy(),
            plus.nsize.copy(),
            plus.listn.copy(),
            plus.listm.copy(),
            np.ascontiguousarray(values, dtype=np.float64),
        )
    timings["tapestry_response_apply"] = time.perf_counter() - start
    return output, tuple(diagnostics), samples, timings


__all__ = ["build_tapestry_early_wdm"]
