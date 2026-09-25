# Sparse Meyer-packet lookup implementation by Neil Cornish.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Reusable local-linear-chirp lookup tables for sparse Meyer-WDM packets."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np

try:
    from numba import njit
except Exception:
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]

        def decorate(func):
            return func

        return decorate

from phenomt_tdi_wdm import PI, WDMShape, phitilde


CHIRPLET_LOOKUP_RATE_MAX = 8.0
CHIRPLET_LOOKUP_RATE_STEP = 0.1
CHIRPLET_LOOKUP_FREQUENCY_STEP = 0.01


@njit(cache=True, inline="always")
def _cubic_lagrange_weight(index: int, fraction: float, order: int) -> float:
    if order == 0:
        if index == 0:
            return -fraction * (fraction - 1.0) * (fraction - 2.0) / 6.0
        if index == 1:
            return (
                (fraction + 1.0)
                * (fraction - 1.0)
                * (fraction - 2.0)
                / 2.0
            )
        if index == 2:
            return -(fraction + 1.0) * fraction * (fraction - 2.0) / 2.0
        return (fraction + 1.0) * fraction * (fraction - 1.0) / 6.0
    if order == 1:
        if index == 0:
            return (-3.0 * fraction * fraction + 6.0 * fraction - 2.0) / 6.0
        if index == 1:
            return (3.0 * fraction * fraction - 4.0 * fraction - 1.0) / 2.0
        if index == 2:
            return (-3.0 * fraction * fraction + 2.0 * fraction + 2.0) / 2.0
        return (3.0 * fraction * fraction - 1.0) / 6.0
    if index == 0:
        return 1.0 - fraction
    if index == 1:
        return 3.0 * fraction - 2.0
    if index == 2:
        return 1.0 - 3.0 * fraction
    return fraction


@njit(cache=True)
def _interpolate_chirplet_table(
    table: np.ndarray,
    frequency_offset: np.ndarray,
    chirp_rate: np.ndarray,
    frequency_minimum: float,
    frequency_step: float,
    chirp_minimum: float,
    chirp_step: float,
    frequency_derivative_order: int,
) -> np.ndarray:
    count = frequency_offset.size
    output = np.empty(count, dtype=np.complex128)
    frequency_count = table.shape[1]
    chirp_count = table.shape[0]
    derivative_scale = frequency_step**frequency_derivative_order
    for point in range(count):
        frequency_position = (
            frequency_offset[point] - frequency_minimum
        ) / frequency_step
        chirp_position = (chirp_rate[point] - chirp_minimum) / chirp_step
        frequency_center = int(math.floor(frequency_position))
        chirp_center = int(math.floor(chirp_position))
        frequency_center = min(max(frequency_center, 1), frequency_count - 3)
        chirp_center = min(max(chirp_center, 1), chirp_count - 3)
        frequency_fraction = frequency_position - frequency_center
        chirp_fraction = chirp_position - chirp_center
        value = 0.0j
        for chirp_index in range(4):
            chirp_weight = _cubic_lagrange_weight(
                chirp_index, chirp_fraction, 0
            )
            row_value = 0.0j
            for frequency_index in range(4):
                frequency_weight = _cubic_lagrange_weight(
                    frequency_index,
                    frequency_fraction,
                    frequency_derivative_order,
                ) / derivative_scale
                row_value += frequency_weight * table[
                    chirp_center + chirp_index - 1,
                    frequency_center + frequency_index - 1,
                ]
            value += chirp_weight * row_value
        output[point] = value
    return output


@dataclass(frozen=True)
class WDMChirpletLookup:
    """Universal WDM overlaps indexed by residual frequency and chirp rate."""

    frequency_offset: np.ndarray
    chirp_rate: np.ndarray
    coefficients: tuple[np.ndarray, ...]
    build_seconds: float
    transform_size: int

    @property
    def entries(self) -> int:
        return int(sum(values.size for values in self.coefficients))

    def interpolate(
        self,
        order: int,
        frequency_offset: np.ndarray,
        chirp_rate: np.ndarray,
        frequency_derivative_order: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        if order < 0 or order >= len(self.coefficients):
            raise ValueError(f"chirplet lookup does not contain moment {order}")
        if frequency_derivative_order < 0 or frequency_derivative_order > 2:
            raise ValueError("lookup frequency derivatives are supported through 2")
        u = np.asarray(frequency_offset, dtype=np.float64)
        v = np.asarray(chirp_rate, dtype=np.float64)
        if u.shape != v.shape:
            raise ValueError("lookup frequency and chirp-rate arrays must match")
        output = np.zeros(u.shape, dtype=np.complex128)
        valid = (
            np.isfinite(u)
            & np.isfinite(v)
            & (u >= self.frequency_offset[0])
            & (u <= self.frequency_offset[-1])
            & (v >= self.chirp_rate[0])
            & (v <= self.chirp_rate[-1])
        )
        if np.any(valid):
            output[valid] = _interpolate_chirplet_table(
                self.coefficients[order],
                np.ascontiguousarray(u[valid]),
                np.ascontiguousarray(v[valid]),
                float(self.frequency_offset[0]),
                float(self.frequency_offset[1] - self.frequency_offset[0]),
                float(self.chirp_rate[0]),
                float(self.chirp_rate[1] - self.chirp_rate[0]),
                frequency_derivative_order,
            )
        return output, valid


_CACHE: dict[tuple[object, ...], WDMChirpletLookup] = {}


def _next_power_of_two(value: int) -> int:
    return 1 << max(0, int(value - 1).bit_length())


def build_wdm_chirplet_lookup(
    shape: WDMShape,
    *,
    maximum_moment_order: int = 1,
    chirp_rate_max: float = CHIRPLET_LOOKUP_RATE_MAX,
    chirp_rate_step: float = CHIRPLET_LOOKUP_RATE_STEP,
    frequency_step: float = CHIRPLET_LOOKUP_FREQUENCY_STEP,
) -> WDMChirpletLookup:
    """Build a symmetric positive/negative-chirp Meyer overlap table."""

    if maximum_moment_order < 0 or maximum_moment_order > 2:
        raise ValueError("chirplet lookup moments are supported through order 2")
    if not math.isfinite(chirp_rate_max) or chirp_rate_max <= 0.0:
        raise ValueError("chirplet lookup rate range must be positive")
    if not math.isfinite(chirp_rate_step) or chirp_rate_step <= 0.0:
        raise ValueError("chirplet lookup rate step must be positive")
    if not math.isfinite(frequency_step) or frequency_step <= 0.0:
        raise ValueError("chirplet lookup frequency step must be positive")

    rate_count = int(math.ceil(chirp_rate_max / chirp_rate_step))
    if rate_count < 2:
        raise ValueError(
            "chirplet lookup needs at least five symmetric chirp-rate rows"
        )
    rate_max = float(rate_count) * chirp_rate_step
    key = (
        shape.nf,
        shape.dt,
        shape.bfrac,
        shape.nx,
        shape.mult,
        maximum_moment_order,
        rate_max,
        chirp_rate_step,
        frequency_step,
    )
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    started = time.perf_counter()
    filter_samples = int(shape.mult * 2 * shape.nf)
    if filter_samples < 8 or filter_samples & (filter_samples - 1):
        raise ValueError("chirplet lookup requires a power-of-two Meyer filter")
    reference_layer = max(2, shape.nf // 16)
    if reference_layer >= shape.nf - 2:
        raise ValueError("WDM grid is too small for a lookup reference layer")

    omega = 2.0 * PI * np.fft.fftfreq(filter_samples, d=shape.dt)
    mother_time = np.fft.fftshift(
        np.fft.ifft(phitilde(omega, shape)) * filter_samples
    ).real
    mother_norm = math.sqrt(float(np.dot(mother_time, mother_time)) * shape.dt)
    if mother_norm <= 0.0:
        raise RuntimeError("zero-norm Meyer window while building chirplet lookup")

    center_omega = float(reference_layer) * shape.DOM
    kernels: list[np.ndarray] = []
    for order in range(maximum_moment_order + 1):
        positive_offset = (omega - center_omega) / shape.DOM
        negative_offset = (-omega - center_omega) / shape.DOM
        spectrum = (
            phitilde(omega - center_omega, shape) * positive_offset**order
            + phitilde(omega + center_omega, shape) * negative_offset**order
        ) / math.sqrt(2.0)
        kernel = np.fft.fftshift(np.fft.ifft(spectrum) * filter_samples).real
        kernels.append(np.asarray(kernel / mother_norm, dtype=np.float64))

    minimum_padding = 1.0 / (frequency_step * shape.DF * shape.Tfilt)
    padding_factor = _next_power_of_two(
        max(1, int(math.ceil(minimum_padding)))
    )
    transform_size = filter_samples * padding_factor
    du = 1.0 / (float(transform_size) * shape.dt * shape.DF)
    frequency_extent = shape.FB / shape.DF + 0.5 * rate_max
    frequency_bins = int(math.ceil(frequency_extent / du))
    frequency_offsets = du * np.arange(
        -frequency_bins, frequency_bins + 1, dtype=np.float64
    )
    reference_bin = int(
        round(reference_layer * shape.DF * transform_size * shape.dt)
    )
    transform_indices = (
        transform_size // 2
        + reference_bin
        + np.arange(-frequency_bins, frequency_bins + 1, dtype=np.int64)
    )
    chirp_rates = chirp_rate_step * np.arange(
        -rate_count, rate_count + 1, dtype=np.float64
    )
    tables = [
        np.empty((chirp_rates.size, frequency_offsets.size), dtype=np.complex128)
        for _ in kernels
    ]
    tau = (
        np.arange(filter_samples, dtype=np.float64) - filter_samples // 2
    ) * shape.dt
    insert = (transform_size - filter_samples) // 2
    padded = np.zeros(transform_size, dtype=np.complex128)
    normalization = math.sqrt(8.0 / 15.0)
    for rate_index, rate in enumerate(chirp_rates):
        fdot = rate * shape.DF / shape.Tfilt
        chirp = np.exp(1j * PI * fdot * tau * tau)
        for order, kernel in enumerate(kernels):
            padded.fill(0.0)
            padded[insert : insert + filter_samples] = kernel * chirp
            transform = np.fft.fftshift(
                np.fft.ifft(np.fft.ifftshift(padded))
            )
            tables[order][rate_index] = (
                normalization
                * float(transform_size)
                * shape.dt
                * transform[transform_indices]
            )

    lookup = WDMChirpletLookup(
        frequency_offsets,
        chirp_rates,
        tuple(tables),
        time.perf_counter() - started,
        transform_size,
    )
    _CACHE[key] = lookup
    return lookup
