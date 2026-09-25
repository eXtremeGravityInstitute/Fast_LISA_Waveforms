"""Directed-link TDI-2 Michelson response for complex polarization carriers.

The 16 signed link terms per channel follow the SGS/PyTDI X2 convention.
The default evaluates six directed links at each output epoch and expands
retarded delay words to first order, matching the fast C response.  A source
callback receives sorted barycentric source times and returns analytic
``(hplus, hcross)`` arrays, each either ``(samples,)`` or
``(families, samples)``.  Outputs have shape ``(3, families, samples)``.
"""

# Copyright (C) 2026 Neil Cornish.
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np


_X_TERMS = (
    (1, 3,  1, ()),
    (3, 1,  1, (13,)),
    (1, 2,  1, (13, 31)),
    (2, 1,  1, (13, 31, 12)),
    (1, 2, -1, ()),
    (2, 1, -1, (12,)),
    (1, 3, -1, (12, 21)),
    (3, 1, -1, (12, 21, 13)),
    (1, 2,  1, (13, 31, 12, 21)),
    (2, 1,  1, (13, 31, 12, 21, 12)),
    (1, 3,  1, (13, 31, 12, 21, 12, 21)),
    (3, 1,  1, (13, 31, 12, 21, 12, 21, 13)),
    (1, 3, -1, (12, 21, 13, 31)),
    (3, 1, -1, (12, 21, 13, 31, 13)),
    (1, 2, -1, (12, 21, 13, 31, 13, 31)),
    (2, 1, -1, (12, 21, 13, 31, 13, 31, 12)),
)


def _position(splines: Sequence, times: np.ndarray, derivative: bool = False) -> np.ndarray:
    values = [(s.derivative()(times) if derivative else s(times)) for s in splines]
    return np.stack(values, axis=-1).reshape(len(times), 3, 3)


def _basis(latitude: float, longitude: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    costh = math.sin(latitude)
    sinth = math.sqrt(max(0.0, 1.0-costh*costh))
    cosph, sinph = math.cos(longitude), math.sin(longitude)
    u = np.array((-costh*cosph, -costh*sinph, sinth))
    v = np.array((sinph, -cosph, 0.0))
    propagation = np.array((-sinth*cosph, -sinth*sinph, -costh))
    return propagation, np.outer(v, v)-np.outer(u, u), np.outer(u, v)+np.outer(v, u)


def _reference_links(times: np.ndarray, splines: Sequence, propagation: np.ndarray,
                     eplus: np.ndarray, ecross: np.ndarray) -> dict[tuple[int, int], tuple[np.ndarray, ...]]:
    position = _position(splines, times)
    velocity = _position(splines, times, derivative=True)
    links = {}
    for receiver in range(3):
        for emitter in range(3):
            if receiver == emitter:
                continue
            radius = position[:, receiver]-position[:, emitter]
            r2 = np.einsum("ij,ij->i", radius, radius)
            rv = np.einsum("ij,ij->i", radius, velocity[:, emitter])
            v2 = np.einsum("ij,ij->i", velocity[:, emitter], velocity[:, emitter])
            light_time = (rv+np.sqrt(rv*rv+(1.0-v2)*r2))/(1.0-v2)
            retarded_time = times-light_time
            if (retarded_time[0] < splines[0].x[0] or
                    retarded_time[-1] > splines[0].x[-1]):
                raise ValueError("TDI-2 retarded link extends beyond the constellation grid")
            emitter_position = _position(splines, retarded_time)[:, emitter]
            emitter_velocity = _position(splines, retarded_time, derivative=True)[:, emitter]
            direction = position[:, receiver]-emitter_position
            direction /= np.linalg.norm(direction, axis=1)[:, None]
            numerator = np.einsum("ij,ij->i", direction,
                                  velocity[:, receiver]-emitter_velocity)
            denominator = 1.0-np.einsum("ij,ij->i", direction, emitter_velocity)
            dot_light_time = numerator/denominator
            kdot = direction @ propagation
            plus = -0.5*np.einsum("ni,nj,ij->n", direction, direction, eplus)/(1.0-kdot)
            cross = -0.5*np.einsum("ni,nj,ij->n", direction, direction, ecross)/(1.0-kdot)
            receiver_source = times-position[:, receiver] @ propagation
            emitter_source = retarded_time-emitter_position @ propagation
            receiver_rate = 1.0-velocity[:, receiver] @ propagation
            emitter_rate = (1.0-dot_light_time)*(1.0-emitter_velocity @ propagation)
            links[receiver, emitter] = (light_time, dot_light_time,
                                        receiver_source, emitter_source,
                                        receiver_rate, emitter_rate, plus, cross)
    return links


def monochromatic_tdi2_transfer(
    times: np.ndarray,
    pixel_frequency: np.ndarray,
    pixel_time_columns: np.ndarray,
    latitude: float,
    longitude: float,
    position_splines: Sequence,
    *,
    maximum_derivative: int = 0,
) -> np.ndarray:
    """Return plus/cross X2/Y2/Z2 symbols and their frequency derivatives.

    The output shape is ``(maximum_derivative+1, 3, 2, pixels)``. Each symbol
    multiplies a positive-phase analytic polarization at the pixel output
    time. Geometry and directed links are shared across all frequency pixels
    at a given time, while delay phasors are evaluated at the pixel frequency.
    """
    times = np.asarray(times, dtype=np.float64)
    frequency = np.asarray(pixel_frequency, dtype=np.float64)
    columns = np.asarray(pixel_time_columns, dtype=np.int64)
    if maximum_derivative not in (0, 1, 2):
        raise ValueError("maximum_derivative must be 0, 1, or 2")
    if times.ndim != 1 or np.any(np.diff(times) <= 0.0):
        raise ValueError("times must be a strictly increasing vector")
    if frequency.ndim != 1 or columns.shape != frequency.shape or \
       np.any(columns < 0) or np.any(columns >= times.size):
        raise ValueError("pixel frequencies and time columns must align")
    propagation, eplus, ecross = _basis(latitude, longitude)
    links = _reference_links(times, position_splines, propagation, eplus, ecross)
    local_time = times[columns]
    response = np.zeros((maximum_derivative+1, 3, 2, frequency.size),
                        dtype=np.complex128)
    for channel in range(3):
        for receiver_label, emitter_label, sign, delay_words in _X_TERMS:
            lag = np.zeros(times.size)
            jacobian = np.ones(times.size)
            for label in delay_words:
                receiver = (label//10-1+channel) % 3
                emitter = (label%10-1+channel) % 3
                light_time, derivative = links[receiver, emitter][:2]
                lag += light_time-derivative*lag
                jacobian *= 1.0-derivative
            receiver = (receiver_label-1+channel) % 3
            emitter = (emitter_label-1+channel) % 3
            link = links[receiver, emitter]
            receiver_delay = (link[2]-link[4]*lag)[columns]-local_time
            emitter_delay = (link[3]-link[5]*lag)[columns]-local_time
            receiver_phase = np.exp(2j*math.pi*frequency*receiver_delay)
            emitter_phase = np.exp(2j*math.pi*frequency*emitter_delay)
            weight = sign*jacobian[columns]
            for order in range(maximum_derivative+1):
                factor = (2j*math.pi)**order
                difference = factor*(receiver_delay**order*receiver_phase-
                                     emitter_delay**order*emitter_phase)
                response[order, channel, 0] += weight*link[6][columns]*difference
                response[order, channel, 1] += weight*link[7][columns]*difference
    return response


def complex_tdi2(
    times: np.ndarray,
    latitude: float,
    longitude: float,
    position_splines: Sequence,
    polarizations: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    *,
    chunk_size: int = 256,
) -> np.ndarray:
    """Evaluate fast X2/Y2/Z2 with directed, first-order delay chains.

    Spacecraft positions in ``position_splines`` are in light-seconds and
    ordered as spacecraft-major Cartesian coordinates.  No C dependency is
    required.  The one-link light time uses the positive quadratic root.
    """
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or not len(times) or np.any(np.diff(times) <= 0.0):
        raise ValueError("TDI-2 output times must be strictly increasing")
    if len(position_splines) != 9:
        raise ValueError("expected nine spacecraft position splines")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    propagation, eplus, ecross = _basis(latitude, longitude)
    output = None
    for start in range(0, len(times), chunk_size):
        t = times[start:start+chunk_size]
        links = _reference_links(t, position_splines, propagation, eplus, ecross)
        zeros = np.zeros_like(t)
        ones = np.ones_like(t)
        delay_nodes = {(ch, ()): (zeros, ones) for ch in range(3)}
        term_times = []
        term_coeff = []
        term_channel = []
        for channel in range(3):
            for receiver, emitter, sign, word in _X_TERMS:
                prefix: tuple[int, ...] = ()
                for label in word:
                    next_prefix = prefix+(label,)
                    key = (channel, next_prefix)
                    if key not in delay_nodes:
                        lag, jacobian = delay_nodes[channel, prefix]
                        directed = (((label//10-1+channel) % 3),
                                    ((label%10-1+channel) % 3))
                        light_time, dot_light_time = links[directed][:2]
                        next_lag = lag+light_time-dot_light_time*lag
                        delay_nodes[key] = (next_lag, jacobian*(1.0-dot_light_time))
                    prefix = next_prefix
                lag, jacobian = delay_nodes[channel, prefix]
                directed = (((receiver-1+channel) % 3),
                            ((emitter-1+channel) % 3))
                _, _, ur, ue, dur, due, plus, cross = links[directed]
                term_times.extend((ur-dur*lag, ue-due*lag))
                term_coeff.extend(((sign*jacobian*plus, sign*jacobian*cross),
                                   (-sign*jacobian*plus, -sign*jacobian*cross)))
                term_channel.extend((channel, channel))
        flat_time = np.concatenate(term_times)
        order = np.argsort(flat_time, kind="stable")
        hp_sorted, hc_sorted = polarizations(flat_time[order])
        hp_sorted = np.atleast_2d(np.asarray(hp_sorted, dtype=np.complex128))
        hc_sorted = np.atleast_2d(np.asarray(hc_sorted, dtype=np.complex128))
        if hp_sorted.shape != hc_sorted.shape or hp_sorted.shape[1] != len(flat_time):
            raise ValueError("polarization callback returned inconsistent arrays")
        if output is None:
            output = np.zeros((3, hp_sorted.shape[0], len(times)), dtype=np.complex128)
        elif output.shape[1] != hp_sorted.shape[0]:
            raise ValueError("polarization callback changed its family count")
        hp = np.empty_like(hp_sorted)
        hc = np.empty_like(hc_sorted)
        hp[:, order] = hp_sorted
        hc[:, order] = hc_sorted
        for index, (plus, cross) in enumerate(term_coeff):
            sl = slice(index*len(t), (index+1)*len(t))
            output[term_channel[index], :, start:start+len(t)] += (
                hp[:, sl]*plus+hc[:, sl]*cross
            )
    return output
