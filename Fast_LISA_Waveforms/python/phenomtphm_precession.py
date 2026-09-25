#!/usr/bin/env python3
# Based on LALSimulation IMRPhenomTPHM: Copyright (C) 2020 Hector Estelles.
# LAL-free Python port and changes: Copyright (C) 2026 Neil Cornish.
# SPDX-License-Identifier: GPL-2.0-or-later
# Distributed without warranty; see the GNU GPL for details.
"""Numerical-precession driver for the LAL-free IMRPhenomTPHM port.

The implementation follows ``IMRPhenomTPHM_Precession.c``.  The ten evolved
variables are ``(Lhat, S1, S2, gamma)`` in total-mass units.  The aligned-spin
IMRPhenomT 2,2 frequency supplies the orbital velocity, while SciPy's DOP853
integrator replaces GSL's eighth-order Prince-Dormand driver.

Only the physically consistent branch is implemented here.  The deliberately
cadence-dependent conventions retained by the C code's LAL-compatibility mode
remain a validation feature of the C implementation, not the Python default.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from scipy.integrate import solve_ivp

from phenomt22 import _omega22_jit, final_spin_2017, qnm_fring22
import phenomthm_fits as fits

try:
    from numba import njit
except ImportError:  # pragma: no cover - NumPy fallback is still supported.
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda function: function


_DEFAULT_ATOL = 1.0e-11
_DEFAULT_RTOL = 1.0e-11
_DEFAULT_INITIAL_STEP = 1.0


def _spin_tuple(spin: Sequence[float], name: str) -> tuple[float, float, float]:
    values = tuple(float(value) for value in spin)
    if len(values) != 3:
        raise ValueError(f"{name} must contain three components")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite")
    if np.linalg.norm(values) > 1.0 + 1.0e-12:
        raise ValueError(f"{name} must have magnitude no greater than one")
    return values


@dataclass(frozen=True)
class PrecessionConfig:
    """Spin vectors and numerical controls at the reference source time.

    ``tau_ref`` is measured in total-mass units relative to the 2,2 amplitude
    peak.  The spin vectors are dimensionless and are specified in the L0
    frame, where ``Lhat=(0,0,1)`` at ``tau_ref``.
    """

    chi1: tuple[float, float, float]
    chi2: tuple[float, float, float]
    tau_ref: float
    absolute_tolerance: float = _DEFAULT_ATOL
    relative_tolerance: float = _DEFAULT_RTOL
    initial_step: float = _DEFAULT_INITIAL_STEP

    @classmethod
    def from_spins(
        cls,
        chi1: Sequence[float],
        chi2: Sequence[float],
        tau_ref: float,
        *,
        absolute_tolerance: float = _DEFAULT_ATOL,
        relative_tolerance: float = _DEFAULT_RTOL,
        initial_step: float = _DEFAULT_INITIAL_STEP,
    ) -> "PrecessionConfig":
        return cls(
            _spin_tuple(chi1, "chi1"),
            _spin_tuple(chi2, "chi2"),
            float(tau_ref),
            float(absolute_tolerance),
            float(relative_tolerance),
            float(initial_step),
        )

    def validate(self) -> None:
        _spin_tuple(self.chi1, "chi1")
        _spin_tuple(self.chi2, "chi2")
        if not math.isfinite(self.tau_ref) or self.tau_ref > 0.0:
            raise ValueError("tau_ref must be finite and no later than the 2,2 peak")
        if self.absolute_tolerance <= 0.0 or self.relative_tolerance <= 0.0:
            raise ValueError("ODE tolerances must be positive")
        if self.initial_step <= 0.0:
            raise ValueError("initial_step must be positive")


@dataclass(frozen=True)
class PrecessionSummary:
    final_spin: float
    aligned_final_spin_at_peak: float
    ringdown_alpha_slope: float
    alpha_reference: float
    gamma_reference: float
    j_frame_x: np.ndarray
    j_frame_y: np.ndarray
    j_frame_z: np.ndarray


@dataclass(frozen=True)
class PrecessionTrajectory:
    tau: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    gamma: np.ndarray
    summary: PrecessionSummary


def _spin_dot_3pn(x: float) -> float:
    return 1.5 - x - 0.5 * x * x


def _spin_dot_5pn(x: float) -> float:
    x2 = x * x
    return 9.0 / 8.0 - x / 2.0 + 7.0 * x2 / 12.0 - 7.0 * x2 * x / 6.0 - x2 * x2 / 24.0


def _spin_dot_7pn(x: float) -> float:
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x5 = x4 * x
    x6 = x3 * x3
    return x6 / 48.0 - 3.0 * x5 / 8.0 - 39.0 * x4 / 16.0 - 23.0 * x3 / 6.0 + 181.0 * x2 / 16.0 - 51.0 * x / 8.0 + 27.0 / 16.0


def _orbital_l_magnitude(eta: float, velocity: float) -> float:
    v2 = velocity * velocity
    l1 = 1.5 + eta / 6.0
    l2 = 27.0 / 8.0 - 19.0 * eta / 8.0 + eta * eta / 24.0
    return eta / velocity * (1.0 + l1 * v2 + l2 * v2 * v2)


@njit(cache=True)
def _precession_rhs(
    tau: float,
    state: np.ndarray,
    eta: float,
    carrier_coeffs: np.ndarray,
    spin_coefficients: np.ndarray,
    frame_x: np.ndarray,
    frame_y: np.ndarray,
    frame_z: np.ndarray,
) -> np.ndarray:
    omega22 = _omega22_jit(tau, eta, carrier_coeffs)
    if not math.isfinite(omega22) or omega22 <= 0.0:
        raise FloatingPointError("non-positive carrier frequency in precession ODE")
    velocity = (0.5 * omega22) ** (1.0 / 3.0)
    v2 = velocity * velocity
    v3 = v2 * velocity
    v5 = v3 * v2
    v6 = v3 * v3
    v7 = v5 * v2
    v9 = v6 * v3
    s13, s15, s17, s23, s25, s27 = spin_coefficients
    factor1 = s13 * v5 + s15 * v7 + s17 * v9
    factor2 = s23 * v5 + s25 * v7 + s27 * v9
    dot1 = state[0] * state[3] + state[1] * state[4] + state[2] * state[5]
    dot2 = state[0] * state[6] + state[1] * state[7] + state[2] * state[8]
    orbital_l = eta / velocity * (
        1.0 + (1.5 + eta / 6.0) * v2
        + (27.0 / 8.0 - 19.0 * eta / 8.0 + eta * eta / 24.0) * v2 * v2
    )
    derivative = np.empty(10, dtype=np.float64)
    for axis in range(3):
        following = (axis + 1) % 3
        preceding = (axis + 2) % 3
        lx_s1 = state[following] * state[preceding + 3] - state[preceding] * state[following + 3]
        lx_s2 = state[following] * state[preceding + 6] - state[preceding] * state[following + 6]
        s1_x_s2 = state[following + 3] * state[preceding + 6] - state[preceding + 3] * state[following + 6]
        derivative[axis + 3] = factor1 * lx_s1 + v6 * (-0.5 * s1_x_s2 - 1.5 * dot2 * lx_s1)
        derivative[axis + 6] = factor2 * lx_s2 + v6 * (0.5 * s1_x_s2 - 1.5 * dot1 * lx_s2)
    projection = 0.0
    for axis in range(3):
        derivative[axis] = -(derivative[axis + 3] + derivative[axis + 6]) / orbital_l
        projection += state[axis] * derivative[axis]
    for axis in range(3):
        derivative[axis] -= projection * state[axis]
    lx = ly = lz = dlx = dly = 0.0
    for axis in range(3):
        lx += state[axis] * frame_x[axis]
        ly += state[axis] * frame_y[axis]
        lz += state[axis] * frame_z[axis]
        dlx += derivative[axis] * frame_x[axis]
        dly += derivative[axis] * frame_y[axis]
    transverse2 = lx * lx + ly * ly
    derivative[9] = -lz * (lx * dly - ly * dlx) / transverse2 if transverse2 > 1.0e-24 else 0.0
    return derivative


def _initial_state(carrier, config: PrecessionConfig) -> np.ndarray:
    m1m = carrier.m1 / carrier.total_mass
    m2m = carrier.m2 / carrier.total_mass
    state = np.empty(10, dtype=np.float64)
    state[:3] = (0.0, 0.0, 1.0)
    state[3:6] = m1m * m1m * np.asarray(config.chi1)
    state[6:9] = m2m * m2m * np.asarray(config.chi2)
    state[9] = 0.0
    return state


def _build_j_frame(carrier, config: PrecessionConfig, state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    velocity = math.cbrt(0.5 * carrier.mode22.omega22(config.tau_ref))
    lmag = _orbital_l_magnitude(carrier.eta, velocity)
    jz = state[3:6] + state[6:9]
    jz = jz.copy()
    jz[2] += lmag
    jz /= np.linalg.norm(jz)

    projection = jz[2]
    jx = np.array(
        (-projection * jz[0], -projection * jz[1], 1.0 - projection * jz[2]),
        dtype=np.float64,
    )
    norm = np.linalg.norm(jx)
    if not math.isfinite(norm) or norm <= np.finfo(np.float64).tiny:
        jx[:] = (1.0, 0.0, 0.0)
    else:
        jx /= norm
    jy = np.cross(jz, jx)
    jy /= np.linalg.norm(jy)
    return jx, jy, jz


def _state_angles(state: np.ndarray, frame: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[float, float]:
    lhat = np.asarray(state[:3], dtype=np.float64)
    lhat = lhat / np.linalg.norm(lhat)
    jx, jy, jz = frame
    return math.atan2(float(np.dot(lhat, jy)), float(np.dot(lhat, jx))), float(np.clip(np.dot(lhat, jz), -1.0, 1.0))


def _unwrap_near(angle: float, reference: float) -> float:
    return reference + math.remainder(angle - reference, 2.0 * math.pi)


def _ringdown_alpha_slope(final_spin: float, final_mass: float) -> float:
    slope = 2.0 * math.pi * (
        qnm_fring22(final_spin) - fits.evaluate_QNMfit_fring21(final_spin)
    ) / final_mass
    return -slope if final_spin < 0.0 else slope


def _final_spin(carrier, peak_state: np.ndarray) -> tuple[float, float]:
    lhat = peak_state[:3] / np.linalg.norm(peak_state[:3])
    spin1 = peak_state[3:6]
    spin2 = peak_state[6:9]
    m1m = carrier.m1 / carrier.total_mass
    m2m = carrier.m2 / carrier.total_mass
    norm1 = m1m * m1m
    norm2 = m2m * m2m
    s1_l = float(np.dot(spin1, lhat))
    s2_l = float(np.dot(spin2, lhat))
    spin_perp = spin1 - s1_l * lhat + spin2 - s2_l * lhat
    aligned = final_spin_2017(carrier.eta, s1_l / norm1, s2_l / norm2)
    final = math.copysign(math.hypot(float(np.linalg.norm(spin_perp)), aligned), aligned)
    return float(np.clip(final, -1.0 + 1.0e-12, 1.0 - 1.0e-12)), aligned


class PrecessionEvolution:
    """Retain the DOP853 continuous solution and accepted integration mesh.

    A backward solve is added/extended only when needed. No spline is fitted
    to accepted endpoints; an instance belongs to one fixed source model.
    """

    def __init__(self, carrier, config: PrecessionConfig):
        self.carrier, self.config = carrier, config
        self.forward = self.backward = None

    def _integrate(self, stop, rhs, reference_state):
        config = self.config
        if stop == config.tau_ref:
            return None
        cached = self.backward if stop < config.tau_ref else self.forward
        if cached is not None and min(cached.t[0], cached.t[-1]) <= stop <= max(cached.t[0], cached.t[-1]):
            return cached
        solution = solve_ivp(
            rhs, (config.tau_ref, stop), reference_state, method="DOP853",
            rtol=config.relative_tolerance, atol=config.absolute_tolerance,
            first_step=min(config.initial_step, abs(stop-config.tau_ref)), dense_output=True)
        if not solution.success or solution.sol is None:
            raise RuntimeError(f"precession integration failed: {solution.message}")
        if stop < config.tau_ref:
            self.backward = solution
        else:
            self.forward = solution
        return solution

    def evaluate(self, tau: Sequence[float] | np.ndarray) -> PrecessionTrajectory:
        return evolve_euler_angles(self.carrier, self.config, tau, _evolution=self)

    def integration_nodes(self, start: float, stop: float = 0.0) -> np.ndarray:
        """Accepted spin nodes plus bounds in [start, stop], in mass units.

        Spin integration ends at zero; post-peak continuation is analytic.
        """
        if not math.isfinite(start) or not math.isfinite(stop) or not start <= stop <= 0:
            raise ValueError("spin mesh requires finite start <= stop <= 0")
        self.evaluate(np.unique(np.asarray([start, stop])))
        nodes = [np.asarray([start, stop, self.config.tau_ref])]
        for solution in (self.backward, self.forward):
            if solution is not None:
                nodes.append(solution.t)
        nodes = np.unique(np.concatenate(nodes))
        return nodes[(nodes >= start) & (nodes <= stop)]


def evolve_euler_angles(carrier, config: PrecessionConfig, tau: Sequence[float] | np.ndarray,
                       *, _evolution: PrecessionEvolution | None = None) -> PrecessionTrajectory:
    """Evolve physical Euler angles on a strictly increasing source-time grid."""

    config.validate()
    tau_array = np.ascontiguousarray(np.asarray(tau, dtype=np.float64))
    if tau_array.ndim != 1 or tau_array.size == 0:
        raise ValueError("tau must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(tau_array)) or np.any(np.diff(tau_array) <= 0.0):
        raise ValueError("tau must be finite and strictly increasing")
    if abs(config.chi1[2] - carrier.chi1) > 1.0e-12 or abs(config.chi2[2] - carrier.chi2) > 1.0e-12:
        raise ValueError("spin z components must match the aligned carrier spins")

    reference_state = _initial_state(carrier, config)
    frame = _build_j_frame(carrier, config, reference_state)
    m1m = carrier.m1 / carrier.total_mass
    m2m = carrier.m2 / carrier.total_mass
    spin_coefficients = np.asarray((
        _spin_dot_3pn(m1m), _spin_dot_5pn(m1m), _spin_dot_7pn(m1m),
        _spin_dot_3pn(m2m), _spin_dot_5pn(m2m), _spin_dot_7pn(m2m),
    ), dtype=np.float64)
    carrier_coeffs = np.ascontiguousarray(carrier.mode22.coeffs, dtype=np.float64)
    sx = m1m * m1m * config.chi1[0] + m2m * m2m * config.chi2[0]
    sy = m1m * m1m * config.chi1[1] + m2m * m2m * config.chi2[1]
    alpha_ref = math.atan2(sy, sx) - math.pi
    gamma_ref = -alpha_ref
    reference_state[9] = gamma_ref

    def rhs(time: float, state: np.ndarray) -> np.ndarray:
        return _precession_rhs(
            time, state, carrier.eta, carrier_coeffs, spin_coefficients,
            frame[0], frame[1], frame[2],
        )

    def integrate(stop: float):
        if _evolution is not None:
            return _evolution._integrate(stop, rhs, reference_state)
        if stop == config.tau_ref:
            return None
        span = abs(stop - config.tau_ref)
        solution = solve_ivp(
            rhs,
            (config.tau_ref, stop),
            reference_state,
            method="DOP853",
            rtol=config.relative_tolerance,
            atol=config.absolute_tolerance,
            first_step=min(config.initial_step, span),
            dense_output=True,
        )
        if not solution.success or solution.sol is None:
            raise RuntimeError(f"precession integration failed: {solution.message}")
        return solution

    earliest = min(float(tau_array[0]), config.tau_ref)
    backward = integrate(earliest)
    forward = integrate(0.0)

    def state_at(time: float) -> np.ndarray:
        if time == config.tau_ref:
            return reference_state.copy()
        solution = backward if time < config.tau_ref else forward
        if solution is None:
            return reference_state.copy()
        return np.asarray(solution.sol(time), dtype=np.float64)

    peak_state = state_at(0.0)
    final_spin, aligned_final_spin = _final_spin(carrier, peak_state)
    alpha_slope = _ringdown_alpha_slope(final_spin, carrier.final_mass)

    raw_alpha = np.empty_like(tau_array)
    cosbeta = np.empty_like(tau_array)
    gamma = np.empty_like(tau_array)
    first_after_peak = int(np.searchsorted(tau_array, 0.0, side="right"))
    for index in range(first_after_peak):
        sample_state = state_at(float(tau_array[index]))
        angle, cosine = _state_angles(sample_state, frame)
        raw_alpha[index] = angle + alpha_ref
        cosbeta[index] = cosine
        gamma[index] = sample_state[9]

    first_at_or_after_ref = int(np.searchsorted(tau_array, config.tau_ref, side="left"))
    previous = alpha_ref
    for index in range(first_at_or_after_ref, first_after_peak):
        raw_alpha[index] = _unwrap_near(float(raw_alpha[index]), previous)
        previous = float(raw_alpha[index])
    previous = alpha_ref
    for index in range(min(first_at_or_after_ref, first_after_peak) - 1, -1, -1):
        raw_alpha[index] = _unwrap_near(float(raw_alpha[index]), previous)
        previous = float(raw_alpha[index])

    peak_alpha_raw, peak_cosbeta = _state_angles(peak_state, frame)
    previous_alpha = alpha_ref if first_after_peak == 0 else float(raw_alpha[first_after_peak - 1])
    peak_alpha = _unwrap_near(peak_alpha_raw + alpha_ref, previous_alpha)
    peak_gamma = float(peak_state[9])

    for index in range(first_after_peak, tau_array.size):
        raw_alpha[index] = peak_alpha + alpha_slope * tau_array[index]
        cosbeta[index] = peak_cosbeta
        gamma[index] = peak_gamma - peak_cosbeta * (raw_alpha[index] - peak_alpha)

    summary = PrecessionSummary(
        final_spin=final_spin,
        aligned_final_spin_at_peak=aligned_final_spin,
        ringdown_alpha_slope=alpha_slope,
        alpha_reference=alpha_ref,
        gamma_reference=gamma_ref,
        j_frame_x=frame[0].copy(),
        j_frame_y=frame[1].copy(),
        j_frame_z=frame[2].copy(),
    )
    return PrecessionTrajectory(
        tau=tau_array,
        alpha=raw_alpha,
        beta=np.arccos(np.clip(cosbeta, -1.0, 1.0)),
        gamma=gamma,
        summary=summary,
    )


__all__ = [
    "PrecessionConfig",
    "PrecessionSummary",
    "PrecessionTrajectory",
    "PrecessionEvolution",
    "evolve_euler_angles",
]
