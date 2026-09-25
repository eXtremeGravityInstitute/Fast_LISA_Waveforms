#!/usr/bin/env python3
# Based on LALSimulation IMRPhenomTPHM: Copyright (C) 2020 Hector Estelles.
# LAL-free Python port and changes: Copyright (C) 2026 Neil Cornish.
# SPDX-License-Identifier: GPL-2.0-or-later
# Distributed without warranty; see the GNU GPL for details.
"""Low-ell rotations and observer projections for IMRPhenomTPHM."""

from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]

        def decorate(function):
            return function

        return decorate


_FACTORIAL = np.asarray(
    (1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0,
     40320.0, 362880.0, 3628800.0),
    dtype=np.float64,
)


@njit(cache=True)
def _wigner_d(ell: int, mp: int, emm: int, beta: float) -> float:
    """Return ``d^ell_{mp,m}(beta)`` in the TPHM/LAL convention."""

    cosine = math.cos(0.5 * beta)
    sine = math.sin(0.5 * beta)
    prefactor = math.sqrt(
        _FACTORIAL[ell + emm]
        * _FACTORIAL[ell - emm]
        * _FACTORIAL[ell + mp]
        * _FACTORIAL[ell - mp]
    )
    total = 0.0
    for k in range(2 * ell + 1):
        a = ell + emm - k
        b = k
        cden = mp - emm + k
        d = ell - mp - k
        cpower = 2 * ell + emm - mp - 2 * k
        spower = mp - emm + 2 * k
        if a < 0 or cden < 0 or d < 0 or cpower < 0 or spower < 0:
            continue
        term = prefactor / (
            _FACTORIAL[a] * _FACTORIAL[b] * _FACTORIAL[cden] * _FACTORIAL[d]
        )
        if k & 1:
            term = -term
        total += term * cosine**cpower * sine**spower
    return total


def build_rotation(ell: int, alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Build ``R^ell_{m,m'}=e^-im alpha d^ell_{m',m} e^-im' gamma``."""

    if ell < 2 or ell > 5:
        raise ValueError("TPHM rotations support 2 <= ell <= 5")
    emms = np.arange(-ell, ell + 1)
    rotation = np.empty((2 * ell + 1, 2 * ell + 1), dtype=np.complex128)
    for row, emm in enumerate(emms):
        alpha_factor = np.exp(-1j * emm * alpha)
        for column, mp in enumerate(emms):
            rotation[row, column] = (
                alpha_factor
                * _wigner_d(ell, int(mp), int(emm), beta)
                * np.exp(-1j * mp * gamma)
            )
    return rotation


def spin_weighted_y_minus2(ell: int, emm: int, theta: float, phi: float) -> complex:
    if ell < 2 or ell > 5 or abs(emm) > ell:
        raise ValueError("invalid spin-weighted spherical-harmonic index")
    return (
        math.sqrt((2.0 * ell + 1.0) / (4.0 * math.pi))
        * _wigner_d(ell, 2, emm, theta)
        * np.exp(1j * emm * phi)
    )


def reference_adjoint_rotation(ell: int, summary) -> np.ndarray:
    beta_ref = math.acos(float(np.clip(summary.j_frame_z[2], -1.0, 1.0)))
    l0_to_j = build_rotation(
        ell,
        summary.alpha_reference,
        beta_ref,
        summary.gamma_reference,
    )
    return l0_to_j.conj().T


def build_projection(
    reference_rotation: np.ndarray,
    ell: int,
    theta: float,
    phi: float,
    polarization: float,
) -> np.ndarray:
    """Fold the fixed J-to-L0 rotation and sky projection into one vector."""

    harmonics = np.asarray(
        [
            spin_weighted_y_minus2(ell, emm, theta, phi)
            * np.exp(2j * polarization)
            for emm in range(-ell, ell + 1)
        ],
        dtype=np.complex128,
    )
    return harmonics @ reference_rotation


@njit(cache=True)
def _project_folded_pair_kernel(
    ell: int,
    abs_m: int,
    alpha: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
    projection: np.ndarray,
    positive_mode: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    count = alpha.size
    strain = np.empty(count, dtype=np.complex128)
    quadrature = np.empty(count, dtype=np.complex128)
    parity = -1.0 if ell & 1 else 1.0
    for index in range(count):
        coefficient_positive = 0.0j
        coefficient_negative = 0.0j
        for emm in range(-ell, ell + 1):
            projected_row = projection[emm + ell] * np.exp(-1j * emm * alpha[index])
            coefficient_positive += projected_row * _wigner_d(
                ell, abs_m, emm, beta[index]
            )
            coefficient_negative += projected_row * _wigner_d(
                ell, -abs_m, emm, beta[index]
            )
        coefficient_positive *= np.exp(-1j * abs_m * gamma[index])
        coefficient_negative *= np.exp(1j * abs_m * gamma[index])
        mode = positive_mode[index]
        strain[index] = (
            coefficient_positive * mode
            + parity * coefficient_negative * np.conj(mode)
        )
        shifted = -1j * mode
        quadrature[index] = (
            coefficient_positive * shifted
            + parity * coefficient_negative * np.conj(shifted)
        )
    return strain, quadrature


def project_folded_pair(
    ell: int,
    abs_m: int,
    alpha: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
    projection: np.ndarray,
    positive_mode: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    arrays = tuple(
        np.ascontiguousarray(np.asarray(value))
        for value in (alpha, beta, gamma, projection, positive_mode)
    )
    return _project_folded_pair_kernel(ell, abs_m, *arrays)


def rotate_multipole_series(
    ell: int,
    alpha: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
    coprecessing_modes: np.ndarray,
    reference_rotation: np.ndarray | None = None,
) -> np.ndarray:
    """Rotate a sample-major multipole, primarily for diagnostics."""

    count = len(alpha)
    output = np.empty_like(coprecessing_modes, dtype=np.complex128)
    for index in range(count):
        value = build_rotation(
            ell, float(alpha[index]), float(beta[index]), float(gamma[index])
        ) @ coprecessing_modes[index]
        if reference_rotation is not None:
            value = reference_rotation @ value
        output[index] = value
    return output


__all__ = [
    "build_projection",
    "build_rotation",
    "project_folded_pair",
    "reference_adjoint_rotation",
    "rotate_multipole_series",
    "spin_weighted_y_minus2",
]
