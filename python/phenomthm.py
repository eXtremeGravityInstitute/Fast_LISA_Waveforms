#!/usr/bin/env python3
# Based on LALSimulation IMRPhenomTHM: Copyright (C) 2020 Hector Estelles.
# LAL-free Python port and changes: Copyright (C) 2026 Neil Cornish.
# SPDX-License-Identifier: GPL-2.0-or-later
# Distributed without warranty; see the GNU GPL for details.
"""Plain NumPy/Numba port of the stripped IMRPhenomTHM mode model.

The validated :mod:`phenomt22` implementation supplies the common 2,2
backbone.  This module ports the higher-mode coefficient construction and the
five positive-m carrier evaluations from ``IMRPhenomTHM.c``.  Complex PN
amplitude arguments are folded into scalar phases, giving the real-amplitude
interface needed by the fast LISA response::

    h_lm(t) = amplitude_lm(t) * exp(-1j * phase_lm(t)).

Negative-m modes follow ``h_l,-m = (-1)^l conj(h_lm)``.  The implementation
has no LAL, JAX, or Phentax runtime dependency.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.interpolate import CubicSpline

import phenomthm_fits as fits
from phenomt22 import (
    GPSEC,
    IMRPhenomT22,
    TSUN,
    _native_gsl_lu_solve,
    final_mass_2017,
    final_spin_2017,
)

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]

        def decorate(function):
            return function

        return decorate


PI = math.pi
T_CUT_AMP = -150.0
T_CUT_FREQ = -150.0
TCP_MERGER = -25.0
DEFAULT_POSITIVE_MODES = ((2, 2), (2, 1), (3, 3), (4, 4), (5, 5))
DEFAULT_MODES = tuple(
    mode for positive in DEFAULT_POSITIVE_MODES for mode in (positive, (positive[0], -positive[1]))
)
_NATIVE_HM_LIB = None
_NATIVE_HM_LOAD_ATTEMPTED = False

# Flat amplitude-coefficient layout used by the Numba evaluation kernel.
A_FAC0 = 0
A_TSHIFT = 1
A_ALPHA1 = 2
A_ALPHA1_PREC = 3
A_C1_PREC = 4
A_C2_PREC = 5
A_C3 = 6
A_C4_PREC = 7
A_MERGER_C1 = 8
A_MERGER_C2 = 9
A_MERGER_C3 = 10
A_MERGER_C4 = 11
A_OMEGA_CUT = 12
A_PHI_CUT = 13
A_PN_START = 14
# PN real: N, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, log, C1, C2, C3.
A_PN_REAL_COUNT = 12
A_PN_IMAG_START = A_PN_START + A_PN_REAL_COUNT
A_SIZE = A_PN_IMAG_START + 7

# Flat phase-coefficient layout.
P_OMEGA_RING = 0
P_OMEGA_RING_PREC = 1
P_OMEGA_PEAK = 2
P_DOMEGA_PEAK = 3
P_ALPHA1 = 4
P_ALPHA1_PREC = 5
P_C1_PREC = 6
P_C2 = 7
P_C3 = 8
P_C4 = 9
P_MC1 = 10
P_MC2 = 11
P_MC3 = 12
P_OFF_MERGER = 13
P_OFF_RD = 14
P_SIZE = 15


@dataclass(frozen=True)
class ModeState:
    ell: int
    emm: int
    amplitude_coeffs: np.ndarray | None
    phase_coeffs: np.ndarray | None
    phoff: float
    zero_by_symmetry: bool

    @property
    def abs_m(self) -> int:
        return abs(self.emm)


@dataclass(frozen=True)
class ModeSeries:
    ell: int
    emm: int
    tau: np.ndarray
    amplitude: np.ndarray
    phase: np.ndarray
    omega: np.ndarray

    @property
    def hlm(self) -> np.ndarray:
        return self.amplitude * np.exp(-1j * self.phase)


def _mode_phase_offset(ell: int, abs_m: int) -> float:
    if (ell, abs_m) == (2, 1):
        return 0.5 * PI
    if (ell, abs_m) == (3, 3):
        return -0.5 * PI
    if (ell, abs_m) == (4, 4):
        return PI
    if (ell, abs_m) == (5, 5):
        return 0.5 * PI
    return 0.0


def _validate_mode(mode: tuple[int, int]) -> tuple[int, int]:
    ell, emm = int(mode[0]), int(mode[1])
    if emm == 0 or (ell, abs(emm)) not in DEFAULT_POSITIVE_MODES:
        raise ValueError(f"unsupported IMRPhenomTHM mode {(ell, emm)}")
    return ell, emm


def _fit(function: str, ell: int, abs_m: int, *arguments: float) -> float:
    name = f"IMRPhenomT_{function}_{ell}{abs_m}"
    return float(getattr(fits, name)(*arguments))


def _qnm(function: str, ell: int, abs_m: int, spin: float, model22: IMRPhenomT22) -> float:
    if (ell, abs_m) == (2, 2):
        if function == "fring":
            return model22.omega_ring / (2.0 * PI) * final_mass_2017(model22.eta, model22.chi1, model22.chi2)
        if function == "fdamp":
            return model22.alpha1rd / (2.0 * PI) * final_mass_2017(model22.eta, model22.chi1, model22.chi2)
        raise ValueError("the 2,2 overtone damping fit is not exposed here")
    if function == "fdampn2":
        name = f"evaluate_QNMfit_fdamp{ell}{abs_m}n2"
    else:
        name = f"evaluate_QNMfit_{function}{ell}{abs_m}"
    return float(getattr(fits, name)(spin))


def _solve(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    try:
        return _native_gsl_lu_solve(matrix, rhs)
    except Exception:
        return np.linalg.solve(matrix, rhs)


def _load_native_hm_library():
    global _NATIVE_HM_LIB, _NATIVE_HM_LOAD_ATTEMPTED
    if _NATIVE_HM_LOAD_ATTEMPTED:
        return _NATIVE_HM_LIB
    _NATIVE_HM_LOAD_ATTEMPTED = True
    here = Path(__file__).resolve().parent
    for candidate in (here / "libphenomthm_native.dylib", here / "libphenomthm_native.so"):
        if not candidate.exists():
            continue
        try:
            library = ctypes.CDLL(str(candidate))
        except OSError:
            continue
        function = library.phenomthm_fill_higher_coeffs
        function.argtypes = [
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
        ]
        function.restype = ctypes.c_int
        _NATIVE_HM_LIB = library
        break
    return _NATIVE_HM_LIB


def _native_higher_coefficients(
    m1: float, m2: float, chi1: float, chi2: float
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    library = _load_native_hm_library()
    if library is None:
        raise RuntimeError("native IMRPhenomTHM coefficient library is not available")
    block_size = A_SIZE + P_SIZE
    values = np.empty(4 * block_size, dtype=np.float64)
    status = library.phenomthm_fill_higher_coeffs(
        m1,
        m2,
        chi1,
        chi2,
        values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if status != 0:
        raise RuntimeError(f"native IMRPhenomTHM coefficient initialization failed with status {status}")
    result: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for index, mode in enumerate(DEFAULT_POSITIVE_MODES[1:]):
        block = values[index * block_size:(index + 1) * block_size]
        result[mode] = (block[:A_SIZE].copy(), block[A_SIZE:].copy())
    return result


def _pn_amplitude_coefficients(
    ell: int, abs_m: int, eta: float, chi1: float, chi2: float, delta: float, spin: float
) -> np.ndarray:
    out = np.zeros(A_SIZE, dtype=np.float64)
    out[A_FAC0] = 2.0 * eta * math.sqrt(16.0 * PI / 5.0)
    real = out[A_PN_START:A_PN_IMAG_START]
    imag = out[A_PN_IMAG_START:A_SIZE]
    chi1sq = chi1 * chi1
    chi2sq = chi2 * chi2

    if (ell, abs_m) == (2, 1):
        out[A_TSHIFT] = fits.IMRPhenomT_tshift_21(eta, spin, chi1 - chi2)
        real[1] = delta / 3.0
        real[2] = -chi1 / 4.0 + chi2 / 4.0 - chi1 * delta / 4.0 - chi2 * delta / 4.0
        real[3] = -17.0 * delta / 84.0 + 5.0 * delta * eta / 21.0
        imag[3] = -delta / 6.0 - 2.0 * delta * math.log(2.0) / 3.0
        real[4] = (
            79.0 * chi1 / 84.0 - 79.0 * chi2 / 84.0
            + 79.0 * chi1 * delta / 84.0 + 79.0 * chi2 * delta / 84.0
            - 43.0 * chi1 / 42.0 + 43.0 * chi2 / 42.0
            - 43.0 * chi1 * delta / 42.0 - 43.0 * chi2 * delta / 42.0
            - 139.0 * chi1 * eta / 84.0 + 139.0 * chi2 * eta / 84.0
            - 139.0 * chi1 * delta * eta / 84.0 - 139.0 * chi2 * delta * eta / 84.0
            + 86.0 * chi1 * eta / 21.0 - 86.0 * chi2 * eta / 21.0
            + 43.0 * chi1 * delta * eta / 21.0 + 43.0 * chi2 * delta * eta / 21.0
            + delta * PI / 3.0
        )
        real[5] = -43.0 * delta / 378.0 - 509.0 * delta * eta / 378.0 + 79.0 * delta * eta * eta / 504.0
        imag[5] = -(
            -17.0 * delta / 168.0 + 353.0 * delta * eta / 84.0
            - 17.0 * delta * math.log(2.0) / 42.0 + delta * eta * math.log(2.0) / 7.0
        )
        real[6] = -17.0 * delta * PI / 84.0 + delta * eta * PI / 14.0
    elif (ell, abs_m) == (3, 3):
        out[A_TSHIFT] = fits.IMRPhenomT_tshift_33(eta, spin)
        real[1] = 0.7763237542601484 * delta
        real[3] = -3.1052950170405937 * delta + 1.5526475085202969 * delta * eta
        imag[3] = -1.371926598204461 * delta
        real[4] = -(
            -0.5822428156951114 * chi1 + 0.5822428156951114 * chi2
            - 7.316679009572791 * delta - 0.5822428156951114 * chi1 * delta
            - 0.5822428156951114 * chi2 * delta + 1.3585665699552598 * chi1
            - 1.3585665699552598 * chi2 + 1.3585665699552598 * chi1 * delta
            + 1.3585665699552598 * chi2 * delta + 1.7467284470853341 * chi1 * eta
            - 1.7467284470853341 * chi2 * eta + 1.7467284470853341 * chi1 * delta * eta
            + 1.7467284470853341 * chi2 * delta * eta - 5.434266279821039 * chi1 * eta
            + 5.434266279821039 * chi2 * eta - 2.7171331399105196 * chi1 * delta * eta
            - 2.7171331399105196 * chi2 * delta * eta
        )
        real[5] = -(-0.08680711070363478 * delta + 8.647776123213047 * delta * eta - 2.0866641516022777 * delta * eta * eta)
    elif (ell, abs_m) == (4, 4):
        out[A_TSHIFT] = fits.IMRPhenomT_tshift_44(eta, spin)
        real[2] = 0.751248226425348 * (1.0 - 3.0 * eta)
        real[4] = -4.049910893365739 + 14.489984730901032 * eta - 5.9758381647470875 * eta * eta
        real[5] = 0.751248226425348 * (4.0 * PI - 12.0 * eta * PI)
        imag[4] = 0.751248226425348 * (-2.854822555520438 + 13.189467666561313 * eta)
        real[6] = -(
            -8.0 * math.sqrt(0.7142857142857143)
            * (5.338016983016983 - 1088119.0 * eta / 28600.0 + 146879.0 * eta * eta / 2340.0 - 226097.0 * eta**3 / 17160.0)
            / 9.0
        )
    elif (ell, abs_m) == (5, 5):
        out[A_TSHIFT] = fits.IMRPhenomT_tshift_55(eta, spin)
        real[3] = 0.8013768943966973 * delta * (1.0 - 2.0 * eta)
        real[5] = 0.8013768943966973 * delta * (-6.743589743589744 + 688.0 * eta / 39.0 - 256.0 * eta * eta / 39.0)
        imag[5] = -3.0177162096765713 * delta + 12.454250695829877 * delta * eta
        real[6] = 12.58799882096634 * delta - 25.175997641932675 * delta * eta
    elif (ell, abs_m) == (2, 2):
        # The production 2,2 mode is evaluated by phenomt22 directly.
        out[A_TSHIFT] = 0.0
        real[0] = 1.0
        real[2] = -2.5476190476190474 + 55.0 * eta / 42.0
        real[3] = (
            -2.0 * chi1 / 3.0 - 2.0 * chi2 / 3.0
            - 2.0 * chi1 * delta / 3.0 + 2.0 * chi2 * delta / 3.0
            + 2.0 * chi1 * eta / 3.0 + 2.0 * chi2 * eta / 3.0 + 2.0 * PI
        )
        real[4] = (
            -1.437169312169312 + chi1sq / 2.0 + chi2sq / 2.0
            + chi1sq * delta / 2.0 - chi2sq * delta / 2.0
            - 1069.0 * eta / 216.0 - chi1sq * eta + 2.0 * chi1 * chi2 * eta
            - chi2sq * eta + 2047.0 * eta * eta / 1512.0
        )
        real[5] = -107.0 * PI / 21.0 + 34.0 * eta * PI / 21.0
        imag[4] = -24.0 * eta
        real[6] = 41.78634662956092 - 278185.0 * eta / 33264.0 - 20261.0 * eta**2 / 2772.0 + 114635.0 * eta**3 / 99792.0 - 856.0 * 0.5772156649015329 / 105.0 + 2.0 * PI**2 / 3.0 + 41.0 * eta * PI**2 / 96.0
        imag[5] = 428.0 * PI / 105.0
        real[7] = -2173.0 * PI / 756.0 - 2495.0 * eta * PI / 378.0 + 40.0 * eta**2 * PI / 27.0
        imag[6] = 14333.0 * eta / 162.0 - 4066.0 * eta**2 / 945.0
        real[8] = -428.0 / 105.0
    return out


@njit(cache=True)
def _inspiral_amplitude(x: float, coeffs: np.ndarray) -> complex:
    xhalf = math.sqrt(x)
    x1half = x * xhalf
    x2 = x * x
    x2half = x2 * xhalf
    x3 = x2 * x
    x3half = x3 * xhalf
    x4 = x2 * x2
    x4half = x4 * xhalf
    x5 = x3 * x2
    real = (
        coeffs[A_PN_START]
        + coeffs[A_PN_START + 1] * xhalf
        + coeffs[A_PN_START + 2] * x
        + coeffs[A_PN_START + 3] * x1half
        + coeffs[A_PN_START + 4] * x2
        + coeffs[A_PN_START + 5] * x2half
        + coeffs[A_PN_START + 6] * x3
        + coeffs[A_PN_START + 7] * x3half
        + coeffs[A_PN_START + 8] * math.log(16.0 * x) * x3
        + coeffs[A_PN_START + 9] * x4
        + coeffs[A_PN_START + 10] * x4half
        + coeffs[A_PN_START + 11] * x5
    )
    imag = (
        coeffs[A_PN_IMAG_START] * xhalf
        + coeffs[A_PN_IMAG_START + 1] * x
        + coeffs[A_PN_IMAG_START + 2] * x1half
        + coeffs[A_PN_IMAG_START + 3] * x2
        + coeffs[A_PN_IMAG_START + 4] * x2half
        + coeffs[A_PN_IMAG_START + 5] * x3
        + coeffs[A_PN_IMAG_START + 6] * x3half
    )
    return coeffs[A_FAC0] * x * complex(real, imag)


@njit(cache=True)
def _complex_amplitude(tau: float, x22: float, coeffs: np.ndarray) -> complex:
    if tau < T_CUT_AMP:
        return _inspiral_amplitude(x22, coeffs)
    tpeak = coeffs[A_TSHIFT]
    if tau > tpeak:
        tanhphi = math.tanh(coeffs[A_C2_PREC] * (tau - tpeak) + coeffs[A_C3])
        value = math.exp(-coeffs[A_ALPHA1_PREC] * (tau - tpeak)) * (
            coeffs[A_C1_PREC] * tanhphi + coeffs[A_C4_PREC]
        )
        return complex(value, 0.0)
    sech1 = 1.0 / math.cosh(coeffs[A_ALPHA1] * (tau - tpeak))
    sech2 = 1.0 / math.cosh(2.0 * coeffs[A_ALPHA1] * (tau - tpeak))
    value = (
        coeffs[A_MERGER_C1]
        + coeffs[A_MERGER_C2] * sech1
        + coeffs[A_MERGER_C3] * sech2 ** (1.0 / 7.0)
        + coeffs[A_MERGER_C4] * (tau - tpeak) ** 2
    )
    return complex(value, 0.0)


@njit(cache=True)
def _merger_omega(tau: float, phase: np.ndarray) -> float:
    x = math.asinh(phase[P_ALPHA1] * tau)
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    omega_bar = (
        1.0 - phase[P_OMEGA_PEAK] / phase[P_OMEGA_RING]
        + phase[P_DOMEGA_PEAK] / phase[P_ALPHA1] * x
        + phase[P_MC1] * x2
        + phase[P_MC2] * x3
        + phase[P_MC3] * x4
    )
    return phase[P_OMEGA_RING] * (1.0 - omega_bar)


@njit(cache=True)
def _rd_omega(tau: float, phase: np.ndarray) -> float:
    expc = math.exp(-phase[P_C2] * tau)
    num = phase[P_C1_PREC] * (-2.0 * phase[P_C2] * phase[P_C4] * expc * expc - phase[P_C2] * phase[P_C3] * expc)
    den = 1.0 + phase[P_C4] * expc * expc + phase[P_C3] * expc
    return num / den + phase[P_OMEGA_RING]


@njit(cache=True)
def _merger_phase_no_offset(tau: float, phase: np.ndarray) -> float:
    alpha = phase[P_ALPHA1]
    x = math.asinh(alpha * tau)
    root = math.sqrt(1.0 + alpha * alpha * tau * tau)
    cc, dd, ee = phase[P_MC1], phase[P_MC2], phase[P_MC3]
    omega_ring = phase[P_OMEGA_RING]
    aux = omega_ring * tau - omega_ring * (
        2.0 * cc * tau + 24.0 * ee * tau + 6.0 * dd * tau * x
        + phase[P_DOMEGA_PEAK] * tau * x / alpha
        + tau * (1.0 - phase[P_OMEGA_PEAK] / omega_ring)
        + cc * tau * x * x + 12.0 * ee * tau * x * x + dd * tau * x**3 + ee * tau * x**4
        - phase[P_DOMEGA_PEAK] * root / (alpha * alpha) - 6.0 * dd * root / alpha
        - 2.0 * cc * x * root / alpha - 24.0 * ee * x * root / alpha
        - 3.0 * dd * x * x * root / alpha - 4.0 * ee * x**3 * root / alpha
    )
    return aux


@njit(cache=True)
def _carrier_phase(tau: float, phi22: float, abs_m: int, amp: np.ndarray, phase: np.ndarray) -> float:
    if tau < T_CUT_FREQ:
        return 0.5 * abs_m * phi22
    if tau > 0.0:
        expc = math.exp(-phase[P_C2] * tau)
        num = 1.0 + phase[P_C3] * expc + phase[P_C4] * expc * expc
        den = 1.0 + phase[P_C3] + phase[P_C4]
        return phase[P_C1_PREC] * math.log(num / den) + phase[P_OMEGA_RING_PREC] * tau + phase[P_OFF_RD] - amp[A_PHI_CUT]
    return _merger_phase_no_offset(tau, phase) + phase[P_OFF_MERGER] - amp[A_PHI_CUT]


@njit(cache=True)
def _evaluate_higher_array(
    tau: np.ndarray,
    phi22: np.ndarray,
    omega22: np.ndarray,
    omega22_minus: np.ndarray,
    abs_m: int,
    phoff: float,
    amp: np.ndarray,
    phase: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = tau.size
    amplitude = np.empty(n, dtype=np.float64)
    carrier_phase = np.empty(n, dtype=np.float64)
    omega = np.empty(n, dtype=np.float64)
    eps = 1.0e-6
    for i in range(n):
        x = (0.5 * omega22[i]) ** (2.0 / 3.0)
        complex_amp = _complex_amplitude(tau[i], x, amp)
        amplitude[i] = abs(complex_amp)
        carrier_phase[i] = _carrier_phase(tau[i], phi22[i], abs_m, amp, phase) - phoff - math.atan2(complex_amp.imag, complex_amp.real)
        if tau[i] < T_CUT_FREQ:
            x0 = (0.5 * omega22_minus[i]) ** (2.0 / 3.0)
            a0 = _inspiral_amplitude(x0, amp)
            a1 = _inspiral_amplitude(x, amp)
            delta_arg = math.atan2(math.sin(math.atan2(a1.imag, a1.real) - math.atan2(a0.imag, a0.real)), math.cos(math.atan2(a1.imag, a1.real) - math.atan2(a0.imag, a0.real)))
            omega[i] = 0.5 * abs_m * omega22[i] - delta_arg / eps
        elif tau[i] > 0.0:
            omega[i] = _rd_omega(tau[i], phase)
        else:
            omega[i] = _merger_omega(tau[i], phase)
    return amplitude, carrier_phase, omega


def _set_amplitude_coefficients(model: "IMRPhenomTHM", ell: int, abs_m: int) -> np.ndarray:
    eta, spin, dchi, delta = model.eta, model.spin, model.dchi, model.delta
    amp = _pn_amplitude_coefficients(ell, abs_m, eta, model.chi1, model.chi2, delta, spin)
    insp = np.array([_fit(f"Inspiral_Amp_CP{i}", ell, abs_m, eta, spin, dchi, delta) for i in (1, 2, 3)])
    merger_cp = _fit("Merger_Amp_CP1", ell, abs_m, eta, spin, dchi, delta)
    peak = _fit("PeakAmp", ell, abs_m, eta, spin, dchi, delta)
    rd_name = f"IMRPhenomT_RD_Amp_C3_{ell}{abs_m}"
    rd_function = getattr(fits, rd_name)
    rd_c3 = float(rd_function(eta, spin, dchi) if (ell, abs_m) in ((2, 1), (5, 5)) else rd_function(eta, spin))

    fdamp = _qnm("fdamp", ell, abs_m, model.afinal, model.mode22) / model.final_mass
    fdamp2 = _qnm("fdampn2", ell, abs_m, model.afinal, model.mode22) / model.final_mass
    alpha1 = 2.0 * PI * fdamp
    alpha2 = 2.0 * PI * fdamp2
    amp[A_ALPHA1] = alpha1
    amp[A_ALPHA1_PREC] = alpha1
    amp[A_C3] = rd_c3
    c2 = 0.5 * (alpha2 - alpha1)
    tanhc3 = math.tanh(rd_c3)
    if abs(c2) > abs(0.5 * alpha1 / tanhc3):
        c2 = -0.5 * alpha1 / tanhc3
    amp[A_C2_PREC] = c2
    coshc3 = math.cosh(rd_c3)
    amp[A_C1_PREC] = peak * alpha1 * coshc3 * coshc3 / c2
    amp[A_C4_PREC] = peak - amp[A_C1_PREC] * tanhc3

    times = (-2000.0, -250.0, -150.0)
    matrix = np.empty((3, 3), dtype=np.float64)
    rhs = np.empty(3, dtype=np.float64)
    for index, tau in enumerate(times):
        omega = model.mode22.omega22(tau)
        x = (0.5 * omega) ** (2.0 / 3.0)
        x2 = x * x
        x3 = x2 * x
        x4 = x2 * x2
        matrix[index] = (x4, x4 * math.sqrt(x), x3 * x2)
        rhs[index] = (insp[index] - _inspiral_amplitude(x, amp).real) / (amp[A_FAC0] * x)
    amp[A_PN_START + 9:A_PN_START + 12] = _solve(matrix, rhs)

    omega2 = model.mode22.omega22(T_CUT_AMP)
    omega1 = model.mode22.omega22(T_CUT_AMP - 1.0e-6)
    x2 = (0.5 * omega2) ** (2.0 / 3.0)
    x1 = (0.5 * omega1) ** (2.0 / 3.0)
    acut = _inspiral_amplitude(x2, amp)
    aprevious = _inspiral_amplitude(x1, amp)
    sign = math.copysign(1.0, acut.real)
    amp_insp = sign * abs(acut)
    damp_meco = sign * (abs(acut) - abs(aprevious)) / 1.0e-6
    tpeak = amp[A_TSHIFT]
    tau_cut = T_CUT_AMP - tpeak
    tau_cp = TCP_MERGER - tpeak
    sech1 = 1.0 / math.cosh(alpha1 * tau_cut)
    sech2 = 1.0 / math.cosh(2.0 * alpha1 * tau_cut)
    sechcp1 = 1.0 / math.cosh(alpha1 * tau_cp)
    sechcp2 = 1.0 / math.cosh(2.0 * alpha1 * tau_cp)
    matrix4 = np.array(
        [
            [1.0, sech1, sech2 ** (1.0 / 7.0), tau_cut * tau_cut],
            [1.0, sechcp1, sechcp2 ** (1.0 / 7.0), tau_cp * tau_cp],
            [1.0, 1.0, 1.0, 0.0],
            [0.0, -alpha1 * sech1 * math.tanh(alpha1 * tau_cut), (-2.0 / 7.0) * alpha1 * math.sinh(2.0 * alpha1 * tau_cut) * sech2 ** (8.0 / 7.0), 2.0 * tau_cut],
        ],
        dtype=np.float64,
    )
    amp[A_MERGER_C1:A_MERGER_C4 + 1] = _solve(matrix4, np.array((amp_insp, merger_cp, peak, damp_meco)))
    arg2 = math.atan2(acut.imag, acut.real)
    arg1 = math.atan2(aprevious.imag, aprevious.real)
    amp[A_OMEGA_CUT] = -math.atan2(math.sin(arg2 - arg1), math.cos(arg2 - arg1)) / 1.0e-6
    amp[A_PHI_CUT] = arg2 + (PI if math.copysign(1.0, acut.real) < 0.0 else 0.0)
    return amp


def _set_phase_coefficients(model: "IMRPhenomTHM", ell: int, abs_m: int, amp: np.ndarray) -> np.ndarray:
    eta, spin, dchi, delta = model.eta, model.spin, model.dchi, model.delta
    out = np.zeros(P_SIZE, dtype=np.float64)
    out[P_OMEGA_RING] = 2.0 * PI * _qnm("fring", ell, abs_m, model.afinal, model.mode22) / model.final_mass
    out[P_ALPHA1] = 2.0 * PI * _qnm("fdamp", ell, abs_m, model.afinal, model.mode22) / model.final_mass
    alpha2 = 2.0 * PI * _qnm("fdampn2", ell, abs_m, model.afinal, model.mode22) / model.final_mass
    out[P_OMEGA_RING_PREC] = out[P_OMEGA_RING]
    out[P_ALPHA1_PREC] = out[P_ALPHA1]
    out[P_OMEGA_PEAK] = _fit("PeakFrequency", ell, abs_m, eta, spin, dchi, delta) if (ell, abs_m) != (3, 3) else fits.IMRPhenomT_PeakFrequency_33(eta, spin, dchi)
    out[P_C3] = _fit("RD_Freq_D3", ell, abs_m, eta, spin, dchi, delta)
    out[P_C2] = _fit("RD_Freq_D2", ell, abs_m, eta, spin, dchi, delta)
    out[P_C4] = 0.0
    out[P_C1_PREC] = (1.0 + out[P_C3]) * (out[P_OMEGA_RING] - out[P_OMEGA_PEAK]) / out[P_C2] / out[P_C3]

    omega_cut22 = model.mode22.omega22(T_CUT_FREQ)
    omega_cut = 0.5 * abs_m * omega_cut22
    domega_cut22 = (omega_cut22 - model.mode22.omega22(T_CUT_FREQ - 1.0e-7)) / 1.0e-7
    domega_cut = -0.5 * abs_m * domega_cut22 / out[P_OMEGA_RING]
    domega_peak = -(_rd_omega(1.0e-7, out) - _rd_omega(0.0, out)) / 1.0e-7 / out[P_OMEGA_RING]
    out[P_DOMEGA_PEAK] = domega_peak
    omega_cut_bar = 1.0 - (omega_cut + amp[A_OMEGA_CUT]) / out[P_OMEGA_RING]
    omega_merger = _fit("Merger_Freq_CP1", ell, abs_m, eta, spin, dchi, delta)
    omega_merger_cp = 1.0 - omega_merger / out[P_OMEGA_RING]
    alpha = out[P_ALPHA1]
    xcut = math.asinh(alpha * T_CUT_FREQ)
    xcp = math.asinh(alpha * TCP_MERGER)
    dencut = math.sqrt(1.0 + T_CUT_FREQ**2 * alpha**2)
    matrix = np.array(
        [
            [xcut**2, xcut**3, xcut**4],
            [xcp**2, xcp**3, xcp**4],
            [2.0 * alpha * xcut / dencut, 3.0 * alpha * xcut**2 / dencut, 4.0 * alpha * xcut**3 / dencut],
        ]
    )
    rhs = np.array(
        [
            omega_cut_bar - (1.0 - out[P_OMEGA_PEAK] / out[P_OMEGA_RING]) - domega_peak / alpha * xcut,
            omega_merger_cp - (1.0 - out[P_OMEGA_PEAK] / out[P_OMEGA_RING]) - domega_peak / alpha * xcp,
            domega_cut - domega_peak / dencut,
        ]
    )
    out[P_MC1:P_MC3 + 1] = _solve(matrix, rhs)
    return out


class IMRPhenomTHM:
    """LAL-free plain-Python IMRPhenomTHM mode evaluator."""

    def __init__(
        self,
        m1_seconds: float,
        m2_seconds: float,
        chi1: float,
        chi2: float,
        modes: Sequence[tuple[int, int]] | str = "default",
        *,
        coefficient_backend: str = "auto",
        final_spin: float | None = None,
    ):
        if m2_seconds > m1_seconds:
            m1_seconds, m2_seconds = m2_seconds, m1_seconds
            chi1, chi2 = chi2, chi1
        if m1_seconds <= 0.0 or m2_seconds <= 0.0:
            raise ValueError("component masses must be positive")
        self.m1 = float(m1_seconds)
        self.m2 = float(m2_seconds)
        self.chi1 = float(chi1)
        self.chi2 = float(chi2)
        self.total_mass = self.m1 + self.m2
        self.eta = min(0.25, self.m1 * self.m2 / self.total_mass**2)
        self.delta = abs((self.m1 - self.m2) / self.total_mass)
        self.spin = (self.m1**2 * self.chi1 + self.m2**2 * self.chi2) / (self.m1**2 + self.m2**2)
        self.dchi = self.chi1 - self.chi2
        self.final_mass = final_mass_2017(self.eta, self.chi1, self.chi2)
        self.afinal_aligned = final_spin_2017(self.eta, self.chi1, self.chi2)
        if final_spin is None:
            self.afinal = self.afinal_aligned
        else:
            self.afinal = float(final_spin)
            if not math.isfinite(self.afinal) or not -1.0 < self.afinal < 1.0:
                raise ValueError("final_spin must be finite and strictly between -1 and 1")
        self.mode22 = IMRPhenomT22.from_masses(
            self.m1,
            self.m2,
            self.chi1,
            self.chi2,
            coefficient_backend=coefficient_backend,
            final_spin=final_spin,
        )
        requested = DEFAULT_MODES if modes == "default" else tuple(modes)
        requested = tuple(_validate_mode(mode) for mode in requested)
        fits.set_fit_powers(self.eta, self.spin)
        backend = coefficient_backend.lower()
        if final_spin is not None:
            if backend in {"native", "reference"}:
                raise ValueError(
                    "a final-spin override requires coefficient_backend='python' or 'auto'"
                )
            backend = "python"
        native_higher = None
        if any((ell, abs(emm)) != (2, 2) for ell, emm in requested) and backend in {"auto", "native"}:
            try:
                native_higher = _native_higher_coefficients(
                    self.m1, self.m2, self.chi1, self.chi2
                )
            except Exception:
                if backend == "native":
                    raise
        self.coefficient_backend = "native" if native_higher is not None else self.mode22.coefficient_backend
        self.backend_name = "local-thm"
        positive: dict[tuple[int, int], ModeState] = {}
        for ell, emm in requested:
            key = (ell, abs(emm))
            if key in positive:
                continue
            zero = abs(emm) % 2 == 1 and self.delta < 1.0e-10 and abs(self.chi1 - self.chi2) < 1.0e-10
            if key == (2, 2):
                positive[key] = ModeState(ell, abs(emm), None, None, 0.0, zero)
            elif zero:
                positive[key] = ModeState(ell, abs(emm), None, None, _mode_phase_offset(*key), True)
            else:
                if native_higher is not None:
                    amp, phase = native_higher[key]
                else:
                    amp = _set_amplitude_coefficients(self, ell, abs(emm))
                    phase = _set_phase_coefficients(self, ell, abs(emm), amp)
                positive[key] = ModeState(ell, abs(emm), amp, phase, _mode_phase_offset(*key), False)
        self._positive_states = positive
        self.modes = requested

    @classmethod
    def from_solar_masses(
        cls, m1_solar: float, m2_solar: float, chi1: float, chi2: float, *args, **kwargs
    ) -> "IMRPhenomTHM":
        return cls(m1_solar * TSUN, m2_solar * TSUN, chi1, chi2, *args, **kwargs)

    @property
    def positive_modes(self) -> tuple[tuple[int, int], ...]:
        return tuple(self._positive_states)

    def _phase22_grid(self, tau: np.ndarray, phi0: float) -> np.ndarray:
        if tau.size < 2 or np.any(np.diff(tau) <= 0.0):
            raise ValueError("automatic 2,2 phase integration requires a strictly increasing grid")
        omega = self.mode22.evaluate_tau(tau)["omega22"]
        spline = CubicSpline(tau, omega, bc_type="natural")
        phase = np.empty_like(tau)
        phase[0] = phi0
        for i in range(1, tau.size):
            phase[i] = phase[i - 1] + spline.integrate(tau[i - 1], tau[i])
        return phase

    def evaluate_tau(
        self,
        tau: Iterable[float] | np.ndarray,
        *,
        phi22: Iterable[float] | np.ndarray | None = None,
        phi0: float = 0.0,
        phi22_at_tcut: float | None = None,
    ) -> Mapping[tuple[int, int], ModeSeries]:
        tau_array = np.ascontiguousarray(np.asarray(tau, dtype=np.float64))
        if tau_array.ndim != 1:
            raise ValueError("tau must be one-dimensional")
        if phi22 is None:
            phi22_array = self._phase22_grid(tau_array, phi0)
        else:
            phi22_array = np.ascontiguousarray(np.asarray(phi22, dtype=np.float64))
            if phi22_array.shape != tau_array.shape:
                raise ValueError("phi22 and tau must have the same shape")
        base = self.mode22.evaluate_tau(tau_array)
        omega22 = np.ascontiguousarray(base["omega22"])
        omega22_minus = np.ascontiguousarray(self.mode22.evaluate_tau(tau_array - 1.0e-6)["omega22"])
        evaluation_phase_coeffs: dict[tuple[int, int], np.ndarray] = {}

        if any(mode != (2, 2) for mode in self._positive_states):
            if phi22_at_tcut is not None:
                phi_cut = float(phi22_at_tcut)
            elif tau_array[0] <= T_CUT_FREQ <= tau_array[-1]:
                phi_cut = float(np.interp(T_CUT_FREQ, tau_array, phi22_array))
            else:
                raise ValueError(
                    "higher-mode phase construction requires either a grid "
                    "spanning tau=-150 or phi22_at_tcut"
                )
            for state in self._positive_states.values():
                if state.phase_coeffs is None:
                    continue
                phase = state.phase_coeffs.copy()
                phase[P_OFF_MERGER] = 0.5 * state.abs_m * phi_cut - _merger_phase_no_offset(T_CUT_FREQ, phase)
                phase[P_OFF_RD] = _merger_phase_no_offset(0.0, phase) + phase[P_OFF_MERGER]
                positive_series_key = (state.ell, state.abs_m)
                evaluation_phase_coeffs[positive_series_key] = phase

        positive_series: dict[tuple[int, int], ModeSeries] = {}
        for key, state in self._positive_states.items():
            if state.zero_by_symmetry:
                zero = np.zeros_like(tau_array)
                positive_series[key] = ModeSeries(*key, tau_array, zero.copy(), zero.copy(), zero.copy())
            elif key == (2, 2):
                positive_series[key] = ModeSeries(2, 2, tau_array, np.asarray(base["amp_abs"]), phi22_array.copy(), omega22.copy())
            else:
                phase_coeffs = evaluation_phase_coeffs[key]
                amplitude, phase, omega = _evaluate_higher_array(
                    tau_array, phi22_array, omega22, omega22_minus, state.abs_m,
                    state.phoff, state.amplitude_coeffs, phase_coeffs
                )
                positive_series[key] = ModeSeries(*key, tau_array, amplitude, phase, omega)

        result: dict[tuple[int, int], ModeSeries] = {}
        for ell, emm in self.modes:
            positive = positive_series[(ell, abs(emm))]
            if emm > 0:
                result[(ell, emm)] = positive
            else:
                phase = -positive.phase - ell * PI
                result[(ell, emm)] = ModeSeries(ell, emm, tau_array, positive.amplitude.copy(), phase, -positive.omega)
        return result

    def evaluate_times(
        self,
        times: Iterable[float] | np.ndarray,
        *,
        tc: float,
        phic: float = 0.0,
        distance_gpc: float | None = None,
    ) -> Mapping[tuple[int, int], ModeSeries]:
        times_array = np.asarray(times, dtype=np.float64)
        tau = (times_array - tc) / self.total_mass
        # Match the stripped C construction: use the analytic LAL-style phase
        # only to set the absolute reference at the first point, then integrate
        # the smooth fitted omega22 on the caller's grid.
        phi_start = self.mode22.phase22(float(tau[0])) + phic - self.mode22.phase22(0.0)
        phi = self._phase22_grid(tau, phi_start)
        dimensionless = self.evaluate_tau(tau, phi22=phi)
        scale = 1.0 if distance_gpc is None else math.sqrt(2.0) * self.eta * self.total_mass / (distance_gpc * GPSEC)
        return {
            mode: ModeSeries(
                series.ell,
                series.emm,
                times_array,
                scale * series.amplitude,
                series.phase,
                series.omega / self.total_mass,
            )
            for mode, series in dimensionless.items()
        }


__all__ = [
    "DEFAULT_MODES",
    "DEFAULT_POSITIVE_MODES",
    "IMRPhenomTHM",
    "ModeSeries",
    "ModeState",
    "NUMBA_AVAILABLE",
]
