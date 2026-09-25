#!/usr/bin/env python3
# Based on LALSimulation IMRPhenomTHM: Copyright (C) 2020 Hector Estelles.
# Stripped C changes and Python port: Copyright (C) 2024, 2026 Neil Cornish.
# SPDX-License-Identifier: GPL-2.0-or-later
# Distributed without warranty; see the GNU GPL for details.
"""Faithful Python port of the stripped 2,2 IMRPhenomT model.

This module mirrors the local C implementation in ``IMRPhenomT.c`` as closely
as possible.  The C code provides the intrinsic 2,2 angular frequency
``omega22(t/M)`` and complex mode amplitude.  The phase used by
``PhenomT_TDI.c`` is obtained by integrating the angular frequency on the chosen
time grid; the helper ``phase_from_times`` follows that convention.

The coefficient construction contains small ill-conditioned linear systems and
a finite-difference derivative at the inspiral/merger join.  Last-bit changes in
that setup can be amplified to visible merger-frequency differences, so the
default ``coefficient_backend="auto"`` first reads coefficients from the local
``phenomt22_c_reference`` executable when it is available.  Use
``coefficient_backend="python"`` to exercise the pure Python transcription.

For LAL comparisons, use ``lal_reference_tau_from_frequency`` to choose the
phase reference epoch from ``fmin``/``fRef``.  This intentionally mirrors LAL's
``GetTimeOfFreq`` convention, whose early-time root finder uses the shifted
TaylorT3 theta even when the default production waveform branch does not.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from dataclasses import dataclass
import math
from pathlib import Path
import subprocess
from typing import Iterable

import numpy as np

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only when numba is absent.
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]

        def decorate(func):
            return func

        return decorate


TSUN = 4.925490947641267e-6
GPSEC = 1.0292712503794875e17
T_CUT_AMP = -150.0
TCP_MERGER = -25.0
EULER_GAMMA = 0.577215664901532860606512090082402431
_GSL_LIB = None
_GSL_LOAD_ATTEMPTED = False
_NATIVE_MODEL_LIB = None
_NATIVE_MODEL_LOAD_ATTEMPTED = False

I_T_CUT22 = 0
I_OMEGA_RING = 1
I_OMEGA_PEAK = 2
I_DOMEGA_PEAK = 3
I_A0 = 4
I_TSHIFT = 5
I_ALPHA1RD = 6
I_ALPHA2RD = 7
I_ALPHA21RD = 8
I_C1 = 9
I_C2 = 10
I_C3 = 11
I_C4 = 12
I_MERGER_C1 = 13
I_MERGER_C2 = 14
I_MERGER_C3 = 15
I_MERGER_C4 = 16
I_OMEGA_CUT_PNAMP = 17
I_PHI_CUT_PNAMP = 18
I_OMEGA = 19
I_AMPR = I_OMEGA + 14
I_AMPI = I_AMPR + 14
I_CARRAY = I_AMPI + 14
I_CMARRAY = I_CARRAY + 5
NCOEFF = I_CMARRAY + 4


def _powers_from_one(x: float, n: int) -> np.ndarray:
    out = np.zeros(n, dtype=np.float64)
    out[0] = 1.0
    for i in range(1, n):
        out[i] = out[i - 1] * x
    return out


def _eta_spin_powers(eta: float, spin: float) -> tuple[np.ndarray, np.ndarray]:
    etapow = np.zeros(9, dtype=np.float64)
    spow = np.zeros(7, dtype=np.float64)
    etapow[1] = eta
    for i in range(2, 9):
        etapow[i] = etapow[i - 1] * eta
    spow[1] = spin
    for i in range(2, 7):
        spow[i] = spow[i - 1] * spin
    return etapow, spow


def _sech(x: float) -> float:
    return 1.0 / math.cosh(x)


def _load_gsl_library():
    global _GSL_LIB, _GSL_LOAD_ATTEMPTED
    if _GSL_LOAD_ATTEMPTED:
        return _GSL_LIB

    _GSL_LOAD_ATTEMPTED = True
    candidates = [
        ctypes.util.find_library("gsl"),
        "/opt/homebrew/lib/libgsl.dylib",
        "/usr/local/lib/libgsl.dylib",
        "/opt/local/lib/libgsl.dylib",
        "libgsl.so",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            lib = ctypes.CDLL(candidate)
        except OSError:
            continue

        size_t = ctypes.c_size_t
        lib.gsl_matrix_alloc.argtypes = [size_t, size_t]
        lib.gsl_matrix_alloc.restype = ctypes.c_void_p
        lib.gsl_matrix_free.argtypes = [ctypes.c_void_p]
        lib.gsl_matrix_free.restype = None
        lib.gsl_matrix_set.argtypes = [ctypes.c_void_p, size_t, size_t, ctypes.c_double]
        lib.gsl_matrix_set.restype = None
        lib.gsl_vector_alloc.argtypes = [size_t]
        lib.gsl_vector_alloc.restype = ctypes.c_void_p
        lib.gsl_vector_free.argtypes = [ctypes.c_void_p]
        lib.gsl_vector_free.restype = None
        lib.gsl_vector_set.argtypes = [ctypes.c_void_p, size_t, ctypes.c_double]
        lib.gsl_vector_set.restype = None
        lib.gsl_vector_get.argtypes = [ctypes.c_void_p, size_t]
        lib.gsl_vector_get.restype = ctypes.c_double
        lib.gsl_permutation_alloc.argtypes = [size_t]
        lib.gsl_permutation_alloc.restype = ctypes.c_void_p
        lib.gsl_permutation_free.argtypes = [ctypes.c_void_p]
        lib.gsl_permutation_free.restype = None
        lib.gsl_linalg_LU_decomp.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        lib.gsl_linalg_LU_decomp.restype = ctypes.c_int
        lib.gsl_linalg_LU_solve.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        lib.gsl_linalg_LU_solve.restype = ctypes.c_int
        _GSL_LIB = lib
        return _GSL_LIB

    return None


def _native_gsl_lu_solve(mat: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lib = _load_gsl_library()
    if lib is None:
        raise RuntimeError("GSL library not available")

    a_in = np.asarray(mat, dtype=np.float64)
    b_in = np.asarray(rhs, dtype=np.float64)
    n = int(b_in.shape[0])
    matrix = lib.gsl_matrix_alloc(n, n)
    b_vec = lib.gsl_vector_alloc(n)
    x_vec = lib.gsl_vector_alloc(n)
    perm = lib.gsl_permutation_alloc(n)
    if not matrix or not b_vec or not x_vec or not perm:
        raise MemoryError("GSL allocation failed")

    try:
        for i in range(n):
            lib.gsl_vector_set(b_vec, i, float(b_in[i]))
            for j in range(n):
                lib.gsl_matrix_set(matrix, i, j, float(a_in[i, j]))

        signum = ctypes.c_int()
        status = lib.gsl_linalg_LU_decomp(matrix, perm, ctypes.byref(signum))
        if status != 0:
            raise RuntimeError(f"gsl_linalg_LU_decomp failed with status {status}")
        status = lib.gsl_linalg_LU_solve(matrix, perm, b_vec, x_vec)
        if status != 0:
            raise RuntimeError(f"gsl_linalg_LU_solve failed with status {status}")

        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            out[i] = lib.gsl_vector_get(x_vec, i)
        return out
    finally:
        if matrix:
            lib.gsl_matrix_free(matrix)
        if b_vec:
            lib.gsl_vector_free(b_vec)
        if x_vec:
            lib.gsl_vector_free(x_vec)
        if perm:
            lib.gsl_permutation_free(perm)


def _fallback_lu_solve(mat: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve a tiny dense system when native GSL is not available.

    The IMRPhenomT calibration systems are small but can be rather
    ill-conditioned, especially the inspiral-frequency collocation system.
    This right-looking partial-pivot LU is close to GSL's arithmetic path, but
    the native GSL call below is preferred when bit-level agreement is needed.
    """
    a = np.array(mat, dtype=np.float64, copy=True)
    b = np.array(rhs, dtype=np.float64, copy=True)
    n = b.shape[0]
    perm = list(range(n))

    for j in range(n):
        pivot = j
        max_abs = 0.0
        for i in range(j, n):
            abs_s = abs(a[i, j])
            if abs_s > max_abs:
                max_abs = abs_s
                pivot = i

        if pivot != j:
            a[[j, pivot], :] = a[[pivot, j], :]
            perm[j], perm[pivot] = perm[pivot], perm[j]

        ajj = a[j, j]
        if ajj != 0.0:
            for i in range(j + 1, n):
                aij = a[i, j] / ajj
                a[i, j] = aij
                for k in range(j + 1, n):
                    a[i, k] -= aij * a[j, k]

    x = np.empty_like(b)
    for i in range(n):
        x[i] = b[perm[i]]

    for i in range(n):
        s = x[i]
        for j in range(i):
            s -= a[i, j] * x[j]
        x[i] = s

    for ii in range(n):
        i = n - 1 - ii
        s = x[i]
        for j in range(i + 1, n):
            s -= a[i, j] * x[j]
        x[i] = s / a[i, i]

    return x


def _gsl_lu_solve(mat: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    try:
        return _native_gsl_lu_solve(mat, rhs)
    except Exception:
        return _fallback_lu_solve(mat, rhs)


def _load_native_model_library():
    global _NATIVE_MODEL_LIB, _NATIVE_MODEL_LOAD_ATTEMPTED
    if _NATIVE_MODEL_LOAD_ATTEMPTED:
        return _NATIVE_MODEL_LIB

    _NATIVE_MODEL_LOAD_ATTEMPTED = True
    here = Path(__file__).resolve().parent
    candidates = [
        here / "libphenomt22_native.dylib",
        here / "libphenomt22_native.so",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            lib = ctypes.CDLL(str(candidate))
        except OSError:
            continue
        lib.phenomt22_fill_coeffs.argtypes = [
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
        ]
        lib.phenomt22_fill_coeffs.restype = ctypes.c_int
        _NATIVE_MODEL_LIB = lib
        return _NATIVE_MODEL_LIB

    return None


def _native_model_coeffs(m1: float, m2: float, chi1: float, chi2: float) -> np.ndarray:
    lib = _load_native_model_library()
    if lib is None:
        raise RuntimeError("native IMRPhenomT coefficient library not available")

    coeffs = np.zeros(NCOEFF, dtype=np.float64)
    ptr = coeffs.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    status = lib.phenomt22_fill_coeffs(float(m1), float(m2), float(chi1), float(chi2), ptr)
    if status != 0:
        raise RuntimeError(f"phenomt22_fill_coeffs failed with status {status}")
    return coeffs


def _reference_executable_coeffs(m1: float, m2: float, chi1: float, chi2: float) -> np.ndarray:
    exe = Path(__file__).resolve().parent / "phenomt22_c_reference"
    if not exe.exists():
        raise RuntimeError("phenomt22_c_reference executable not available")

    cmd = [
        str(exe),
        f"{m1:.17e}",
        f"{m2:.17e}",
        f"{chi1:.17e}",
        f"{chi2:.17e}",
        "0.0",
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=True)
    coeffs = np.zeros(NCOEFF, dtype=np.float64)
    scalar_index = {
        "tCut22": I_T_CUT22,
        "omegaRING": I_OMEGA_RING,
        "omegaPeak": I_OMEGA_PEAK,
        "domegaPeak": I_DOMEGA_PEAK,
        "A0": I_A0,
        "tshift": I_TSHIFT,
        "alpha1RD": I_ALPHA1RD,
        "alpha2RD": I_ALPHA2RD,
        "alpha21RD": I_ALPHA21RD,
        "c1": I_C1,
        "c2": I_C2,
        "c3": I_C3,
        "c4": I_C4,
        "mergerC1": I_MERGER_C1,
        "mergerC2": I_MERGER_C2,
        "mergerC3": I_MERGER_C3,
        "mergerC4": I_MERGER_C4,
        "omegaCutPNAMP": I_OMEGA_CUT_PNAMP,
        "phiCutPNAMP": I_PHI_CUT_PNAMP,
    }

    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) < 2 or parts[0] != "#":
            continue
        tag = parts[1]
        if tag == "coeff_scalars":
            for i in range(2, len(parts), 2):
                coeffs[scalar_index[parts[i]]] = float(parts[i + 1])
        elif tag == "omega":
            coeffs[I_OMEGA : I_OMEGA + 14] = [float(x) for x in parts[2:]]
        elif tag == "ampR":
            coeffs[I_AMPR : I_AMPR + 14] = [float(x) for x in parts[2:]]
        elif tag == "ampI":
            coeffs[I_AMPI : I_AMPI + 14] = [float(x) for x in parts[2:]]
        elif tag == "carray":
            coeffs[I_CARRAY : I_CARRAY + 5] = [float(x) for x in parts[2:]]
        elif tag == "CMarray":
            coeffs[I_CMARRAY : I_CMARRAY + 4] = [float(x) for x in parts[2:]]

    if coeffs[I_A0] == 0.0 or coeffs[I_OMEGA_RING] == 0.0:
        raise RuntimeError("failed to parse coefficients from phenomt22_c_reference")
    return coeffs


def qnm_fring22(final_spin: float) -> float:
    x = final_spin
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x5 = x3 * x2
    x6 = x3 * x3
    x7 = x4 * x3
    return (
        0.05947169566573468
        - 0.14989771215394762 * x
        + 0.09535606290986028 * x2
        + 0.02260924869042963 * x3
        - 0.02501704155363241 * x4
        - 0.005852438240997211 * x5
        + 0.0027489038393367993 * x6
        + 0.0005821983163192694 * x7
    ) / (
        1.0
        - 2.8570126619966296 * x
        + 2.373335413978394 * x2
        - 0.6036964688511505 * x4
        + 0.0873798215084077 * x6
    )


def qnm_fdamp22(final_spin: float) -> float:
    x = final_spin
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x5 = x3 * x2
    x6 = x3 * x3
    return (
        0.014158792290965177
        - 0.036989395871554566 * x
        + 0.026822526296575368 * x2
        + 0.0008490933750566702 * x3
        - 0.004843996907020524 * x4
        - 0.00014745235759327472 * x5
        + 0.0001504546201236794 * x6
    ) / (
        1.0
        - 2.5900842798681376 * x
        + 1.8952576220623967 * x2
        - 0.31416610693042507 * x4
        + 0.009002719412204133 * x6
    )


def qnm_fdamp22n2(final_spin: float) -> float:
    x = final_spin
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x5 = x3 * x2
    x6 = x3 * x3
    x7 = x4 * x3
    x8 = x4 * x4
    return 0.043611742588188715 + (
        -0.004016191313442792 * x
        - 0.0027646155943395426 * x2
        + 0.001141927763953028 * x3
        + 0.007938320030300492 * x4
        - 0.0008263166671238823 * x5
        - 0.014025760257115768 * x6
        + 0.001792158578158245 * x7
        + 0.008824138122361842 * x8
    ) / (2.0 - 1.9477781396815619 * x)


def final_mass_2017(eta: float, chi1: float, chi2: float) -> float:
    delta = math.sqrt(1.0 - 4.0 * eta)
    m1 = 0.5 * (1.0 + delta)
    m2 = 0.5 * (1.0 - delta)
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta
    spin = (m1 * m1 * chi1 + m2 * m2 * chi2) / (m1 * m1 + m2 * m2)
    spin2 = spin * spin
    spin3 = spin2 * spin
    dchi = chi1 - chi2
    dchi2 = dchi * dchi

    no_spin = (
        0.057190958417936644 * eta
        + 0.5609904135313374 * eta2
        - 0.84667563764404 * eta3
        + 3.145145224278187 * eta4
    )
    eq_spin = (
        no_spin
        * (
            1.0
            + (-0.13084389181783257 - 1.1387311580238488 * eta + 5.49074464410971 * eta2) * spin
            + (-0.17762802148331427 + 2.176667900182948 * eta2) * spin2
            + (-0.6320191645391563 + 4.952698546796005 * eta - 10.023747993978121 * eta2) * spin3
        )
        / (1.0 + (-0.9919475346968611 + 0.367620218664352 * eta + 4.274567337924067 * eta2) * spin)
    )
    eq_spin -= no_spin
    uneq_spin = (
        -0.09803730445895877 * dchi * delta * (1.0 - 3.2283713377939134 * eta) * eta2
        + 0.01118530335431078 * dchi2 * eta3
        - 0.01978238971523653 * dchi * delta * (1.0 - 4.91667749015812 * eta) * eta * spin
    )
    return 1.0 - (no_spin + eq_spin + uneq_spin)


def final_spin_2017(eta: float, chi1: float, chi2: float) -> float:
    delta = math.sqrt(1.0 - 4.0 * eta)
    m1 = 0.5 * (1.0 + delta)
    m2 = 0.5 * (1.0 - delta)
    m1sq = m1 * m1
    m2sq = m2 * m2
    eta2 = eta * eta
    eta3 = eta2 * eta
    spin = (m1sq * chi1 + m2sq * chi2) / (m1sq + m2sq)
    spin2 = spin * spin
    spin3 = spin2 * spin
    dchi = chi1 - chi2
    dchi2 = dchi * dchi

    no_spin = (3.4641016151377544 * eta + 20.0830030082033 * eta2 - 12.333573402277912 * eta3) / (
        1.0 + 7.2388440419467335 * eta
    )
    eq_spin = (m1sq + m2sq) * spin + (
        (-0.8561951310209386 * eta - 0.09939065676370885 * eta2 + 1.668810429851045 * eta3) * spin
        + (0.5881660363307388 * eta - 2.149269067519131 * eta2 + 3.4768263932898678 * eta3) * spin2
        + (0.142443244743048 * eta - 0.9598353840147513 * eta2 + 1.9595643107593743 * eta3) * spin3
    ) / (1.0 + (-0.9142232693081653 + 2.3191363426522633 * eta - 9.710576749140989 * eta3) * spin)
    uneq_spin = (
        0.3223660562764661 * dchi * delta * (1.0 + 9.332575956437443 * eta) * eta2
        - 0.059808322561702126 * dchi2 * eta3
        + 2.3170397514509933 * dchi * delta * (1.0 - 3.2624649875884852 * eta) * eta3 * spin
    )
    return no_spin + eq_spin + uneq_spin


def peak_frequency_22(etapow: np.ndarray, spow: np.ndarray, dchi: float, delta: float) -> float:
    eta = etapow[1]
    spin = spow[1]
    return (
        0.27212130745330404
        + 0.40972689759932074 * eta
        - 0.0018392172960247433 * eta * dchi * dchi
        + spin * (0.09558832959428547 - 0.04834585264918328 * eta - 0.15275173823699056 * etapow[2])
        - 3.4232387074402153 * etapow[2]
        + 32.853772442252605 * etapow[3]
        - 1.4976829186605336 * dchi * delta * (1.0 - 4.775645585721007 * eta) * etapow[3]
        - 0.9981117852179613 * dchi * delta * (1.0 - 5.260098925354571 * eta) * spin * etapow[3]
        - 125.22505746137587 * etapow[4]
        + 179.3797198714914 * etapow[5]
        + (0.054391696704622204 - 0.1482682698299456 * eta + 0.08938162810617255 * etapow[2]) * spow[2]
        + (-0.020719540055375383 + 0.5090144456500953 * eta - 1.5809441589349338 * etapow[2]) * spow[3]
        + (0.024240736699062685 - 0.09490089674418004 * eta + 0.09518501714836035 * etapow[2]) * spow[4]
        + (0.09759303647532228 - 1.105520690228567 * eta + 2.921271981239294 * etapow[2]) * spow[5]
    )


def rd_freq_d2_22(etapow: np.ndarray, spow: np.ndarray, dchi: float, delta: float) -> float:
    eta = etapow[1]
    spin = spow[1]
    return (
        0.1598180460429256
        + 0.19120040104567676 * eta
        + (-0.012853620630980167 - 0.006532392920798404 * eta) * spin
        - 0.7733759581766899 * etapow[2]
        + 0.18151402648790957 * dchi * delta * (1.0 - 9.041198282315879 * eta) * etapow[2]
        + 0.27147713896183995 * dchi * delta * (1.0 - 5.653323210961101 * eta) * spin * etapow[2]
        - 0.01603489049446065 * dchi * dchi * etapow[3]
        + (-0.046785083372074494 + 0.102759380109996 * eta) * spow[2]
        + (0.0009883572415502464 - 0.050384608002279486 * eta) * spow[3]
    )


def rd_freq_d3_22(etapow: np.ndarray, spow: np.ndarray, dchi: float, delta: float) -> float:
    eta = etapow[1]
    spin = spow[1]
    return (
        2.6456463496860927
        - 28.079375863863458 * eta
        + 323.1691069138812 * etapow[2]
        - 0.5040057675360762 * dchi * delta * (1.0 + 21.786482297795278 * eta) * etapow[2]
        + 1.561247215701216 * dchi * delta * (1.0 - 1.7508069810164308 * eta) * spin * etapow[2]
        + spin * (3.091917073632116 - 17.345283345692266 * eta + 33.40735388809028 * etapow[2])
        - 1490.8128941604907 * etapow[3]
        + 0.1619056474567525 * dchi * dchi * etapow[3]
        + 2376.3257196613886 * etapow[4]
        + (0.734022429223849 - 0.029342234233198747 * eta - 9.281610698291932 * etapow[2]) * spow[2]
    )


def inspiral_amp_cp1_22(etapow: np.ndarray, spow: np.ndarray, dchi: float, delta: float) -> float:
    eta = etapow[1]
    spin = spow[1]
    return (
        0.00006480771730217768 * eta * dchi * dchi
        - 0.3543965558027252 * dchi * delta * (1.0 - 2.463526130684083 * eta) * etapow[3]
        + 0.01879295038873938 * dchi * delta * (1.0 - 5.236796607517272 * eta) * spin * etapow[3]
        + spin
        * (
            0.1472653807120573 * eta
            - 1.9636752493349356 * etapow[2]
            + 14.177521724634461 * etapow[3]
            - 48.94620901701877 * etapow[4]
            + 63.83730899015984 * etapow[5]
        )
        + eta
        * (
            0.8493442097893826
            - 13.211067914003836 * eta
            + 311.99021467938235 * etapow[2]
            - 4731.025904601601 * etapow[3]
            + 44821.93042533854 * etapow[4]
            - 264474.1374080295 * etapow[5]
            + 943246.2317701122 * etapow[6]
            - 1.8588135904328802e6 * etapow[7]
            + 1.5524778581809246e6 * etapow[8]
        )
        + (
            0.04902976057622393 * eta
            - 1.0152511131279736 * etapow[2]
            + 8.286289152216145 * etapow[3]
            - 30.19775956110767 * etapow[4]
            + 40.670065442751955 * etapow[5]
        )
        * spow[2]
        + (
            0.04780630695082567 * eta
            - 1.2177827888317065 * etapow[2]
            + 11.505675146308567 * etapow[3]
            - 46.733420749352135 * etapow[4]
            + 68.40821782168776 * etapow[5]
        )
        * spow[3]
    )


def inspiral_amp_cp2_22(etapow: np.ndarray, spow: np.ndarray, dchi: float, delta: float) -> float:
    eta = etapow[1]
    spin = spow[1]
    return (
        0.000100027278976821 * eta * dchi * dchi
        - 0.7578403155712378 * dchi * delta * (1.0 - 2.056456271350877 * eta) * etapow[3]
        - 0.14126282637778914 * dchi * delta * (1.0 - 2.5840771007494916 * eta) * spin * etapow[3]
        + spin * (0.2331970217833686 * eta - 1.5473968380422929 * etapow[2] + 5.973401506474942 * etapow[3] - 9.110484789161045 * etapow[4])
        + eta
        * (
            0.9904613241626621
            - 6.708006572605403 * eta
            + 127.40270095439482 * etapow[2]
            - 1723.355339710798 * etapow[3]
            + 15430.10086310527 * etapow[4]
            - 88744.26044058547 * etapow[5]
            + 313650.01696201024 * etapow[6]
            - 617887.8122937253 * etapow[7]
            + 518220.9267888211 * etapow[8]
        )
        + (0.08934817374146888 * eta - 0.8887847358339216 * etapow[2] + 3.7233864099350784 * etapow[3] - 5.814765403882651 * etapow[4]) * spow[2]
        + (0.04471990627820145 * eta - 0.642458648615624 * etapow[2] + 3.393481171493086 * etapow[3] - 6.092083983738554 * etapow[4]) * spow[3]
    )


def inspiral_amp_cp3_22(etapow: np.ndarray, spow: np.ndarray, dchi: float, delta: float) -> float:
    eta = etapow[1]
    spin = spow[1]
    return (
        0.0002459376633671657 * eta * dchi * dchi
        - 0.8794763631110696 * dchi * delta * (1.0 - 2.0751630535350096 * eta) * etapow[3]
        - 0.3319387797134261 * dchi * delta * (1.0 - 3.1838055629892184 * eta) * spin * etapow[3]
        + spin * (0.23505507416274007 * eta - 1.2449030421324767 * etapow[2] + 4.315803728759738 * etapow[3] - 6.384257606413192 * etapow[4])
        + eta
        * (
            1.0208762064809185
            - 3.3799457394243957 * eta
            + 16.242639717123314 * etapow[2]
            + 299.2297416582362 * etapow[3]
            - 5913.920743907752 * etapow[4]
            + 46388.231537995445 * etapow[5]
            - 192261.0498470111 * etapow[6]
            + 413750.14250475995 * etapow[7]
            - 364403.84935539874 * etapow[8]
        )
        + (0.09630827896641526 * eta - 0.7915321134872877 * etapow[2] + 2.86907420250287 * etapow[3] - 4.038995403653199 * etapow[4]) * spow[2]
        + (0.07395420485618898 * eta - 1.0289224187583748 * etapow[2] + 5.275845823734598 * etapow[3] - 9.206158044409037 * etapow[4]) * spow[3]
    )


def merger_amp_cp1_22(etapow: np.ndarray, spow: np.ndarray, dchi: float, delta: float) -> float:
    eta = etapow[1]
    spin = spow[1]
    return (
        0.0004059354652663733 * eta * dchi * dchi
        - 0.9382383412276684 * dchi * delta * (1.0 - 2.509151362054917 * eta) * etapow[3]
        - 0.6560748977864668 * dchi * delta * (1.0 - 3.426294113321932 * eta) * spin * etapow[3]
        + spin * (0.23465398091766254 * eta - 1.3398914201113978 * etapow[2] + 5.9073801933446495 * etapow[3] - 10.84221896204708 * etapow[4])
        + eta
        * (
            1.2946032382158479
            - 3.3343035556341816 * eta
            + 91.6430240976277 * etapow[2]
            - 1687.6195123629968 * etapow[3]
            + 19726.50907350641 * etapow[4]
            - 140798.18973779568 * etapow[5]
            + 594095.3303894227 * etapow[6]
            - 1.358657562562124e6 * etapow[7]
            + 1.2958912179017465e6 * etapow[8]
        )
        + (0.03174875260265387 * eta + 0.23082150180902375 * etapow[2] - 1.9901867982613048 * etapow[3] + 4.009389679757772 * etapow[4]) * spow[2]
        + (-0.04033221614773138 * eta + 0.8426888041517518 * etapow[2] - 4.742283264846479 * etapow[3] + 9.059923021547936 * etapow[4]) * spow[3]
    )


def peak_amp_22(etapow: np.ndarray, spow: np.ndarray, dchi: float, delta: float) -> float:
    eta = etapow[1]
    spin = spow[1]
    return (
        0.0017885007700308166 * eta * dchi * dchi
        - 0.5846280668038513 * dchi * delta * (1.0 - 4.879882766464646 * eta) * etapow[3]
        - 0.874161608112943 * dchi * delta * (1.0 - 1.690095043235707 * eta) * spin * etapow[3]
        + spin * (0.203557188205307 * eta - 2.4368458739010563 * etapow[2] + 12.206344183078137 * etapow[3] - 23.417979354674692 * etapow[4])
        + eta
        * (
            1.4701266133411792
            - 1.387711607537906 * eta
            + 25.641251409467607 * etapow[2]
            - 186.013359336165 * etapow[3]
            + 801.3039484150348 * etapow[4]
            - 1893.8181854645718 * etapow[5]
            + 1946.531703997353 * etapow[6]
        )
        + (-0.0018659293826992745 * eta - 0.1888206507658455 * etapow[2] + 1.4677324802664107 * etapow[3] - 1.4019283350536489 * etapow[4]) * spow[2]
        + (-0.14699838946027494 * eta + 2.6186847787143837 * etapow[2] - 15.574381075605208 * etapow[3] + 31.239292792717016 * etapow[4]) * spow[3]
    )


def rd_amp_c3_22(etapow: np.ndarray, spow: np.ndarray) -> float:
    eta = etapow[1]
    spin = spow[1]
    return (
        -0.48053994718185694
        + 0.7023672141561462 * eta
        + spin * (-0.3597773028596323 + 1.4330280386796503 * eta - 3.239121799338561 * etapow[2])
        - 0.1993836305574211 * etapow[2]
        + (-0.2651107472061685 + 1.6433443489711386 * eta - 2.757772023954491 * etapow[2]) * spow[2]
        + (-0.01973537883495192 - 0.2410762147438714 * eta + 2.7315015976869756 * etapow[2]) * spow[3]
    )


def merger_freq_cp1_22(etapow: np.ndarray, spow: np.ndarray, dchi: float, delta: float) -> float:
    eta = etapow[1]
    spin = spow[1]
    return (
        -0.3926039690467202 * dchi * delta * (1.0 - 2.359180951434749 * eta) * etapow[2]
        - 0.28551098014898896 * dchi * delta * (1.0 - 3.414696100901444 * eta) * spin * etapow[2]
        + 0.003414004344822246 * dchi * dchi * etapow[3]
        + spin * (0.05697014130854102 + 0.07170430925984912 * eta - 0.9606499306623374 * etapow[2] + 5.440955307244598 * etapow[3] - 10.594319036394571 * etapow[4])
        + (0.10030959768350425 + 44.56725135920024 * eta + 163.96290948585087 * etapow[2] - 143.05635831020462 * etapow[3] + 393.8084861740473 * etapow[4])
        * math.pow(1.0 + 436.6494065618 * eta, -1.0)
        + (0.021213606590798472 + 0.2148355967310081 * eta - 2.7747405367196265 * etapow[2] + 13.771088220299802 * etapow[3] - 25.128755397215368 * etapow[4]) * spow[2]
        + (-0.003645992092251503 + 0.2137524962844931 * eta - 0.644979226062801 * etapow[2] - 1.7314849842209137 * etapow[3] + 5.573297392347478 * etapow[4]) * spow[3]
        + (0.029352214609533665 - 0.6020287633594307 * eta + 7.014738679280164 * etapow[2] - 36.027159248248296 * etapow[3] + 63.42605850359639 * etapow[4]) * spow[4]
        + (0.0356519646654399 - 0.5569780178251297 * eta + 4.017784725334053 * etapow[2] - 15.05881246593488 * etapow[3] + 22.94821359434365 * etapow[4]) * spow[5]
    )


def inspiral_taylor_t3_t0(etapow: np.ndarray, spow: np.ndarray, dchi: float, delta: float) -> float:
    eta = etapow[1]
    spin = spow[1]
    return (1.0 / eta) * (
        (-20.74399646637014 - 106.27711276502542 * eta) / (1.0 + 0.6516016033332481 * eta)
        + 0.0012450290074562259 * dchi * delta * (1.0 - 4.701633367918768e6 * eta) * etapow[2]
        - 111.5049997379579 * dchi * delta * (1.0 + 19.95458485773613 * eta) * spin * etapow[2]
        + 1204.6829118499857 * (1.0 - 4.025474056585855 * eta) * dchi * dchi * etapow[3]
        + spin
        * (
            338.7318821277009
            - 1553.5891860091408 * eta
            + 19614.263378999745 * etapow[2]
            - 156449.78737303324 * etapow[3]
            + 577363.3090369126 * etapow[4]
            - 802867.433363341 * etapow[5]
        )
        + (-55.75053935847546 - 290.36341163610575 * eta + 7873.7667183299345 * etapow[2] - 43585.59040070178 * etapow[3] + 87229.84668746481 * etapow[4] - 32469.263449695136 * etapow[5]) * spow[2]
        + (-102.8269343111326 + 5121.845705262981 * eta - 93026.46878769135 * etapow[2] + 650989.6793529999 * etapow[3] - 1.8846061037110784e6 * etapow[4] + 1.861602620702142e6 * etapow[5]) * spow[3]
        + (-7.294950933078567 + 314.24955197427136 * eta - 3751.8509582195657 * etapow[2] + 21205.339564205595 * etapow[3] - 46448.94771114493 * etapow[4] + 20310.512558558552 * etapow[5]) * spow[4]
        + (97.22312282683716 - 4556.60375328623 * eta + 76308.73046927384 * etapow[2] - 468784.4188333802 * etapow[3] + 998692.0246600509 * etapow[4] - 322905.9042578296 * etapow[5]) * spow[5]
    )


def inspiral_fit_values(etapow: np.ndarray, spow: np.ndarray, dchi: float, delta: float) -> np.ndarray:
    fit_values = np.zeros(6, dtype=np.float64)
    eta = etapow[1]
    spin = spow[1]

    fit_values[1] = (
        -0.014968864336704284 * dchi * delta * (1.0 - 1.942061808318584 * eta) * etapow[2]
        + 0.0017312772309375462 * dchi * delta * (1.0 - 0.07106994121956058 * eta) * spin * etapow[2]
        + spin * (0.0019208448318368731 - 0.0013579968243452476 * eta - 0.0033501404728414627 * etapow[2] + 0.008914420175326192 * etapow[3])
        + 6.687615165457298e-6 * dchi * dchi * etapow[3]
        + (0.02104073275966069 + 717.1534194224539 * eta + 85.37320237350282 * etapow[2] + 12.789214868358362 * etapow[3] - 16.00243777208413 * etapow[4]) / (1.0 + 32934.586638893634 * eta)
        + (-8.306810248117731e-6 + 0.00009918593182087119 * eta - 0.003805916669791129 * etapow[2] + 0.009854209286892323 * etapow[3]) * spow[2]
        + (-5.578836442449699e-6 - 0.0030378960591856616 * eta + 0.03746366675135751 * etapow[2] - 0.10298471015315146 * etapow[3]) * spow[3]
        + (0.00004425141111368952 - 0.0008702073302258368 * eta + 0.006538604805919268 * etapow[2] - 0.01578597166324495 * etapow[3]) * spow[4]
        + (-0.000019469656288570753 + 0.002969863931498354 * eta - 0.03643271052162611 * etapow[2] + 0.09959495981802587 * etapow[3]) * spow[5]
        + (-0.000042037164406446896 + 0.0007336074135429041 * eta - 0.005603356997202016 * etapow[2] + 0.013439843000090702 * etapow[3]) * spow[6]
    )
    fit_values[2] = (
        -0.04486391236129559 * dchi * delta * (1.0 - 1.8997912248414794 * eta) * etapow[2]
        - 0.003531802135161727 * dchi * delta * (1.0 - 8.001211450141325 * eta) * spin * etapow[2]
        + spin * (0.0061664395419698285 - 0.0040934633081508905 * eta - 0.009180337242551828 * etapow[2] + 0.020338583755834694 * etapow[3])
        + 0.00006524644306613066 * dchi * dchi * etapow[3]
        + (0.03711511661217631 - 0.10663782888636487 * eta - 0.09963406984414182 * etapow[2] + 0.6597367702009397 * etapow[3] - 2.777344875144891 * etapow[4] + 4.220674345359693 * etapow[5]) / (1.0 - 3.2125452791404148 * eta)
        + (0.00044302547647888445 + 0.000424246501303979 * eta - 0.01394093576260671 * etapow[2] + 0.02634851560709597 * etapow[3]) * spow[2]
        + (0.00011582043047950321 - 0.008282652950117982 * eta + 0.08965067576998058 * etapow[2] - 0.23963885130463913 * etapow[3]) * spow[3]
        + (0.0006123158975881322 - 0.007809160444435783 * eta + 0.028517174579539676 * etapow[2] - 0.03717957419042746 * etapow[3]) * spow[4]
        + (-0.0000885530893214531 + 0.005939789043536808 * eta - 0.07106551435109858 * etapow[2] + 0.1891131957235774 * etapow[3]) * spow[5]
        + (-0.0005110853374341054 + 0.0038762476596420855 * eta + 0.005094077179675256 * etapow[2] - 0.047971766995287136 * etapow[3]) * spow[6]
    )
    fit_values[3] = (
        -0.10196878573773932 * dchi * delta * (1.0 - 1.8918584778973513 * eta) * etapow[2]
        - 0.018820536453940443 * dchi * delta * (1.0 - 3.7307154599131183 * eta) * spin * etapow[2]
        - 0.00013162098437956188 * dchi * dchi * etapow[3]
        + spin * (0.0145572994468378 - 0.0017482433991394227 * eta - 0.10299007619034371 * etapow[2] + 0.4581039376357615 * etapow[3] - 0.7123678787549022 * etapow[4])
        + (0.05489007025458171 + 5.852073438961151 * eta + 2.74597705533403 * etapow[2] + 4.834336623113389 * etapow[3] - 26.931994454691022 * etapow[4] + 57.67035368809743 * etapow[5]) / (1.0 + 105.52132834236778 * eta)
        + (0.003001211395915229 + 0.0017929418998452987 * eta - 0.13776590125456148 * etapow[2] + 0.7471133710854526 * etapow[3] - 1.3620323111858437 * etapow[4]) * spow[2]
        + (0.001143282743686261 - 0.05793457776296727 * eta + 0.7841331051705482 * etapow[2] - 3.4936244160305323 * etapow[3] + 4.802357041496856 * etapow[4]) * spow[3]
        + (0.0009168588840889624 - 0.03261437094899735 * eta + 0.3472881896838799 * etapow[2] - 1.3634383958859384 * etapow[3] + 1.7313939586675267 * etapow[4]) * spow[4]
        + (-0.0002794014744432316 + 0.055911057147527664 * eta - 0.8686311380514122 * etapow[2] + 4.096191294930781 * etapow[3] - 6.009676060669872 * etapow[4]) * spow[5]
        + (-0.0005046018052528331 + 0.029804593053788925 * eta - 0.3792653361049425 * etapow[2] + 1.6366976231421981 * etapow[3] - 2.26904099961476 * etapow[4]) * spow[6]
    )
    fit_values[4] = (
        -0.1831889759662071 * dchi * delta * (1.0 - 1.8484261527766557 * eta) * etapow[2]
        - 0.07586202965525136 * dchi * delta * (1.0 - 3.2918162656371983 * eta) * spin * etapow[2]
        + 0.0019259052728265817 * dchi * dchi * etapow[3]
        + spin * (0.02685637375751212 + 0.013341664908359861 * eta - 0.3057217933283597 * etapow[2] + 1.395763446325911 * etapow[3] - 2.2559396974665376 * etapow[4])
        + (0.0725639467287476 + 12.39400068457852 * eta + 12.907450928972402 * etapow[2] - 7.422660061864399 * etapow[3] + 66.32985901506036 * etapow[4] - 117.85875779454518 * etapow[5]) / (1.0 + 168.63492460136445 * eta)
        + (0.0087781653701194 + 0.006944161553839352 * eta - 0.3301149078235105 * etapow[2] + 1.6835714783903248 * etapow[3] - 2.950404929598742 * etapow[4]) * spow[2]
        + (0.0037229746496019625 - 0.17155338099487646 * eta + 2.5881802140836774 * etapow[2] - 13.14710199375518 * etapow[3] + 21.366803256010915 * etapow[4]) * spow[3]
        + (0.00278507305662002 - 0.12475855143364532 * eta + 1.8640209516178643 * etapow[2] - 10.117078727717564 * etapow[3] + 17.94244821676711 * etapow[4]) * spow[4]
        + (0.0010273954584773936 + 0.1713357629442166 * eta - 3.017249223460983 * etapow[2] + 15.855096360798678 * etapow[3] - 26.444621592311933 * etapow[4]) * spow[5]
        + (-0.00012207946532225968 + 0.11709700788855186 * eta - 2.0950821618097026 * etapow[2] + 11.925324501640054 * etapow[3] - 21.683978511818076 * etapow[4]) * spow[6]
    )
    fit_values[5] = (
        -0.2508206617297265 * dchi * delta * (1.0 - 1.861010982421798 * eta) * etapow[2]
        - 0.1392163711259171 * dchi * delta * (1.0 - 3.2669366465555796 * eta) * spin * etapow[2]
        + 0.0023126403170013045 * dchi * dchi * etapow[3]
        + spin * (0.036750064163293766 + 0.036904343404333906 * eta - 0.5238739410356437 * etapow[2] + 2.3292117112945223 * etapow[3] - 3.654184701923543 * etapow[4])
        + (0.08373610487663233 + 6.301736487754372 * eta + 9.03911386193751 * etapow[2] + 4.91153188278086 * etapow[3]) / (1.0 + 72.64820846804257 * eta)
        + (0.014963449678540705 + 0.008354571522567225 * eta - 0.41723078020683 * etapow[2] + 2.2007932082378785 * etapow[3] - 4.245354787320365 * etapow[4]) * spow[2]
        + (0.005706180633326235 - 0.15748500622007494 * eta + 2.3477109912232845 * etapow[2] - 11.413877195221694 * etapow[3] + 17.033120593116756 * etapow[4]) * spow[3]
        + (0.003890296981717687 - 0.15985471334551038 * eta + 2.560312006077997 * etapow[2] - 14.400920672743332 * etapow[3] + 26.10406142567958 * etapow[4]) * spow[4]
        + (0.005305988847210204 + 0.10869207132210629 * eta - 2.4201307115268875 * etapow[2] + 12.544899744864924 * etapow[3] - 19.550600837316903 * etapow[4]) * spow[5]
        + (0.002917248769788225 + 0.11851143848720952 * eta - 2.6640023622893416 * etapow[2] + 15.993378498844761 * etapow[3] - 29.752144941054446 * etapow[4]) * spow[6]
    )
    return fit_values


def _init_inspiral_t3(m1: float, m2: float, chi1: float, chi2: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    total_mass = m1 + m2
    delta = (m1 - m2) / total_mass
    eta = m1 * m2 / (total_mass * total_mass)
    eta2 = eta * eta
    eta3 = eta2 * eta
    chi1sq = chi1 * chi1
    chi2sq = chi2 * chi2
    omega = np.zeros(14, dtype=np.float64)
    amp_r = np.zeros(14, dtype=np.float64)
    amp_i = np.zeros(14, dtype=np.float64)

    omega[0] = 1.0
    omega[1] = 0.0
    omega[2] = 0.27641369047619047 + (11.0 * eta) / 32.0
    omega[3] = (-19.0 * (chi1 + chi2) * eta) / 80.0 + (
        -113.0 * (chi2 * (-1.0 + delta) - chi1 * (1.0 + delta)) - 96.0 * math.pi
    ) / 320.0
    omega[4] = (
        (1855099.0 + 1714608.0 * chi2 * chi2 * (-1.0 + delta) - 1714608.0 * chi1 * chi1 * (1.0 + delta)) / 1.4450688e7
        + ((56975.0 + 61236.0 * chi1 * chi1 - 119448.0 * chi1 * chi2 + 61236.0 * chi2 * chi2) * eta) / 258048.0
        + (371.0 * eta * eta) / 2048.0
    )
    omega[5] = (
        (-17.0 * (chi1 + chi2) * eta * eta) / 128.0
        + (-146597.0 * (chi2 * (-1.0 + delta) - chi1 * (1.0 + delta)) - 46374.0 * math.pi) / 129024.0
        + (eta * (-2.0 * (chi1 * (1213.0 - 63.0 * delta) + chi2 * (1213.0 + 63.0 * delta)) + 117.0 * math.pi)) / 2304.0
    )
    omega[6] = (
        -2.499258364444952
        - (16928263.0 * chi1sq) / 1.376256e8
        - (16928263.0 * chi2sq) / 1.376256e8
        - (16928263.0 * chi1sq * delta) / 1.376256e8
        + (16928263.0 * chi2sq * delta) / 1.376256e8
        + ((-2318475.0 + 18767224.0 * chi1sq - 54663952.0 * chi1 * chi2 + 18767224.0 * chi2sq) * eta2) / 1.376256e8
        + (235925.0 * eta3) / 1.769472e6
        + (107.0 * EULER_GAMMA) / 280.0
        - (6127.0 * chi1 * math.pi) / 12800.0
        - (6127.0 * chi2 * math.pi) / 12800.0
        - (6127.0 * chi1 * delta * math.pi) / 12800.0
        + (6127.0 * chi2 * delta * math.pi) / 12800.0
        + (53.0 * math.pi * math.pi) / 200.0
        + (
            eta
            * (
                632550449425.0
                + 35200873512.0 * chi1sq
                - 28527282000.0 * chi1 * chi2
                + 9605339856.0 * chi1sq * delta
                - 1512.0 * chi2sq * (-23281001.0 + 6352738.0 * delta)
                + 34172264448.0 * (chi1 + chi2) * math.pi
                - 22912243200.0 * math.pi * math.pi
            )
        )
        / 1.040449536e11
    )
    omega[7] = (
        (-12029.0 * (chi1 + chi2) * eta3) / 92160.0
        + (
            eta2
            * (
                507654.0 * chi1 * chi2sq
                - 838782.0 * chi2sq * chi2
                + chi2 * (-840149.0 + 507654.0 * chi1sq - 870576.0 * delta)
                + chi1 * (-840149.0 - 838782.0 * chi1sq + 870576.0 * delta)
                + 1701228.0 * math.pi
            )
        )
        / 1.548288e7
        + (
            eta
            * (
                218532006.0 * chi1 * chi2sq * (-1.0 + delta)
                - 1134.0 * chi2sq * chi2 * (-206917.0 + 71931.0 * delta)
                - chi2 * (1496368361.0 - 429508815.0 * delta + 218532006.0 * chi1sq * (1.0 + delta))
                + chi1 * (-1496368361.0 - 429508815.0 * delta + 1134.0 * chi1sq * (206917.0 + 71931.0 * delta))
                - 144.0 * (488825.0 + 923076.0 * chi1sq - 1782648.0 * chi1 * chi2 + 923076.0 * chi2sq) * math.pi
            )
        )
        / 1.8579456e8
        + (
            -6579635551.0 * chi2 * (-1.0 + delta)
            + 535759434.0 * chi2sq * chi2 * (-1.0 + delta)
            - chi1 * (-6579635551.0 + 535759434.0 * chi1sq) * (1.0 + delta)
            + (-565550067.0 - 465230304.0 * chi2sq * (-1.0 + delta) + 465230304.0 * chi1sq * (1.0 + delta)) * math.pi
        )
        / 1.30056192e9
    )

    amp0 = 2.0 * eta * math.sqrt(16.0 * math.pi / 5.0)
    amp_r[0] = 1.0
    amp_i[0] = 0.0
    amp_r[1] = 0.0
    amp_i[1] = 0.0
    amp_r[2] = -2.5476190476190474 + (55.0 * eta) / 42.0
    amp_i[2] = 0.0
    amp_r[3] = (
        (-2.0 * chi1) / 3.0
        - (2.0 * chi2) / 3.0
        - (2.0 * chi1 * delta) / (3.0 * ((1.0 - delta) / 2.0 + (1.0 + delta) / 2.0))
        + (2.0 * chi2 * delta) / (3.0 * ((1.0 - delta) / 2.0 + (1.0 + delta) / 2.0))
        + (2.0 * chi1 * eta) / 3.0
        + (2.0 * chi2 * eta) / 3.0
        + 2.0 * math.pi
    )
    amp_i[3] = 0.0
    amp_r[4] = (
        -1.437169312169312
        + chi1sq / 2.0
        + chi2sq / 2.0
        + (chi1sq * delta) / 2.0
        - (chi2sq * delta) / 2.0
        - (1069.0 * eta) / 216.0
        - chi1sq * eta
        + 2.0 * chi1 * chi2 * eta
        - chi2sq * eta
        + (2047.0 * eta2) / 1512.0
    )
    amp_i[4] = 0.0
    amp_r[5] = -(107.0 * math.pi) / 21.0 + (34.0 * eta * math.pi) / 21.0
    amp_i[5] = -24.0 * eta
    amp_r[6] = (
        41.78634662956092
        - (278185.0 * eta) / 33264.0
        - (20261.0 * eta2) / 2772.0
        + (114635.0 * eta3) / 99792.0
        - (856.0 * EULER_GAMMA) / 105.0
        + (2.0 * math.pi * math.pi) / 3.0
        + (41.0 * eta * math.pi * math.pi) / 96.0
    )
    amp_i[6] = (428.0 / 105.0) * math.pi
    amp_r[7] = (-2173.0 * math.pi) / 756.0 - (2495.0 * eta * math.pi) / 378.0 + (40.0 * eta2 * math.pi) / 27.0
    amp_i[7] = (14333.0 * eta) / 162.0 - (4066.0 * eta2) / 945.0
    return omega, amp_r, amp_i, amp0


def _taylor_t3(theta: float, omega: np.ndarray) -> float:
    th = _powers_from_one(theta, 8)
    omt = 0.0
    for i in range(8):
        omt += omega[i] * th[i]
    omt += 107.0 / 280.0 * math.log(2.0 * theta) * th[6]
    return omt * th[3] / 4.0


def _inspiral_omega_ansatz(theta: float, omega: np.ndarray) -> float:
    theta8 = math.pow(theta, 8.0)
    theta9 = theta8 * theta
    theta10 = theta9 * theta
    theta11 = theta10 * theta
    theta12 = theta11 * theta
    theta13 = theta12 * theta
    fac = theta * theta * theta / 8.0
    correction = omega[8] * theta8 + omega[9] * theta9 + omega[10] * theta10 + omega[11] * theta11 + omega[12] * theta12 + omega[13] * theta13
    return _taylor_t3(theta, omega) + 2.0 * fac * correction


def _inspiral_amp_complex(x: float, coeffs: np.ndarray) -> complex:
    xhalf = math.sqrt(x)
    xpow = _powers_from_one(xhalf, 11)
    amp_r = 0.0
    amp_i = 0.0
    for i in range(11):
        amp_r += coeffs[I_AMPR + i] * xpow[i]
        amp_i += coeffs[I_AMPI + i] * xpow[i]
    amp_r -= 428.0 / 105.0 * math.log(16.0 * x) * xpow[6]
    return coeffs[I_A0] * x * complex(amp_r, amp_i)


def _complex_amp_orientation(x: float, coeffs: np.ndarray) -> float:
    amp = _inspiral_amp_complex(x, coeffs)
    return math.atan2(amp.imag, amp.real)


def _rd_omega(t: float, coeffs: np.ndarray) -> float:
    c1 = coeffs[I_CARRAY + 1]
    c2 = coeffs[I_CARRAY + 2]
    c3 = coeffs[I_CARRAY + 3]
    c4 = coeffs[I_CARRAY + 4]
    expc = math.exp(-c2 * t)
    expc2 = expc * expc
    num = c1 * (-2.0 * c2 * c4 * expc2 - c2 * c3 * expc)
    den = 1.0 + c4 * expc2 + c3 * expc
    return num / den + coeffs[I_OMEGA_RING]


def _merger_omega_bar(t: float, coeffs: np.ndarray) -> float:
    x = math.asinh(coeffs[I_ALPHA1RD] * t)
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    return (
        1.0
        - coeffs[I_OMEGA_PEAK] / coeffs[I_OMEGA_RING]
        + (coeffs[I_DOMEGA_PEAK] / coeffs[I_ALPHA1RD]) * x
        + coeffs[I_CMARRAY + 1] * x2
        + coeffs[I_CMARRAY + 2] * x3
        + coeffs[I_CMARRAY + 3] * x4
    )


def _rd_amp(t: float, coeffs: np.ndarray) -> float:
    tau = t - coeffs[I_TSHIFT]
    return math.exp(-coeffs[I_ALPHA1RD] * tau) * (coeffs[I_C1] * math.tanh(coeffs[I_C2] * tau + coeffs[I_C3]) + coeffs[I_C4])


def _merger_amp(t: float, coeffs: np.ndarray) -> float:
    tau = t - coeffs[I_TSHIFT]
    sech1 = _sech(coeffs[I_ALPHA1RD] * tau)
    sech2 = _sech(2.0 * coeffs[I_ALPHA1RD] * tau)
    return coeffs[I_MERGER_C1] + coeffs[I_MERGER_C2] * sech1 + coeffs[I_MERGER_C3] * math.pow(sech2, 1.0 / 7.0) + coeffs[I_MERGER_C4] * tau * tau


def _omega22_py(t: float, eta: float, coeffs: np.ndarray) -> float:
    if t < coeffs[I_T_CUT22]:
        theta = math.pow(-eta * t / 5.0, -1.0 / 8.0)
        omega = coeffs[I_OMEGA : I_OMEGA + 14]
        return _inspiral_omega_ansatz(theta, omega)
    if t > 0.0:
        return _rd_omega(t, coeffs)
    return coeffs[I_OMEGA_RING] * (1.0 - _merger_omega_bar(t, coeffs))


def _lal_reference_omega22_py(t: float, eta: float, tt0: float, t_early: float, coeffs: np.ndarray) -> float:
    if t < t_early:
        theta = math.pow(eta * (tt0 - t) / 5.0, -1.0 / 8.0)
    elif t < coeffs[I_T_CUT22]:
        theta = math.pow(-eta * t / 5.0, -1.0 / 8.0)
    else:
        theta = 0.0

    if t < coeffs[I_T_CUT22]:
        return _inspiral_omega_ansatz(theta, coeffs[I_OMEGA : I_OMEGA + 14])
    if t > 0.0:
        return _rd_omega(t, coeffs)
    return coeffs[I_OMEGA_RING] * (1.0 - _merger_omega_bar(t, coeffs))


def _inspiral_phase22_py(t: float, eta: float, coeffs: np.ndarray) -> float:
    """LAL-style analytic antiderivative of the inspiral omega22 ansatz."""

    thetabar = math.pow(-eta * t, -1.0 / 8.0)
    theta2 = thetabar * thetabar
    theta3 = theta2 * thetabar
    theta4 = theta2 * theta2
    theta5 = theta4 * thetabar
    theta6 = theta3 * theta3
    theta7 = theta6 * thetabar
    omega = coeffs[I_OMEGA : I_OMEGA + 14]
    pow5_1_8 = math.pow(5.0, 0.125)
    pow5_1_4 = math.pow(5.0, 0.25)
    pow5_3_8 = math.pow(5.0, 0.375)
    pow5_1_2 = math.sqrt(5.0)
    pow5_5_8 = math.pow(5.0, 0.625)
    pow5_3_4 = math.pow(5.0, 0.75)
    pow5_7_8 = math.pow(5.0, 0.875)
    bracket = (
        3.0 * (-107.0 + 280.0 * omega[6]) * pow5_3_4
        + 321.0 * math.log(2.0 * thetabar * pow5_1_8) * pow5_3_4
        + 420.0 * omega[7] * thetabar * pow5_7_8
        + 56.0 * (25.0 * omega[8] + 3.0 * eta * t) * theta2
        + 1050.0 * omega[9] * pow5_1_8 * theta3
        + 280.0 * (3.0 * omega[10] + eta * omega[2] * t) * pow5_1_4 * theta4
        + 140.0 * (5.0 * omega[11] + 3.0 * eta * omega[3] * t) * pow5_3_8 * theta5
        + 120.0 * (5.0 * omega[12] + 7.0 * eta * omega[4] * t) * pow5_1_2 * theta6
        + 525.0 * omega[13] * pow5_5_8 * theta7
        + 105.0 * eta * omega[5] * t * math.log(-t) * pow5_5_8 * theta7
    )
    return -math.pow(5.0, -0.625) * math.pow(thetabar, -7.0) * bracket / (84.0 * eta * eta * t)


def _merger_phase22_base_py(t: float, coeffs: np.ndarray) -> float:
    """Analytic merger phase before its continuity offset is applied."""

    alpha1rd = coeffs[I_ALPHA1RD]
    omega_ring = coeffs[I_OMEGA_RING]
    omega_peak = coeffs[I_OMEGA_PEAK]
    domega_peak = coeffs[I_DOMEGA_PEAK]
    cc = coeffs[I_CMARRAY + 1]
    dd = coeffs[I_CMARRAY + 2]
    ee = coeffs[I_CMARRAY + 3]
    x = math.asinh(alpha1rd * t)
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    root = math.sqrt(1.0 + alpha1rd * alpha1rd * t * t)
    correction = (
        2.0 * cc * t
        + 24.0 * ee * t
        + 6.0 * dd * t * x
        + domega_peak * t * x / alpha1rd
        + t * (1.0 - omega_peak / omega_ring)
        + cc * t * x2
        + 12.0 * ee * t * x2
        + dd * t * x3
        + ee * t * x4
        - domega_peak * root / (alpha1rd * alpha1rd)
        - 6.0 * dd * root / alpha1rd
        - 2.0 * cc * x * root / alpha1rd
        - 24.0 * ee * x * root / alpha1rd
        - 3.0 * dd * x2 * root / alpha1rd
        - 4.0 * ee * x3 * root / alpha1rd
    )
    return omega_ring * t - omega_ring * correction


def _phase22_py(t: float, eta: float, coeffs: np.ndarray) -> float:
    """Continuous analytic 22 phase used to anchor the integrated phase."""

    t_cut22 = coeffs[I_T_CUT22]
    phase_insp_cut = _inspiral_phase22_py(t_cut22, eta, coeffs)
    merger_offset = phase_insp_cut - _merger_phase22_base_py(t_cut22, coeffs)
    if t < t_cut22:
        return _inspiral_phase22_py(t, eta, coeffs)
    if t <= 0.0:
        return _merger_phase22_base_py(t, coeffs) + merger_offset

    carray = coeffs[I_CARRAY : I_CARRAY + 5]
    expc = math.exp(-carray[2] * t)
    numerator = 1.0 + carray[3] * expc + carray[4] * expc * expc
    denominator = 1.0 + carray[3] + carray[4]
    rd_offset = _merger_phase22_base_py(0.0, coeffs) + merger_offset
    return carray[1] * math.log(numerator / denominator) + coeffs[I_OMEGA_RING] * t + rd_offset


def _amp22_py(t: float, x: float, coeffs: np.ndarray) -> complex:
    if t < T_CUT_AMP:
        return _inspiral_amp_complex(x, coeffs)
    if t > coeffs[I_TSHIFT]:
        return complex(_rd_amp(t, coeffs), 0.0)
    return complex(_merger_amp(t, coeffs), 0.0)


def _pack_coeffs(
    *,
    t_cut22: float,
    omega_ring: float,
    omega_peak: float,
    domega_peak: float,
    amp0: float,
    tshift: float,
    alpha1rd: float,
    alpha2rd: float,
    alpha21rd: float,
    c1: float,
    c2: float,
    c3: float,
    c4: float,
    merger_c1: float,
    merger_c2: float,
    merger_c3: float,
    merger_c4: float,
    omega_cut_pnamp: float,
    phi_cut_pnamp: float,
    omega: np.ndarray,
    amp_r: np.ndarray,
    amp_i: np.ndarray,
    carray: np.ndarray,
    cmarray: np.ndarray,
) -> np.ndarray:
    coeffs = np.zeros(NCOEFF, dtype=np.float64)
    coeffs[I_T_CUT22] = t_cut22
    coeffs[I_OMEGA_RING] = omega_ring
    coeffs[I_OMEGA_PEAK] = omega_peak
    coeffs[I_DOMEGA_PEAK] = domega_peak
    coeffs[I_A0] = amp0
    coeffs[I_TSHIFT] = tshift
    coeffs[I_ALPHA1RD] = alpha1rd
    coeffs[I_ALPHA2RD] = alpha2rd
    coeffs[I_ALPHA21RD] = alpha21rd
    coeffs[I_C1] = c1
    coeffs[I_C2] = c2
    coeffs[I_C3] = c3
    coeffs[I_C4] = c4
    coeffs[I_MERGER_C1] = merger_c1
    coeffs[I_MERGER_C2] = merger_c2
    coeffs[I_MERGER_C3] = merger_c3
    coeffs[I_MERGER_C4] = merger_c4
    coeffs[I_OMEGA_CUT_PNAMP] = omega_cut_pnamp
    coeffs[I_PHI_CUT_PNAMP] = phi_cut_pnamp
    coeffs[I_OMEGA : I_OMEGA + 14] = omega
    coeffs[I_AMPR : I_AMPR + 14] = amp_r
    coeffs[I_AMPI : I_AMPI + 14] = amp_i
    coeffs[I_CARRAY : I_CARRAY + 5] = carray
    coeffs[I_CMARRAY : I_CMARRAY + 4] = cmarray
    return coeffs


def _set_inspiral_fit(eta: float, dchi: float, spin: float, delta: float, omega: np.ndarray) -> None:
    etapow, spow = _eta_spin_powers(eta, spin)
    theta_points = np.array([0.33, 0.45, 0.55, 0.65, 0.75, 0.82], dtype=np.float64)
    omega_points = np.zeros(6, dtype=np.float64)

    tt0 = inspiral_taylor_t3_t0(etapow, spow, dchi, delta)
    t_early = -5.0 / (eta * math.pow(theta_points[0], 8.0))
    theta_ini = math.pow(eta * (tt0 - t_early) / 5.0, -1.0 / 8.0)
    omega_points[0] = _taylor_t3(theta_ini, omega)
    omega_points[1:] = inspiral_fit_values(etapow, spow, dchi, delta)[1:]

    mat = np.zeros((6, 6), dtype=np.float64)
    rhs = np.zeros(6, dtype=np.float64)
    for idx, theta in enumerate(theta_points):
        theta8 = math.pow(theta, 8.0)
        theta9 = theta * theta8
        theta10 = theta * theta9
        theta11 = theta * theta10
        theta12 = theta * theta11
        theta13 = theta * theta12
        rhs[idx] = (4.0 / theta / theta / theta) * (omega_points[idx] - _taylor_t3(theta, omega))
        mat[idx, :] = [theta8, theta9, theta10, theta11, theta12, theta13]
    omega[8:14] = _gsl_lu_solve(mat, rhs)


def _set_merger_ringdown_fit(
    m1: float,
    m2: float,
    chi1: float,
    chi2: float,
    eta: float,
    delta: float,
    spin: float,
    dchi: float,
    omega: np.ndarray,
    amp_r: np.ndarray,
    amp_i: np.ndarray,
    amp0: float,
    final_spin_override: float | None = None,
) -> np.ndarray:
    etapow, spow = _eta_spin_powers(eta, spin)
    carray = np.zeros(5, dtype=np.float64)
    cmarray = np.zeros(4, dtype=np.float64)

    final_mass = final_mass_2017(eta, chi1, chi2)
    final_spin = (
        final_spin_2017(eta, chi1, chi2)
        if final_spin_override is None
        else float(final_spin_override)
    )
    fring = qnm_fring22(final_spin) / final_mass
    fdamp = qnm_fdamp22(final_spin) / final_mass
    fdamp_n2 = qnm_fdamp22n2(final_spin) / final_mass
    alpha1rd = 2.0 * math.pi * fdamp
    omega_ring = 2.0 * math.pi * fring

    omega_peak = peak_frequency_22(etapow, spow, dchi, delta)
    carray[3] = rd_freq_d3_22(etapow, spow, dchi, delta)
    carray[2] = rd_freq_d2_22(etapow, spow, dchi, delta)
    carray[4] = 0.0
    carray[1] = (1.0 + carray[3] + carray[4]) * (omega_ring - omega_peak) / carray[2] / (carray[3] + 2.0 * carray[4])

    t_cut22 = -5.0 / (eta * math.pow(0.81, 8.0))
    omega_cut = _inspiral_omega_ansatz(0.81, omega)
    t_merger22 = -5.0 / (eta * math.pow(0.95, 8.0))
    omega_merger_cp = 1.0 - merger_freq_cp1_22(etapow, spow, dchi, delta) / omega_ring
    omega_cut_bar = 1.0 - omega_cut / omega_ring
    theta2 = math.pow(-eta * t_cut22 / 5.0, -1.0 / 8.0)
    theta1 = math.pow(-eta * (t_cut22 - 0.0000001) / 5.0, -1.0 / 8.0)
    domega_cut = -(_inspiral_omega_ansatz(theta2, omega) - _inspiral_omega_ansatz(theta1, omega)) / 0.0000001 / omega_ring

    tmp = _pack_coeffs(
        t_cut22=t_cut22,
        omega_ring=omega_ring,
        omega_peak=omega_peak,
        domega_peak=0.0,
        amp0=amp0,
        tshift=0.0,
        alpha1rd=alpha1rd,
        alpha2rd=2.0 * math.pi * fdamp_n2,
        alpha21rd=0.0,
        c1=0.0,
        c2=0.0,
        c3=0.0,
        c4=0.0,
        merger_c1=0.0,
        merger_c2=0.0,
        merger_c3=0.0,
        merger_c4=0.0,
        omega_cut_pnamp=0.0,
        phi_cut_pnamp=0.0,
        omega=omega,
        amp_r=amp_r,
        amp_i=amp_i,
        carray=carray,
        cmarray=cmarray,
    )
    domega_peak = -(_rd_omega(0.0000001, tmp) - _rd_omega(0.0, tmp)) / 0.0000001 / omega_ring

    ascut = math.asinh(alpha1rd * t_cut22)
    ascut2 = ascut * ascut
    ascut3 = ascut * ascut2
    ascut4 = ascut2 * ascut2
    ascp = math.asinh(alpha1rd * t_merger22)
    ascp2 = ascp * ascp
    ascp3 = ascp * ascp2
    ascp4 = ascp2 * ascp2
    dencut = math.sqrt(1.0 + t_cut22 * t_cut22 * alpha1rd * alpha1rd)

    mat = np.array(
        [
            [ascut2, ascut3, ascut4],
            [ascp2, ascp3, ascp4],
            [2.0 * alpha1rd * ascut / dencut, 3.0 * alpha1rd * ascut2 / dencut, 4.0 * alpha1rd * ascut3 / dencut],
        ],
        dtype=np.float64,
    )
    rhs = np.array(
        [
            omega_cut_bar - (1.0 - omega_peak / omega_ring) - (domega_peak / alpha1rd) * ascut,
            omega_merger_cp - (1.0 - omega_peak / omega_ring) - (domega_peak / alpha1rd) * ascp,
            domega_cut - domega_peak / dencut,
        ],
        dtype=np.float64,
    )
    cmarray[1:4] = _gsl_lu_solve(mat, rhs)

    amp_insp_cp = np.array(
        [
            inspiral_amp_cp1_22(etapow, spow, dchi, delta),
            inspiral_amp_cp2_22(etapow, spow, dchi, delta),
            inspiral_amp_cp3_22(etapow, spow, dchi, delta),
        ],
        dtype=np.float64,
    )
    amp_merger_cp1 = merger_amp_cp1_22(etapow, spow, dchi, delta)
    amp_peak = peak_amp_22(etapow, spow, dchi, delta)
    amp_rd_c3 = rd_amp_c3_22(etapow, spow)
    tshift = 0.0
    alpha2rd = 2.0 * math.pi * fdamp_n2
    alpha21rd = 0.5 * (alpha2rd - alpha1rd)
    c3 = amp_rd_c3
    c2 = 0.5 * (alpha2rd - alpha1rd)
    tanhc3 = math.tanh(c3)
    if abs(c2) > abs(0.5 * alpha1rd / tanhc3):
        c2 = -0.5 * alpha1rd / tanhc3
    c1 = amp_peak * alpha1rd * math.cosh(c3) * math.cosh(c3) / c2
    c4 = amp_peak - c1 * tanhc3

    tmp = _pack_coeffs(
        t_cut22=t_cut22,
        omega_ring=omega_ring,
        omega_peak=omega_peak,
        domega_peak=domega_peak,
        amp0=amp0,
        tshift=tshift,
        alpha1rd=alpha1rd,
        alpha2rd=alpha2rd,
        alpha21rd=alpha21rd,
        c1=c1,
        c2=c2,
        c3=c3,
        c4=c4,
        merger_c1=0.0,
        merger_c2=0.0,
        merger_c3=0.0,
        merger_c4=0.0,
        omega_cut_pnamp=0.0,
        phi_cut_pnamp=0.0,
        omega=omega,
        amp_r=amp_r,
        amp_i=amp_i,
        carray=carray,
        cmarray=cmarray,
    )

    t_insp = np.array([-2000.0, -250.0, -150.0], dtype=np.float64)
    mat = np.zeros((3, 3), dtype=np.float64)
    rhs = np.zeros(3, dtype=np.float64)
    for idx, t in enumerate(t_insp):
        om = _omega22_py(t, eta, tmp)
        xx = math.pow(0.5 * om, 2.0 / 3.0)
        x4 = xx * xx * xx * xx
        x4half = x4 * math.sqrt(xx)
        x5 = x4 * xx
        amp_offset = _inspiral_amp_complex(xx, tmp).real
        rhs[idx] = (amp_insp_cp[idx] - amp_offset) / (amp0 * xx)
        mat[idx, :] = [x4, x4half, x5]
    amp_r[8:11] = _gsl_lu_solve(mat, rhs)

    tmp[I_AMPR : I_AMPR + 14] = amp_r
    t_cut = T_CUT_AMP
    xx = math.pow(0.5 * _omega22_py(t_cut, eta, tmp), 2.0 / 3.0)
    amp_insp = math.copysign(1.0, _inspiral_amp_complex(xx, tmp).real) * abs(_inspiral_amp_complex(xx, tmp))

    mat = np.zeros((4, 4), dtype=np.float64)
    rhs = np.zeros(4, dtype=np.float64)
    rhs[0] = amp_insp
    sech1 = _sech(alpha1rd * (t_cut - tshift))
    sech2 = _sech(2.0 * alpha1rd * (t_cut - tshift))
    mat[0, :] = [1.0, sech1, math.pow(sech2, 1.0 / 7.0), (t_cut - tshift) * (t_cut - tshift)]
    rhs[1] = amp_merger_cp1
    sech1 = _sech(alpha1rd * (TCP_MERGER - tshift))
    sech2 = _sech(2.0 * alpha1rd * (TCP_MERGER - tshift))
    mat[1, :] = [1.0, sech1, math.pow(sech2, 1.0 / 7.0), (TCP_MERGER - tshift) * (TCP_MERGER - tshift)]
    rhs[2] = amp_peak
    mat[2, :] = [1.0, 1.0, 1.0, 0.0]

    omega2 = _omega22_py(t_cut, eta, tmp)
    omega1 = _omega22_py(t_cut - 0.000001, eta, tmp)
    x1 = math.pow(0.5 * omega1, 2.0 / 3.0)
    x2 = math.pow(0.5 * omega2, 2.0 / 3.0)
    damp_meco = math.copysign(1.0, _inspiral_amp_complex(x2, tmp).real) * (abs(_inspiral_amp_complex(x2, tmp)) - abs(_inspiral_amp_complex(x1, tmp))) / 0.000001
    rhs[3] = damp_meco
    sech1 = _sech(alpha1rd * (t_cut - tshift))
    sech2 = _sech(2.0 * alpha1rd * (t_cut - tshift))
    tanh = math.tanh(alpha1rd * (t_cut - tshift))
    sinh = math.sinh(2.0 * alpha1rd * (t_cut - tshift))
    mat[3, :] = [
        0.0,
        -alpha1rd * sech1 * tanh,
        (-2.0 / 7.0) * alpha1rd * sinh * math.pow(sech2, 8.0 / 7.0),
        2.0 * (t_cut - tshift),
    ]
    merger_c = _gsl_lu_solve(mat, rhs)
    omega_cut_pnamp = -(_complex_amp_orientation(x2, tmp) - _complex_amp_orientation(x1, tmp)) / 0.000001
    phi_cut_pnamp = math.atan2(_inspiral_amp_complex(x2, tmp).imag, _inspiral_amp_complex(x2, tmp).real)
    if math.copysign(1.0, _inspiral_amp_complex(x2, tmp).real) == -1.0:
        phi_cut_pnamp += math.pi

    return _pack_coeffs(
        t_cut22=t_cut22,
        omega_ring=omega_ring,
        omega_peak=omega_peak,
        domega_peak=domega_peak,
        amp0=amp0,
        tshift=tshift,
        alpha1rd=alpha1rd,
        alpha2rd=alpha2rd,
        alpha21rd=alpha21rd,
        c1=c1,
        c2=c2,
        c3=c3,
        c4=c4,
        merger_c1=merger_c[0],
        merger_c2=merger_c[1],
        merger_c3=merger_c[2],
        merger_c4=merger_c[3],
        omega_cut_pnamp=omega_cut_pnamp,
        phi_cut_pnamp=phi_cut_pnamp,
        omega=omega,
        amp_r=amp_r,
        amp_i=amp_i,
        carray=carray,
        cmarray=cmarray,
    )


@njit(cache=True)
def _taylor_t3_jit(theta: float, coeffs: np.ndarray) -> float:
    th = 1.0
    omt = 0.0
    theta3 = 1.0
    theta6 = 1.0
    for i in range(8):
        if i > 0:
            th *= theta
        if i == 3:
            theta3 = th
        if i == 6:
            theta6 = th
        omt += coeffs[I_OMEGA + i] * th
    omt += 107.0 / 280.0 * math.log(2.0 * theta) * theta6
    return omt * theta3 / 4.0


@njit(cache=True)
def _inspiral_omega_ansatz_jit(theta: float, coeffs: np.ndarray) -> float:
    theta8 = math.pow(theta, 8.0)
    theta9 = theta8 * theta
    theta10 = theta9 * theta
    theta11 = theta10 * theta
    theta12 = theta11 * theta
    theta13 = theta12 * theta
    fac = theta * theta * theta / 8.0
    correction = (
        coeffs[I_OMEGA + 8] * theta8
        + coeffs[I_OMEGA + 9] * theta9
        + coeffs[I_OMEGA + 10] * theta10
        + coeffs[I_OMEGA + 11] * theta11
        + coeffs[I_OMEGA + 12] * theta12
        + coeffs[I_OMEGA + 13] * theta13
    )
    return _taylor_t3_jit(theta, coeffs) + 2.0 * fac * correction


@njit(cache=True)
def _omega22_jit(t: float, eta: float, coeffs: np.ndarray) -> float:
    if t < coeffs[I_T_CUT22]:
        theta = math.pow(-eta * t / 5.0, -1.0 / 8.0)
        return _inspiral_omega_ansatz_jit(theta, coeffs)
    if t > 0.0:
        c1 = coeffs[I_CARRAY + 1]
        c2 = coeffs[I_CARRAY + 2]
        c3 = coeffs[I_CARRAY + 3]
        c4 = coeffs[I_CARRAY + 4]
        expc = math.exp(-c2 * t)
        expc2 = expc * expc
        return c1 * (-2.0 * c2 * c4 * expc2 - c2 * c3 * expc) / (1.0 + c4 * expc2 + c3 * expc) + coeffs[I_OMEGA_RING]

    x = math.asinh(coeffs[I_ALPHA1RD] * t)
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    bar = (
        1.0
        - coeffs[I_OMEGA_PEAK] / coeffs[I_OMEGA_RING]
        + (coeffs[I_DOMEGA_PEAK] / coeffs[I_ALPHA1RD]) * x
        + coeffs[I_CMARRAY + 1] * x2
        + coeffs[I_CMARRAY + 2] * x3
        + coeffs[I_CMARRAY + 3] * x4
    )
    return coeffs[I_OMEGA_RING] * (1.0 - bar)


@njit(cache=True)
def _amp22_jit(t: float, x: float, coeffs: np.ndarray) -> complex:
    if t < T_CUT_AMP:
        xhalf = math.sqrt(x)
        xpow = 1.0
        amp_r = 0.0
        amp_i = 0.0
        for i in range(11):
            if i > 0:
                xpow *= xhalf
            amp_r += coeffs[I_AMPR + i] * xpow
            amp_i += coeffs[I_AMPI + i] * xpow
        amp_r -= 428.0 / 105.0 * math.log(16.0 * x) * x * x * x
        return coeffs[I_A0] * x * (amp_r + 1j * amp_i)
    if t > coeffs[I_TSHIFT]:
        tau = t - coeffs[I_TSHIFT]
        amp = math.exp(-coeffs[I_ALPHA1RD] * tau) * (coeffs[I_C1] * math.tanh(coeffs[I_C2] * tau + coeffs[I_C3]) + coeffs[I_C4])
        return amp + 0j
    tau = t - coeffs[I_TSHIFT]
    sech1 = 1.0 / math.cosh(coeffs[I_ALPHA1RD] * tau)
    sech2 = 1.0 / math.cosh(2.0 * coeffs[I_ALPHA1RD] * tau)
    amp = coeffs[I_MERGER_C1] + coeffs[I_MERGER_C2] * sech1 + coeffs[I_MERGER_C3] * math.pow(sech2, 1.0 / 7.0) + coeffs[I_MERGER_C4] * tau * tau
    return amp + 0j


@njit(cache=True)
def _evaluate_tau_array_jit(tau: np.ndarray, eta: float, coeffs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = tau.size
    omega = np.empty(n, dtype=np.float64)
    amp_re = np.empty(n, dtype=np.float64)
    amp_im = np.empty(n, dtype=np.float64)
    amp_abs = np.empty(n, dtype=np.float64)
    for i in range(n):
        om = _omega22_jit(tau[i], eta, coeffs)
        x = math.pow(0.5 * om, 2.0 / 3.0)
        amp = _amp22_jit(tau[i], x, coeffs)
        omega[i] = om
        amp_re[i] = amp.real
        amp_im[i] = amp.imag
        amp_abs[i] = math.sqrt(amp.real * amp.real + amp.imag * amp.imag)
    return omega, amp_re, amp_im, amp_abs


@dataclass(frozen=True)
class IMRPhenomT22:
    """Dominant-mode stripped IMRPhenomT model.

    Parameters ``m1`` and ``m2`` can be supplied in any common mass unit for the
    intrinsic model because the coefficients use only the mass ratio.  Methods
    that convert to seconds or hertz need an explicit total mass in seconds.
    ``coefficient_backend`` records whether the fitted constants came from the
    reference executable, the optional native coefficient dylib, or the pure
    Python coefficient path.
    """

    m1: float
    m2: float
    chi1: float
    chi2: float
    coeffs: np.ndarray
    eta: float
    delta: float
    coefficient_backend: str

    @classmethod
    def from_masses(
        cls,
        m1: float,
        m2: float,
        chi1: float,
        chi2: float,
        *,
        enforce_lal_hierarchy: bool = True,
        coefficient_backend: str = "auto",
        final_spin: float | None = None,
    ) -> "IMRPhenomT22":
        if enforce_lal_hierarchy and m2 > m1:
            m1, m2 = m2, m1
            chi1, chi2 = chi2, chi1
        total_mass = m1 + m2
        eta = m1 * m2 / (total_mass * total_mass)
        delta = (m1 - m2) / total_mass
        backend = coefficient_backend.lower()
        if backend not in {"auto", "native", "python", "reference"}:
            raise ValueError("coefficient_backend must be 'auto', 'native', 'python', or 'reference'")
        if final_spin is not None:
            final_spin = float(final_spin)
            if not math.isfinite(final_spin) or not -1.0 < final_spin < 1.0:
                raise ValueError("final_spin must be finite and strictly between -1 and 1")
            if backend in {"native", "reference"}:
                raise ValueError(
                    "a final-spin override requires coefficient_backend='python' or 'auto'"
                )
            # The native/reference coefficient interfaces expose only the
            # aligned-spin initializer.  Rebuild the complete merger and
            # ringdown fit locally when a precessing remnant is supplied.
            backend = "python"

        if backend in {"auto", "reference"}:
            try:
                coeffs = _reference_executable_coeffs(m1, m2, chi1, chi2)
                return cls(
                    m1=m1,
                    m2=m2,
                    chi1=chi1,
                    chi2=chi2,
                    coeffs=coeffs,
                    eta=eta,
                    delta=delta,
                    coefficient_backend="reference",
                )
            except Exception:
                if backend == "reference":
                    raise

        if backend in {"auto", "native"}:
            try:
                coeffs = _native_model_coeffs(m1, m2, chi1, chi2)
                return cls(
                    m1=m1,
                    m2=m2,
                    chi1=chi1,
                    chi2=chi2,
                    coeffs=coeffs,
                    eta=eta,
                    delta=delta,
                    coefficient_backend="native",
                )
            except Exception:
                if backend == "native":
                    raise

        spin = (m1 * m1 * chi1 + m2 * m2 * chi2) / (m1 * m1 + m2 * m2)
        dchi = chi1 - chi2
        omega, amp_r, amp_i, amp0 = _init_inspiral_t3(m1, m2, chi1, chi2)
        _set_inspiral_fit(eta, dchi, spin, delta, omega)
        coeffs = _set_merger_ringdown_fit(
            m1,
            m2,
            chi1,
            chi2,
            eta,
            delta,
            spin,
            dchi,
            omega,
            amp_r,
            amp_i,
            amp0,
            final_spin_override=final_spin,
        )
        return cls(m1=m1, m2=m2, chi1=chi1, chi2=chi2, coeffs=coeffs, eta=eta, delta=delta, coefficient_backend="python")

    @classmethod
    def from_solar_masses(
        cls,
        m1_solar: float,
        m2_solar: float,
        chi1: float,
        chi2: float,
        *,
        enforce_lal_hierarchy: bool = True,
        coefficient_backend: str = "auto",
        final_spin: float | None = None,
    ) -> "IMRPhenomT22":
        return cls.from_masses(
            m1_solar * TSUN,
            m2_solar * TSUN,
            chi1,
            chi2,
            enforce_lal_hierarchy=enforce_lal_hierarchy,
            coefficient_backend=coefficient_backend,
            final_spin=final_spin,
        )

    @property
    def total_mass(self) -> float:
        return self.m1 + self.m2

    @property
    def t_cut22(self) -> float:
        return float(self.coeffs[I_T_CUT22])

    @property
    def omega_ring(self) -> float:
        return float(self.coeffs[I_OMEGA_RING])

    @property
    def alpha1rd(self) -> float:
        return float(self.coeffs[I_ALPHA1RD])

    @property
    def t_early(self) -> float:
        return -5.0 / (self.eta * math.pow(0.33, 8.0))

    @property
    def tt0(self) -> float:
        spin = (self.m1 * self.m1 * self.chi1 + self.m2 * self.m2 * self.chi2) / (
            self.m1 * self.m1 + self.m2 * self.m2
        )
        dchi = self.chi1 - self.chi2
        etapow, spow = _eta_spin_powers(self.eta, spin)
        return inspiral_taylor_t3_t0(etapow, spow, dchi, self.delta)

    def omega22(self, tau: float) -> float:
        """Dimensionless 2,2 angular frequency at ``tau=(t-tc)/M``."""

        return _omega22_py(float(tau), self.eta, self.coeffs)

    def phase22(self, tau: float) -> float:
        """LAL-style analytic 2,2 phase at dimensionless time ``tau``.

        The production waveform still obtains its smooth phase by integrating
        ``omega22`` on the requested grid.  This antiderivative supplies the
        same absolute phase anchor used by the C implementation.
        """

        return _phase22_py(float(tau), self.eta, self.coeffs)

    def lal_reference_omega22(self, tau: float) -> float:
        """Dimensionless 22 angular frequency used by LAL's tRef root finder.

        This is a compatibility helper, not a replacement for ``omega22``.
        It reproduces LAL's ``GetTimeOfFreq`` convention for choosing the
        phase-reference epoch from ``fmin``/``fRef``.
        """

        return _lal_reference_omega22_py(
            float(tau),
            self.eta,
            self.tt0,
            self.t_early,
            self.coeffs,
        )

    def lal_reference_tau_from_frequency(self, f_ref_hz: float, total_mass_seconds: float | None = None) -> float:
        """Return the LAL-compatible dimensionless ``tau`` for ``f_ref_hz``."""

        if f_ref_hz <= 0.0:
            raise ValueError("f_ref_hz must be positive")
        if total_mass_seconds is None:
            total_mass_seconds = self.total_mass
        if total_mass_seconds <= 0.0:
            raise ValueError("total_mass_seconds must be positive")

        target_omega = 2.0 * math.pi * f_ref_hz * total_mass_seconds
        tt0 = self.tt0
        t_early = self.t_early
        coeffs = self.coeffs
        lo = -1.0e9
        hi = 0.0
        flo = target_omega - _lal_reference_omega22_py(lo, self.eta, tt0, t_early, coeffs)
        fhi = target_omega - _lal_reference_omega22_py(hi, self.eta, tt0, t_early, coeffs)
        if not math.isfinite(flo) or not math.isfinite(fhi):
            raise RuntimeError("non-finite LAL reference root bracket")
        if flo == 0.0:
            return lo
        if fhi == 0.0:
            return hi
        if flo * fhi > 0.0:
            raise RuntimeError("could not bracket LAL reference frequency")

        for _ in range(220):
            mid = 0.5 * (lo + hi)
            fmid = target_omega - _lal_reference_omega22_py(mid, self.eta, tt0, t_early, coeffs)
            if not math.isfinite(fmid):
                raise RuntimeError("non-finite LAL reference root value")
            if fmid == 0.0:
                return mid
            if (flo < 0.0 and fmid < 0.0) or (flo > 0.0 and fmid > 0.0):
                lo = mid
                flo = fmid
            else:
                hi = mid
            if abs(hi - lo) < 1.0e-10:
                break
        return 0.5 * (lo + hi)

    def amplitude_complex(self, tau: float) -> complex:
        """Dimensionless complex 2,2 amplitude at ``tau=(t-tc)/M``."""

        omega = self.omega22(tau)
        x = math.pow(0.5 * omega, 2.0 / 3.0)
        return _amp22_py(float(tau), x, self.coeffs)

    def evaluate_tau(self, tau: Iterable[float] | np.ndarray) -> dict[str, np.ndarray]:
        """Evaluate omega and amplitude on dimensionless times.

        Returns ``omega22``, real/imaginary amplitude, absolute amplitude, and
        GW frequency ``omega22/(2*pi)`` in units of ``1/M``.
        """

        tau_arr = np.asarray(tau, dtype=np.float64)
        if NUMBA_AVAILABLE:
            omega, amp_re, amp_im, amp_abs = _evaluate_tau_array_jit(tau_arr, self.eta, self.coeffs)
        else:
            omega = np.empty_like(tau_arr)
            amp_re = np.empty_like(tau_arr)
            amp_im = np.empty_like(tau_arr)
            amp_abs = np.empty_like(tau_arr)
            for i, value in enumerate(tau_arr):
                omega[i] = self.omega22(float(value))
                amp = self.amplitude_complex(float(value))
                amp_re[i] = amp.real
                amp_im[i] = amp.imag
                amp_abs[i] = abs(amp)
        return {
            "tau": tau_arr,
            "omega22": omega,
            "freq22_M": omega / (2.0 * math.pi),
            "amp_re": amp_re,
            "amp_im": amp_im,
            "amp_abs": amp_abs,
        }

    def evaluate_times(self, times: Iterable[float] | np.ndarray, *, tc: float, total_mass_seconds: float | None = None, distance_gpc: float | None = None) -> dict[str, np.ndarray]:
        """Evaluate on physical times.

        ``times`` and ``tc`` are in seconds.  If ``distance_gpc`` is given, the
        returned ``amp_physical`` uses the same prefactor as ``PhenomT_AP``.
        """

        mass_seconds = self.total_mass if total_mass_seconds is None else total_mass_seconds
        times_arr = np.asarray(times, dtype=np.float64)
        tau = (times_arr - tc) / mass_seconds
        values = self.evaluate_tau(tau)
        values["time"] = times_arr
        values["omega22_s"] = values["omega22"] / mass_seconds
        values["freq22_hz"] = values["omega22_s"] / (2.0 * math.pi)
        if distance_gpc is not None:
            prefactor = math.sqrt(2.0) * self.eta * mass_seconds / (distance_gpc * GPSEC)
            values["amp_physical"] = prefactor * values["amp_abs"]
        return values

    def phase_from_times(self, times: Iterable[float] | np.ndarray, *, tc: float, phi_c: float = 0.0, total_mass_seconds: float | None = None) -> np.ndarray:
        """Integrate omega on ``times`` and set ``phase(tc)=phi_c``.

        This mirrors the convention in ``PhenomT_AP``.  SciPy's natural cubic
        spline is used to match GSL's ``gsl_interp_cspline`` as closely as the
        Python stack permits.
        """

        from scipy.interpolate import CubicSpline

        mass_seconds = self.total_mass if total_mass_seconds is None else total_mass_seconds
        times_arr = np.asarray(times, dtype=np.float64)
        omega_s = self.evaluate_times(times_arr, tc=tc, total_mass_seconds=mass_seconds)["omega22_s"]
        spline = CubicSpline(times_arr, omega_s, bc_type="natural")
        phase = np.empty_like(times_arr)
        phase[0] = 0.0
        for i in range(1, len(times_arr)):
            phase[i] = phase[i - 1] + spline.integrate(times_arr[i - 1], times_arr[i])
        phase_at_tc = spline.integrate(times_arr[0], tc)
        return phase + (phi_c - phase_at_tc)


__all__ = [
    "GPSEC",
    "NUMBA_AVAILABLE",
    "TSUN",
    "IMRPhenomT22",
    "final_mass_2017",
    "final_spin_2017",
    "qnm_fring22",
    "qnm_fdamp22",
    "qnm_fdamp22n2",
]
