#!/usr/bin/env python3
# Based on LALSimulation IMRPhenomTPHM: Copyright (C) 2020 Hector Estelles.
# LAL-free Python port and changes: Copyright (C) 2026 Neil Cornish.
# SPDX-License-Identifier: GPL-2.0-or-later
# Distributed without warranty; see the GNU GPL for details.
"""Pure Python/Numba numerical IMRPhenomTPHM waveform model.

The model twists the five folded IMRPhenomTHM carriers with a single numerical
precession trajectory.  It exposes both the summed observer strain and the
per-parent-carrier complex quadratures needed by the partitioned-FFT TDI path.
No LAL runtime dependency is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.integrate import quad

from phenomt22 import GPSEC, TSUN
from phenomthm import DEFAULT_POSITIVE_MODES, IMRPhenomTHM, ModeSeries
from phenomtphm_precession import (
    PrecessionConfig,
    PrecessionEvolution,
    PrecessionSummary,
    PrecessionTrajectory,
    evolve_euler_angles,
)
from phenomtphm_rotation import (
    build_projection,
    project_folded_pair,
    reference_adjoint_rotation,
    rotate_multipole_series,
)


@lru_cache(maxsize=4)
def _phase_quadrature_rule(order):
    return np.polynomial.legendre.leggauss(order)


@dataclass(frozen=True)
class TPHMCarrierSeries:
    ell: int
    abs_m: int
    tau: np.ndarray
    strain: np.ndarray
    quadrature: np.ndarray
    amplitude: np.ndarray
    phase: np.ndarray
    omega: np.ndarray
    coprecessing_mode: np.ndarray


@dataclass(frozen=True)
class TPHMWaveform:
    tau: np.ndarray
    strain: np.ndarray
    strain_quadrature: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    gamma: np.ndarray
    carriers: Mapping[tuple[int, int], TPHMCarrierSeries]
    j_modes: Mapping[tuple[int, int], np.ndarray] | None = None
    l0_modes: Mapping[tuple[int, int], np.ndarray] | None = None

    @property
    def hplus(self) -> np.ndarray:
        return self.strain.real

    @property
    def hcross(self) -> np.ndarray:
        return -self.strain.imag

    @property
    def analytic_hplus(self) -> np.ndarray:
        return self.strain.real - 1j * self.strain_quadrature.real

    @property
    def analytic_hcross(self) -> np.ndarray:
        return -self.strain.imag + 1j * self.strain_quadrature.imag


def _canonical_carriers(
    modes: Sequence[tuple[int, int]] | str,
) -> tuple[tuple[int, int], ...]:
    requested = DEFAULT_POSITIVE_MODES if modes == "default" else tuple(modes)
    output: list[tuple[int, int]] = []
    for ell, emm in requested:
        key = (int(ell), abs(int(emm)))
        if key not in DEFAULT_POSITIVE_MODES:
            raise ValueError(f"unsupported TPHM parent carrier {key}")
        if key not in output:
            output.append(key)
    if not output:
        raise ValueError("at least one TPHM carrier is required")
    return tuple(output)


class IMRPhenomTPHM:
    """Numerical-precession twist-up of the LAL-free IMRPhenomTHM model.

    The default ``reconstruction='consistent'`` uses the evolved remnant spin
    throughout every merger/ringdown coefficient.  The C implementation keeps
    a separate LAL-compatible policy for diagnostic comparisons; its known
    cadence and mixed-remnant conventions are intentionally not made the
    Python default.
    """

    def __init__(
        self,
        m1_seconds: float,
        m2_seconds: float,
        chi1: Sequence[float],
        chi2: Sequence[float],
        *,
        tau_ref: float,
        modes: Sequence[tuple[int, int]] | str = "default",
        coefficient_backend: str = "auto",
        absolute_tolerance: float = 1.0e-11,
        relative_tolerance: float = 1.0e-11,
        initial_step: float = 1.0,
        reconstruction: str = "consistent",
    ):
        if reconstruction != "consistent":
            raise ValueError(
                "the Python port currently exposes only reconstruction='consistent'"
            )
        spin1 = tuple(float(value) for value in chi1)
        spin2 = tuple(float(value) for value in chi2)
        if len(spin1) != 3 or len(spin2) != 3:
            raise ValueError("chi1 and chi2 must be three-component spin vectors")
        m1 = float(m1_seconds)
        m2 = float(m2_seconds)
        if m1 <= 0.0 or m2 <= 0.0:
            raise ValueError("component masses must be positive")
        if m2 > m1:
            m1, m2 = m2, m1
            spin1, spin2 = spin2, spin1

        self.m1 = m1
        self.m2 = m2
        self.total_mass = m1 + m2
        self.chi1 = spin1
        self.chi2 = spin2
        self.carrier_modes = _canonical_carriers(modes)
        self.reconstruction = reconstruction
        self.precession_config = PrecessionConfig.from_spins(
            spin1,
            spin2,
            tau_ref,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            initial_step=initial_step,
        )

        # Keep the original aligned-spin 2,2 model as the PN precession driver.
        # Feeding the reconstructed ringdown back into this ODE would make the
        # precession evolution depend circularly on its own remnant estimate.
        self.precession_driver = IMRPhenomTHM(
            m1,
            m2,
            spin1[2],
            spin2[2],
            modes=((2, 2),),
            coefficient_backend=coefficient_backend,
        )
        summary_grid = (
            np.asarray((tau_ref, 0.0), dtype=np.float64)
            if tau_ref < 0.0
            else np.asarray((0.0,), dtype=np.float64)
        )
        self._precession_evolution = PrecessionEvolution(self.precession_driver, self.precession_config)
        initial_trajectory = self._precession_evolution.evaluate(summary_grid)
        self.precession_summary = initial_trajectory.summary

        # Rebuild the entire THM merger/ringdown with the evolved remnant spin.
        # The final-spin override selects the local coefficient path because
        # the legacy native coefficient interface has no such argument.
        self.carrier = IMRPhenomTHM(
            m1,
            m2,
            spin1[2],
            spin2[2],
            modes=self.carrier_modes,
            coefficient_backend=coefficient_backend,
            final_spin=self.precession_summary.final_spin,
        )
        self._reference_rotations = {
            ell: reference_adjoint_rotation(ell, self.precession_summary)
            for ell in sorted({mode[0] for mode in self.carrier_modes})
        }
        self.backend_name = "local-tphm"

    @classmethod
    def from_solar_masses(
        cls,
        m1_solar: float,
        m2_solar: float,
        chi1: Sequence[float],
        chi2: Sequence[float],
        **kwargs,
    ) -> "IMRPhenomTPHM":
        return cls(
            m1_solar * TSUN,
            m2_solar * TSUN,
            chi1,
            chi2,
            **kwargs,
        )

    @property
    def final_spin(self) -> float:
        return self.precession_summary.final_spin

    def euler_angles(self, tau: Iterable[float] | np.ndarray) -> PrecessionTrajectory:
        return self._precession_evolution.evaluate(tau)

    def precession_nodes(self, start: float, stop: float = 0.0) -> np.ndarray:
        """Accepted spin integration mesh plus interval bounds, in mass units."""
        return self._precession_evolution.integration_nodes(start, stop)

    def phase_from_anchor(
        self, tau: float, tau_anchor: float, phi22_anchor: float
    ) -> float:
        """Integrate the 2,2 carrier phase from a persistent model anchor."""

        if tau == tau_anchor:
            return float(phi22_anchor)
        integral, _error = quad(
            self.carrier.mode22.omega22,
            float(tau_anchor),
            float(tau),
            epsabs=1.0e-10,
            epsrel=1.0e-11,
            limit=2048,
            points=[value for value in (self.carrier.mode22.t_cut22, 0.0) if min(tau, tau_anchor) < value < max(tau, tau_anchor)],
        )
        return float(phi22_anchor) + integral

    def _phase22_quadrature_grid(self, tau, phi0, order=8):
        """Integrate frequency only; do not add twisted-waveform samples.

        Gauss--Legendre quadrature on a frequency-resolved physical grid,
        with exact 22 frequency joins inserted internally. This is not an
        accuracy guarantee for an arbitrary unresolved caller grid.
        """
        if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or not 2 <= order <= 32:
            raise ValueError("phase quadrature order must be an integer in [2, 32]")
        tau = np.asarray(tau, dtype=float)
        if tau.ndim != 1 or tau.size < 2 or not np.all(np.isfinite(tau)) or np.any(np.diff(tau) <= 0):
            raise ValueError("phase quadrature requires a finite strictly increasing grid")
        joins = np.array([self.carrier.mode22.t_cut22, 0.])
        grid = np.unique(np.r_[tau, joins[(joins > tau[0]) & (joins < tau[-1])]])
        x, weights = _phase_quadrature_rule(int(order))
        halfwidth = .5*np.diff(grid)
        middle = .5*(grid[1:]+grid[:-1])
        queries = middle[:, None]+halfwidth[:, None]*x
        omega = self.carrier.mode22.evaluate_tau(queries.ravel())["omega22"].reshape(queries.shape)
        phase = phi0+np.r_[0., np.cumsum(halfwidth*(omega @ weights))]
        return np.ascontiguousarray(phase[np.searchsorted(grid, tau)])

    def evaluate_tau(
        self,
        tau: Iterable[float] | np.ndarray,
        *,
        phi22: Iterable[float] | np.ndarray | None = None,
        phi0: float = 0.0,
        phi22_at_tcut: float | None = None,
        observer_theta: float,
        observer_phi: float,
        polarization: float,
        return_modes: bool = False,
    ) -> TPHMWaveform:
        tau_array = np.ascontiguousarray(np.asarray(tau, dtype=np.float64))
        if tau_array.ndim != 1 or tau_array.size == 0:
            raise ValueError("tau must be a nonempty one-dimensional array")
        if np.any(np.diff(tau_array) <= 0.0):
            raise ValueError("tau must be strictly increasing")

        trajectory = self.euler_angles(tau_array)
        coprecessing = self.carrier.evaluate_tau(
            tau_array,
            phi22=phi22,
            phi0=phi0,
            phi22_at_tcut=phi22_at_tcut,
        )
        projections = {
            ell: build_projection(
                rotation,
                ell,
                observer_theta,
                observer_phi,
                polarization,
            )
            for ell, rotation in self._reference_rotations.items()
        }

        total = np.zeros(tau_array.size, dtype=np.complex128)
        total_quadrature = np.zeros_like(total)
        carriers: dict[tuple[int, int], TPHMCarrierSeries] = {}
        for ell, abs_m in self.carrier_modes:
            series = coprecessing[(ell, abs_m)]
            positive_mode = np.ascontiguousarray(series.hlm)
            strain, quadrature = project_folded_pair(
                ell,
                abs_m,
                trajectory.alpha,
                trajectory.beta,
                trajectory.gamma,
                projections[ell],
                positive_mode,
            )
            key = (ell, abs_m)
            carriers[key] = TPHMCarrierSeries(
                ell=ell,
                abs_m=abs_m,
                tau=tau_array,
                strain=strain,
                quadrature=quadrature,
                amplitude=series.amplitude,
                phase=series.phase,
                omega=series.omega,
                coprecessing_mode=positive_mode,
            )
            total += strain
            total_quadrature += quadrature

        j_modes = None
        l0_modes = None
        if return_modes:
            j_modes = {}
            l0_modes = {}
            by_ell: dict[int, list[tuple[int, ModeSeries]]] = {}
            for (ell, abs_m), series in coprecessing.items():
                by_ell.setdefault(ell, []).append((abs_m, series))
            for ell, members in by_ell.items():
                multipole = np.zeros(
                    (tau_array.size, 2 * ell + 1), dtype=np.complex128
                )
                parity = -1.0 if ell & 1 else 1.0
                for abs_m, series in members:
                    positive = series.hlm
                    multipole[:, abs_m + ell] = positive
                    multipole[:, -abs_m + ell] = parity * np.conj(positive)
                j_series = rotate_multipole_series(
                    ell,
                    trajectory.alpha,
                    trajectory.beta,
                    trajectory.gamma,
                    multipole,
                )
                l0_series = np.einsum(
                    "ab,ib->ia", self._reference_rotations[ell], j_series
                )
                for offset, emm in enumerate(range(-ell, ell + 1)):
                    j_modes[(ell, emm)] = j_series[:, offset]
                    l0_modes[(ell, emm)] = l0_series[:, offset]

        return TPHMWaveform(
            tau=tau_array,
            strain=total,
            strain_quadrature=total_quadrature,
            alpha=trajectory.alpha,
            beta=trajectory.beta,
            gamma=trajectory.gamma,
            carriers=carriers,
            j_modes=j_modes,
            l0_modes=l0_modes,
        )

    def evaluate_times(
        self,
        times: Iterable[float] | np.ndarray,
        *,
        tc: float,
        phic: float = 0.0,
        observer_theta: float,
        observer_phi: float,
        polarization: float,
        distance_gpc: float | None = None,
        return_modes: bool = False,
        phase_quadrature_order: int | None = None,
    ) -> TPHMWaveform:
        """Evaluate source times, optionally integrating frequency by quadrature.

        None preserves the original caller-grid natural-spline integration.
        The physical LIGO grid uses order 8, without adding waveform samples.
        """
        times_array = np.ascontiguousarray(np.asarray(times, dtype=np.float64))
        tau = (times_array - tc) / self.total_mass
        phi_start = (
            self.carrier.mode22.phase22(float(tau[0]))
            + phic
            - self.carrier.mode22.phase22(0.0)
        )
        phi22 = (self.carrier._phase22_grid(tau, phi_start) if phase_quadrature_order is None else
                 self._phase22_quadrature_grid(tau, phi_start, phase_quadrature_order))
        phi_cut = self.phase_from_anchor(
            -150.0, float(tau[0]), float(phi22[0])
        )
        result = self.evaluate_tau(
            tau,
            phi22=phi22,
            phi22_at_tcut=phi_cut,
            observer_theta=observer_theta,
            observer_phi=observer_phi,
            polarization=polarization,
            return_modes=return_modes,
        )
        scale = (
            1.0
            if distance_gpc is None
            else math.sqrt(2.0)
            * self.carrier.eta
            * self.total_mass
            / (distance_gpc * GPSEC)
        )
        if scale == 1.0:
            return result
        carriers = {
            key: TPHMCarrierSeries(
                ell=value.ell,
                abs_m=value.abs_m,
                tau=times_array,
                strain=scale * value.strain,
                quadrature=scale * value.quadrature,
                amplitude=scale * value.amplitude,
                phase=value.phase,
                omega=value.omega / self.total_mass,
                coprecessing_mode=scale * value.coprecessing_mode,
            )
            for key, value in result.carriers.items()
        }
        j_modes = (
            None
            if result.j_modes is None
            else {key: scale * value for key, value in result.j_modes.items()}
        )
        l0_modes = (
            None
            if result.l0_modes is None
            else {key: scale * value for key, value in result.l0_modes.items()}
        )
        return TPHMWaveform(
            tau=times_array,
            strain=scale * result.strain,
            strain_quadrature=scale * result.strain_quadrature,
            alpha=result.alpha,
            beta=result.beta,
            gamma=result.gamma,
            carriers=carriers,
            j_modes=j_modes,
            l0_modes=l0_modes,
        )


__all__ = [
    "IMRPhenomTPHM",
    "PrecessionConfig",
    "PrecessionSummary",
    "PrecessionTrajectory",
    "TPHMCarrierSeries",
    "TPHMWaveform",
]
