#!/usr/bin/env python3
# Galactic-binary LISA response and WDM implementation by Neil Cornish.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fast sparse LISA TDI+WDM waveforms for circular Galactic binaries.

The source is exposed as one slowly evolving amplitude/phase/frequency
carrier, matching the carrier interface used by the FEW and IMRPhenomT
drivers.  It reuses the common constellation, delayed TDI response, AP
extraction, Meyer window, heterodyne, and sparse WDM containers in
``phenomt_tdi_wdm.py``.  Since a circular Galactic binary occupies one narrow
band for the full observation, its natural block plan is a single real
heterodyned FFT rather than the more general complex multi-block FEW path.
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
from scipy.interpolate import CubicSpline

from phenomt_tdi_wdm import (
    DEFAULT_UCB_FREQUENCY_HZ,
    TDIWDMResult,
    UCBSourceParams,
    WDMChannel,
    WDMShape,
    build_constellation_splines,
    generate_ucb_tdi_wdm,
)


@dataclass(frozen=True)
class GalacticBinaryCarrier:
    """One circular-binary carrier in amplitude/frequency/phase form."""

    source: UCBSourceParams

    def evaluate(
        self, times: np.ndarray | float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = np.asarray(times, dtype=np.float64)
        return self.source.evaluate(values)

    def phase(self, times: np.ndarray | float) -> np.ndarray:
        return self.evaluate(times)[0]

    def amplitude(self, times: np.ndarray | float) -> np.ndarray:
        return self.evaluate(times)[1]

    def frequency(self, times: np.ndarray | float) -> np.ndarray:
        return self.evaluate(times)[2]

    def polarizations(
        self, times: np.ndarray | float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return hplus, hcross and their carrier quadratures.

        These are the same four real series consumed by the shared delayed-TDI
        construction.  Keeping the quadratures explicit avoids differentiating
        or numerically phase-shifting the rapidly varying carrier.
        """

        phase, amplitude, _ = self.evaluate(times)
        cp = np.cos(phase)
        sp = np.sin(phase)
        cosi = self.source.cos_inclination
        aplus = 0.5 * (1.0 + cosi * cosi)
        across = -cosi
        cos2psi = math.cos(2.0 * self.source.polarization)
        sin2psi = math.sin(2.0 * self.source.polarization)
        hp = amplitude * (aplus * cos2psi * cp + across * sin2psi * sp)
        hc = amplitude * (across * cos2psi * sp - aplus * sin2psi * cp)
        hpf = amplitude * (-aplus * cos2psi * sp + across * sin2psi * cp)
        hcf = amplitude * (across * cos2psi * cp + aplus * sin2psi * sp)
        return hp, hc, hpf, hcf


ConstellationData = tuple[
    np.ndarray,
    list[CubicSpline],
    list[CubicSpline],
    list[CubicSpline],
]


@dataclass(frozen=True)
class GalacticBinaryContext:
    """Observation-dependent state reused by sequential waveform calls."""

    shape: WDMShape
    constellation: ConstellationData
    setup_seconds: float

    @classmethod
    def build(cls, shape: WDMShape = WDMShape()) -> "GalacticBinaryContext":
        start = time.perf_counter()
        constellation = build_constellation_splines(shape)
        return cls(shape, constellation, time.perf_counter() - start)

    def waveform(
        self,
        source: UCBSourceParams = UCBSourceParams(),
        channels: Iterable[str] = ("X", "Y", "Z"),
        *,
        compute_wdm: bool = True,
        wdm_method: str = "fft",
        tdi_generation: int = 1,
    ) -> TDIWDMResult:
        return generate_ucb_tdi_wdm(
            source=source,
            shape=self.shape,
            channels=channels,
            compute_wdm=compute_wdm,
            wdm_method=wdm_method,
            constellation_data=self.constellation,
            tdi_generation=tdi_generation,
        )


@dataclass(frozen=True)
class SparseComparison:
    channel: str
    reference_pixels: int
    model_pixels: int
    union_pixels: int
    match: float
    power_ratio: float
    relative_l2: float


@dataclass(frozen=True)
class GalacticBinaryRun:
    carrier: GalacticBinaryCarrier
    waveform: TDIWDMResult
    elapsed_seconds: float
    comparisons: tuple[SparseComparison, ...] = ()


def _sparse_union_values(
    reference_n: np.ndarray,
    reference_m: np.ndarray,
    reference_values: np.ndarray,
    model: WDMChannel,
    nf: int,
) -> tuple[np.ndarray, np.ndarray]:
    stride = nf + 1
    reference_keys = reference_n.astype(np.int64) * stride + reference_m.astype(np.int64)
    model_keys = model.listn.astype(np.int64) * stride + model.listm.astype(np.int64)
    union = np.union1d(reference_keys, model_keys)
    reference = np.zeros(union.size, dtype=np.float64)
    candidate = np.zeros(union.size, dtype=np.float64)
    reference[np.searchsorted(union, reference_keys)] = reference_values
    candidate[np.searchsorted(union, model_keys)] = model.values
    return reference, candidate


def compare_c_reference(
    waveform: TDIWDMResult,
    directory: Path | str,
) -> tuple[SparseComparison, ...]:
    """Compare against `response --sparse-hetF` skinny C products."""

    root = Path(directory)
    comparisons: list[SparseComparison] = []
    for channel, model in waveform.channels.items():
        path = root / f"ucb_hetF_{channel}_track_pixels.dat"
        data = np.loadtxt(path, comments="#", ndmin=2)
        if data.shape[1] < 3:
            raise ValueError(f"{path} must contain n, m, value columns")
        reference, candidate = _sparse_union_values(
            data[:, 0], data[:, 1], data[:, 2], model, waveform.shape.nf
        )
        reference_power = float(np.dot(reference, reference))
        candidate_power = float(np.dot(candidate, candidate))
        denominator = math.sqrt(reference_power * candidate_power)
        match = float(np.dot(reference, candidate) / denominator)
        comparisons.append(
            SparseComparison(
                channel=channel,
                reference_pixels=int(data.shape[0]),
                model_pixels=int(model.values.size),
                union_pixels=int(reference.size),
                match=match,
                power_ratio=candidate_power / reference_power,
                relative_l2=float(
                    np.linalg.norm(candidate - reference) / math.sqrt(reference_power)
                ),
            )
        )
    return tuple(comparisons)


def generate(
    source: UCBSourceParams,
    context: GalacticBinaryContext,
    channels: Iterable[str] = ("X", "Y", "Z"),
    *,
    c_reference_dir: Path | str | None = None,
    wdm_method: str = "fft",
    tdi_generation: int = 1,
) -> GalacticBinaryRun:
    carrier = GalacticBinaryCarrier(source)
    start = time.perf_counter()
    waveform = context.waveform(source, channels, wdm_method=wdm_method,
                                tdi_generation=tdi_generation)
    elapsed = time.perf_counter() - start
    comparisons = (
        compare_c_reference(waveform, c_reference_dir)
        if c_reference_dir is not None
        else ()
    )
    return GalacticBinaryRun(carrier, waveform, elapsed, comparisons)


def _write_channel(path: Path, channel: WDMChannel) -> None:
    order = np.lexsort((channel.listm, channel.listn))
    data = np.column_stack(
        (channel.listn[order], channel.listm[order], channel.values[order])
    )
    np.savetxt(path, data, fmt=("%d", "%d", "%.15e"), header="n m wdm")


def _write_xyz(path: Path, waveform: TDIWDMResult) -> None:
    stride = waveform.shape.nf + 1
    keys = np.empty(0, dtype=np.int64)
    for channel in waveform.channels.values():
        channel_keys = channel.listn.astype(np.int64) * stride + channel.listm
        keys = np.union1d(keys, channel_keys)
    values = np.zeros((keys.size, 3), dtype=np.float64)
    for index, name in enumerate(("X", "Y", "Z")):
        channel = waveform.channels.get(name)
        if channel is None:
            continue
        channel_keys = channel.listn.astype(np.int64) * stride + channel.listm
        values[np.searchsorted(keys, channel_keys), index] = channel.values
    data = np.column_stack((keys // stride, keys % stride, values))
    np.savetxt(
        path,
        data,
        fmt=("%d", "%d", "%.15e", "%.15e", "%.15e"),
        header="n m X Y Z",
    )


def write_run(run: GalacticBinaryRun, output: Path | str, diagnostics: bool) -> None:
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    waveform = run.waveform
    rows: list[tuple[str, str]] = [
        ("source_model", "circular_galactic_binary"),
        ("wdm_normalization", "wd_viafreq_sqrt_8_over_15"),
        ("nf", str(waveform.shape.nf)),
        ("nt", str(waveform.shape.nt)),
        ("dt_seconds", f"{waveform.shape.dt:.15e}"),
        ("Tobs_seconds", f"{waveform.shape.Tobs:.15e}"),
        ("tdi_samples", str(waveform.tdi.time.size)),
        ("elapsed_seconds", f"{run.elapsed_seconds:.15e}"),
    ]
    rows.extend((f"seconds_{key}", f"{value:.15e}") for key, value in waveform.timings.items())
    for name, channel in waveform.channels.items():
        rows.append((f"{name}_frequency_samples", str(channel.freq.size)))
        rows.append((f"{name}_active_pixels", str(channel.values.size)))
    for comparison in run.comparisons:
        prefix = f"c_reference_{comparison.channel}"
        rows.extend(
            (
                (f"{prefix}_match", f"{comparison.match:.15e}"),
                (f"{prefix}_mismatch", f"{1.0 - comparison.match:.15e}"),
                (f"{prefix}_power_ratio", f"{comparison.power_ratio:.15e}"),
                (f"{prefix}_relative_l2", f"{comparison.relative_l2:.15e}"),
            )
        )
    with (root / "summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("quantity", "value"))
        writer.writerows(rows)

    for name, channel in waveform.channels.items():
        _write_channel(root / f"track_pixels_{name}.dat", channel)
    _write_xyz(root / "track_pixels_XYZ.dat", waveform)

    if diagnostics:
        intrinsic = waveform.intrinsic
        np.savetxt(
            root / "intrinsic_ap.dat",
            np.column_stack(
                (
                    intrinsic.bary_time,
                    intrinsic.detector_time,
                    intrinsic.amplitude,
                    intrinsic.frequency,
                    intrinsic.phase,
                )
            ),
            fmt="%.15e",
            header="source_time detector_time amplitude frequency_hz phase",
        )
        columns: list[np.ndarray] = [waveform.tdi.time, waveform.tdi.detector_time]
        labels = ["detector_time", "reference_source_time"]
        for name in waveform.channels:
            columns.extend(
                (
                    waveform.tdi.amplitude[name],
                    waveform.tdi.phase_offset[name],
                )
            )
            labels.extend((f"amplitude_{name}", f"phase_offset_{name}"))
        np.savetxt(
            root / "tdi_sparse_ap.dat",
            np.column_stack(columns),
            fmt="%.15e",
            header=" ".join(labels),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency", type=float, default=DEFAULT_UCB_FREQUENCY_HZ)
    parser.add_argument("--fdot", type=float)
    parser.add_argument("--fddot", type=float)
    parser.add_argument("--amplitude", type=float)
    parser.add_argument("--m1-solar", type=float, default=0.6)
    parser.add_argument("--m2-solar", type=float, default=0.7)
    parser.add_argument("--distance-kpc", type=float, default=1.0)
    parser.add_argument("--costh", type=float, default=-0.783326909627)
    parser.add_argument("--longitude", type=float, default=3.0)
    parser.add_argument("--cosi", type=float, default=0.0707372016677)
    parser.add_argument("--psi", type=float, default=0.8)
    parser.add_argument("--phi0", type=float, default=1.2)
    parser.add_argument("--nf", type=int, default=4096)
    parser.add_argument("--nt", type=int, default=4096)
    parser.add_argument("--dt", type=float, default=1.875)
    parser.add_argument("--tdi-generation", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--channel",
        action="append",
        choices=("X", "Y", "Z"),
        help="repeat to select channels; defaults to X,Y,Z",
    )
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repeat-runs", type=int, default=3)
    parser.add_argument(
        "--wdm-method",
        choices=("fft", "lookup"),
        default="fft",
        help="WDM evaluator; lookup is an experimental signed-fdot chirplet table",
    )
    parser.add_argument(
        "--c-reference-dir",
        type=Path,
        help="directory containing response --sparse-hetF outputs",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("galactic_binary_validation"))
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.warmup_runs < 0 or args.repeat_runs < 1:
        raise SystemExit("warmup runs must be non-negative and repeat runs positive")
    if not -1.0 <= args.costh <= 1.0 or not -1.0 <= args.cosi <= 1.0:
        raise SystemExit("costh and cosi must lie in [-1,1]")

    shape = WDMShape(nf=args.nf, nt=args.nt, dt=args.dt)
    source = UCBSourceParams(
        frequency_hz=args.frequency,
        ecliptic_costheta=args.costh,
        ecliptic_longitude=args.longitude,
        amplitude=args.amplitude,
        cos_inclination=args.cosi,
        polarization=args.psi,
        phi0=args.phi0,
        fdot=args.fdot,
        fddot=args.fddot,
        m1_solar=args.m1_solar,
        m2_solar=args.m2_solar,
        distance_kpc=args.distance_kpc,
    )
    channels = tuple(args.channel) if args.channel else ("X", "Y", "Z")
    context = GalacticBinaryContext.build(shape)
    for _ in range(args.warmup_runs):
        generate(source, context, channels, wdm_method=args.wdm_method,
                 tdi_generation=args.tdi_generation)
    trials = [
        generate(
            source,
            context,
            channels,
            c_reference_dir=args.c_reference_dir,
            wdm_method=args.wdm_method,
            tdi_generation=args.tdi_generation,
        )
        for _ in range(args.repeat_runs)
    ]
    run = min(trials, key=lambda item: item.elapsed_seconds)

    print("source_model circular_galactic_binary")
    print(
        "wdm_method "
        + (
            "single_real_heterodyned_fft"
            if args.wdm_method == "fft"
            else "signed_fdot_chirplet_lookup"
        )
    )
    print("wdm_normalization wd_viafreq_sqrt_8_over_15")
    print(
        f"wdm_grid nt {shape.nt} nf {shape.nf} dt {shape.dt:.15e} "
        f"Tobs {shape.Tobs:.15e}"
    )
    print(f"context_setup_seconds {context.setup_seconds:.6f}")
    print(f"tdi_samples {run.waveform.tdi.time.size}")
    for name, channel in run.waveform.channels.items():
        print(
            f"channel {name} frequency_samples {channel.freq.size} "
            f"active_pixels {channel.values.size}"
        )
    for key, value in run.waveform.timings.items():
        print(f"seconds_{key} {value:.6f}")
    print(f"seconds_warm_total {run.elapsed_seconds:.6f}")
    for comparison in run.comparisons:
        print(
            f"c_reference {comparison.channel} match {comparison.match:.15e} "
            f"mismatch {1.0 - comparison.match:.6e} "
            f"power_ratio {comparison.power_ratio:.12e} "
            f"relative_l2 {comparison.relative_l2:.6e}"
        )
    if not args.no_write:
        write_run(run, args.output_dir, args.diagnostics)
        print(f"output_dir {args.output_dir}")


if __name__ == "__main__":
    main()
