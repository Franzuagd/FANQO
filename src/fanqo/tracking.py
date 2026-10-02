"""Single-pass physical tracking cache for invariant studies.

Tracking is deliberately separated from invariant construction.  The expensive
Accelerator Toolbox pass is performed once and stored as a memory-mapped data
set.  Any number of invariant constructions can then be evaluated and plotted
without tracking the particles again.

The cache also stores a small FMA summary from the same turn-by-turn data.  FMA
is data only here; this branch intentionally keeps plotting limited to invariant
quality maps and pairwise comparison maps.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import warnings
from pathlib import Path

import numpy as np

from .core import linear as lin
from .core import nonlinear as nl


FMA_VERSION = 2


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _infer_ring_cells(lattice_cfg, parameters):
    magnets = lattice_cfg.define_magnets(parameters)
    by_name = {lin.magnet_field(e, "NAME"): e for e in magnets}
    bend = 0.0
    for name in lattice_cfg.CELL_NAMES:
        elem = by_name[name]
        if lin.magnet_field(elem, "TYPE") == "bending":
            bend += float(lin.magnet_field(elem, "ANGLE"))

    if abs(bend) < 1e-12:
        raise ValueError("Configured cell has zero net bend.")

    cells_float = 360.0 / abs(bend)
    cells = int(round(cells_float))
    if cells < 1 or not np.isclose(cells_float, cells, rtol=0.0, atol=1e-7):
        raise ValueError(
            f"Cell bend {bend:.12g} deg does not make an integer 360-degree ring."
        )
    return cells, bend


def _prepare_physical_ring(config, lattice_cfg, context):
    configured = getattr(config, "TRACKING_PHYSICAL_RING_CELLS", None)
    inferred, cell_bend = _infer_ring_cells(lattice_cfg, context["parameters"])
    n_cells = inferred if configured is None else int(configured)
    if n_cells < 1:
        raise ValueError("TRACKING_PHYSICAL_RING_CELLS must be positive.")

    ring_names = list(lattice_cfg.CELL_NAMES) * n_cells
    magnets, lattice, data, correction, parameters = lin.prepare_lattice(
        parameters=dict(context["parameters"]),
        ring_names=ring_names,
        magnet_builder=lattice_cfg.define_magnets,
        energy_parameter=lattice_cfg.ENERGY_PARAMETER,
        correction_parameter_map=lattice_cfg.CORRECTION_PARAMETER_MAP,
        correct_chromatic=bool(config.CORRECT_CHROMATICITY),
        family1=lattice_cfg.CHROMATIC_FAMILY1,
        family2=lattice_cfg.CHROMATIC_FAMILY2,
        target_chrom_x=lattice_cfg.TARGET_CHROM_X,
        target_chrom_y=lattice_cfg.TARGET_CHROM_Y,
        repetitions=1,
        step=lattice_cfg.STEP,
    )

    total_bend = sum(
        float(lin.magnet_field(e, "ANGLE"))
        for e in lattice
        if lin.magnet_field(e, "TYPE") == "bending"
    )
    return {
        "magnets": magnets,
        "lattice": lattice,
        "data": data,
        "correction": correction,
        "parameters": parameters,
        "n_cells": n_cells,
        "cell_bend_deg": cell_bend,
        "total_bend_deg": total_bend,
    }


def _at_element(elem, at, nsteps):
    field = lin.magnet_field
    name = str(field(elem, "NAME"))
    typ = str(field(elem, "TYPE")).lower()
    L = float(field(elem, "LENGTH"))
    angle = float(field(elem, "ANGLE"))
    K = float(field(elem, "K"))
    S = float(field(elem, "S"))
    O = float(field(elem, "O"))

    if typ == "drift":
        return at.Drift(name, L)
    if typ == "quadrupole":
        if S or O:
            b = np.array([0.0, K, S, O])
            return at.Multipole(name, L, np.zeros_like(b), b, NumIntSteps=nsteps)
        return at.Quadrupole(name, L, k=K, NumIntSteps=nsteps)
    if typ == "sextupole":
        if K or O:
            b = np.array([0.0, K, S, O])
            return at.Multipole(name, L, np.zeros_like(b), b, NumIntSteps=nsteps)
        return at.Sextupole(name, L, h=S, NumIntSteps=nsteps)
    if typ == "multipole":
        b = np.array([0.0, K, S, O])
        a = np.zeros_like(b)
        if L == 0.0:
            return at.ThinMultipole(name, a, b)
        return at.Multipole(name, L, a, b, NumIntSteps=nsteps)
    if typ == "bending":
        if S or O:
            raise NotImplementedError(f"Bend {name} contains S or O.")
        return at.Dipole(
            name,
            L,
            bending_angle=math.radians(angle),
            k=K,
            NumIntSteps=nsteps,
        )
    raise ValueError(f"Unknown magnet type {typ!r}: {name}")


def _tracking_ring(config, lattice_cfg, context):
    if importlib.util.find_spec("at") is None:
        raise ImportError("Accelerator Toolbox is required for tracking.")

    import at
    from at.physics import find_orbit

    native = _prepare_physical_ring(config, lattice_cfg, context)
    if not np.isclose(abs(native["total_bend_deg"]), 360.0, atol=1e-6):
        raise ValueError(
            f"Physical tracking ring bends {native['total_bend_deg']} deg, not 360."
        )

    elements = [
        _at_element(elem, at, int(config.TRACKING_NUM_INT_STEPS))
        for elem in native["lattice"]
    ]
    energy = (
        float(native["parameters"][lattice_cfg.ENERGY_PARAMETER]) * 1.0e9
    )
    ring = at.Lattice(elements, name="FANQO_tracking_cache", energy=energy)
    orbit, _ = find_orbit(ring)
    return ring, np.asarray(orbit, dtype=float).reshape(6), native


def _cache_signature(config, lattice_cfg, context):
    payload = {
        "parameters": _jsonable(context["parameters"]),
        "cell_names": list(lattice_cfg.CELL_NAMES),
        "coords_mm": list(map(float, config.TRACKING_COORDS_MM)),
        "steps": list(map(int, config.TRACKING_STEPS)),
        "turns": int(config.TRACKING_TURNS),
        "delta": float(config.TRACKING_DELTA),
        "num_int_steps": int(config.TRACKING_NUM_INT_STEPS),
        "ring_cells": getattr(config, "TRACKING_PHYSICAL_RING_CELLS", None),
        "chromatic_correction": bool(config.CORRECT_CHROMATICITY),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha1(encoded).hexdigest(), payload


def _dominant_tune(signal):
    """Tune of one normalized complex betatron signal.

    AT harmonic analysis is the primary estimator.  Very small signals have no
    meaningful tune, so they are returned as NaN without asking AT to search.
    AT warnings are intentionally local to this routine and are not printed by
    the research runner.
    """
    signal = np.asarray(signal, dtype=np.complex128).reshape(-1)
    if len(signal) < 16 or not np.all(np.isfinite(signal)):
        return np.nan

    signal = signal - np.mean(signal)
    amplitude = float(np.sqrt(np.mean(np.abs(signal) ** 2)))
    reference = max(float(np.max(np.abs(signal))), 1.0)
    if amplitude <= 1.0e-14 * reference:
        return np.nan

    try:
        from at.lattice import AtWarning
        from at.physics.harmonic_analysis import get_tunes_harmonic

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", AtWarning)
            tune = np.asarray(
                get_tunes_harmonic(
                    signal.reshape(1, -1),
                    method="interp_fft",
                    fmin=0.0,
                    fmax=1.0,
                    num_harmonics=8,
                    maxiter=50,
                    remove_mean=True,
                ),
                dtype=float,
            ).reshape(-1)

        if tune.size and np.isfinite(tune[0]):
            return float(tune[0] % 1.0)

    except (ImportError, AttributeError, TypeError, ValueError):
        pass

    # Numerical fallback only if harmonic analysis did not return a tune.
    spectrum = np.abs(np.fft.fft(signal))
    frequency = np.fft.fftfreq(len(signal))
    if len(spectrum) <= 1:
        return np.nan
    spectrum[0] = 0.0
    k = int(np.argmax(spectrum))
    if not np.isfinite(spectrum[k]) or spectrum[k] <= 0.0:
        return np.nan
    return float(frequency[k] % 1.0)


def _normalized_betatron_signals(trajectory, cs0):
    """Courant-Snyder normalized complex x/y signals.

    This follows the same normalization used by Accelerator Toolbox nonlinear
    tune analysis:

        X  = x / sqrt(beta)
        PX = alpha*x/sqrt(beta) + sqrt(beta)*px
        a  = X - i PX

    and analogously in y.
    """
    z = np.asarray(trajectory, dtype=float)
    bx, ax, _, by, ay, _ = np.asarray(cs0, dtype=float)

    if bx <= 0.0 or by <= 0.0:
        raise ValueError("Positive beta functions are required for FMA.")

    x = z[0] - np.mean(z[0])
    px = z[1] - np.mean(z[1])
    y = z[2] - np.mean(z[2])
    py = z[3] - np.mean(z[3])

    sbx = math.sqrt(float(bx))
    sby = math.sqrt(float(by))

    X = x / sbx
    PX = float(ax) * x / sbx + sbx * px
    Y = y / sby
    PY = float(ay) * y / sby + sby * py

    return X - 1j * PX, Y - 1j * PY


def _fma_from_trajectory(trajectory, cs0):
    """Split-window FMA summary from one saved physical trajectory."""
    z = np.asarray(trajectory, dtype=float)
    n = z.shape[1]
    half = n // 2
    if half < 16:
        return (np.nan,) * 5

    ax_signal, ay_signal = _normalized_betatron_signals(z, cs0)

    qx1 = _dominant_tune(ax_signal[:half])
    qx2 = _dominant_tune(ax_signal[-half:])
    qy1 = _dominant_tune(ay_signal[:half])
    qy2 = _dominant_tune(ay_signal[-half:])

    if not np.all(np.isfinite([qx1, qx2, qy1, qy2])):
        diffusion = np.nan
    else:
        # Circular tune difference: 0.99 and 0.01 differ by 0.02, not 0.98.
        dqx = ((qx2 - qx1 + 0.5) % 1.0) - 0.5
        dqy = ((qy2 - qy1 + 0.5) % 1.0) - 0.5
        diffusion = 0.5 * math.log10(
            max((dqx * dqx + dqy * dqy) / float(half), 1e-20)
        )
        diffusion = float(np.clip(diffusion, -10.0, -2.0))

    return qx1, qx2, qy1, qy2, diffusion



def _refresh_fma(folder, cs0):
    """Recompute FMA arrays from saved turn-by-turn coordinates only."""
    folder = Path(folder).resolve()
    coordinates = np.load(folder / "coordinates.npy", mmap_mode="r")
    completed = np.load(folder / "completed_turns.npy", mmap_mode="r")
    npoints = coordinates.shape[1]

    for i in range(npoints):
        ncomplete = int(completed[i])
        if ncomplete < 1:
            continue
        usable = coordinates[:, i, : ncomplete + 1]
        values = _fma_from_trajectory(usable, cs0)
        qx1[i], qx2[i], qy1[i], qy2[i], diffusion[i] = values

    np.save(folder / "qx1.npy", qx1)
    np.save(folder / "qx2.npy", qx2)
    np.save(folder / "qy1.npy", qy1)
    np.save(folder / "qy2.npy", qy2)
    np.save(folder / "fma_diffusion.npy", diffusion)


def default_cache_directory(config, config_directory):
    value = getattr(config, "TRACKING_CACHE", None)
    if value is None:
        value = Path(config.OUTPUT_DIRECTORY) / "tracking_cache"
    value = Path(value).expanduser()
    if not value.is_absolute():
        value = Path(config_directory) / value
    return value.resolve()


def load_tracking(cache_directory):
    """Open an existing tracking cache with memory mapping."""
    folder = Path(cache_directory).resolve()
    metadata_path = folder / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Tracking cache not found: {folder}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {
        "directory": folder,
        "metadata": metadata,
        "coordinates": np.load(folder / "coordinates.npy", mmap_mode="r"),
        "x_mm": np.load(folder / "x_mm.npy", mmap_mode="r"),
        "y_mm": np.load(folder / "y_mm.npy", mmap_mode="r"),
        "survived": np.load(folder / "survived.npy", mmap_mode="r"),
        "completed_turns": np.load(folder / "completed_turns.npy", mmap_mode="r"),
        "qx1": np.load(folder / "qx1.npy", mmap_mode="r"),
        "qx2": np.load(folder / "qx2.npy", mmap_mode="r"),
        "qy1": np.load(folder / "qy1.npy", mmap_mode="r"),
        "qy2": np.load(folder / "qy2.npy", mmap_mode="r"),
        "fma_diffusion": np.load(folder / "fma_diffusion.npy", mmap_mode="r"),
    }


def track(config, lattice_cfg, context, cache_directory, *, force=False):
    """Track the configured FMA grid once and save all turn-by-turn coordinates."""
    folder = Path(cache_directory).resolve()
    signature, payload = _cache_signature(config, lattice_cfg, context)
    metadata_path = folder / "metadata.json"

    if metadata_path.is_file() and not force:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("signature") == signature:
            if int(metadata.get("fma_version", 0)) != FMA_VERSION:
                native = _prepare_physical_ring(config, lattice_cfg, context)
                cs0 = np.asarray(
                    lin.linear_data(native["data"], "CS0"),
                    dtype=float,
                )
                _refresh_fma(folder, cs0)
                metadata["fma_version"] = FMA_VERSION
                metadata_path.write_text(
                    json.dumps(metadata, indent=2),
                    encoding="utf-8",
                )
            return load_tracking(folder)

    folder.mkdir(parents=True, exist_ok=True)

    ring, orbit, native = _tracking_ring(config, lattice_cfg, context)
    xmin, xmax, ymin, ymax = map(float, config.TRACKING_COORDS_MM)
    nx, ny = map(int, config.TRACKING_STEPS)
    turns = int(config.TRACKING_TURNS)
    delta = float(config.TRACKING_DELTA)
    cs0 = np.asarray(lin.linear_data(native["data"], "CS0"), dtype=float)

    xs = np.linspace(min(xmin, xmax), max(xmin, xmax), nx)
    ys = np.linspace(min(ymin, ymax), max(ymin, ymax), ny)
    X, Y = np.meshgrid(xs, ys)
    x_mm = X.reshape(-1)
    y_mm = Y.reshape(-1)
    npoints = len(x_mm)

    coordinates = np.lib.format.open_memmap(
        folder / "coordinates.npy",
        mode="w+",
        dtype=np.float64,
        shape=(6, npoints, turns + 1),
    )
    coordinates[:] = np.nan

    survived = np.zeros(npoints, dtype=bool)
    completed = np.zeros(npoints, dtype=np.int32)
    qx1 = np.full(npoints, np.nan)
    qx2 = np.full(npoints, np.nan)
    qy1 = np.full(npoints, np.nan)
    qy2 = np.full(npoints, np.nan)
    diffusion = np.full(npoints, np.nan)

    from at.tracking import patpass

    pool_size = getattr(config, "TRACKING_POOL_SIZE", None)

    for iy, y0 in enumerate(ys):
        start = iy * nx
        stop = start + nx
        z0 = np.repeat(orbit.reshape(1, 6), nx, axis=0)
        z0[:, 4] += delta
        z0[:, 0] += 1.0e-3 * xs
        z0[:, 2] += 1.0e-3 * y0
        coordinates[:, start:stop, 0] = z0.T

        kwargs = {"losses": True}
        if pool_size is not None:
            kwargs["pool_size"] = int(pool_size)

        tracked, loss = patpass(
            ring,
            np.asfortranarray(z0.T),
            turns,
            **kwargs,
        )
        tracked = np.asarray(tracked, dtype=float)
        if tracked.ndim == 4:
            tracks = tracked[:, :, 0, :]
        elif tracked.ndim == 3:
            tracks = tracked
        else:
            raise RuntimeError(f"Unexpected patpass shape: {tracked.shape}")

        coordinates[:, start:stop, 1:] = tracks
        lost = np.asarray(
            loss.get("islost", np.zeros(nx, dtype=bool)),
            dtype=bool,
        )

        for local in range(nx):
            global_index = start + local
            part = tracks[:, local, :]
            finite = np.all(np.isfinite(part), axis=0)
            ncomplete = (
                int(np.flatnonzero(~finite)[0])
                if not np.all(finite)
                else turns
            )
            ncomplete = max(0, min(ncomplete, turns))
            completed[global_index] = ncomplete
            survived[global_index] = bool(ncomplete == turns and not lost[local])


    coordinates.flush()
    np.save(folder / "x_mm.npy", x_mm)
    np.save(folder / "y_mm.npy", y_mm)
    np.save(folder / "survived.npy", survived)
    np.save(folder / "completed_turns.npy", completed)

    # FMA is a post-processing step over the saved trajectories.  It can be
    # regenerated later without running patpass again.
    _refresh_fma(folder, cs0)

    metadata = {
        "signature": signature,
        "tracking": payload,
        "n_cells": int(native["n_cells"]),
        "cell_bend_deg": float(native["cell_bend_deg"]),
        "total_bend_deg": float(native["total_bend_deg"]),
        "npoints": int(npoints),
        "turns": int(turns),
        "shape": [6, int(npoints), int(turns + 1)],
        "fma_version": FMA_VERSION,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return load_tracking(folder)


def _coefficient_signature(result):
    physical = nl.physical_coefficients(result["Ix"], result["state"])
    return hashlib.sha1(np.asarray(physical, dtype=np.float64).tobytes()).hexdigest()


def invariance_metrics(
    result,
    tracking,
    *,
    floor_fraction=1e-12,
    force=False,
):
    """Evaluate one Ix on cached trajectories; no particle tracking occurs here."""
    folder = Path(tracking["directory"])
    method = str(result["method"])
    target = folder / f"invariance_{method}.npz"
    coefficient_signature = _coefficient_signature(result)
    tracking_signature = str(tracking["metadata"].get("signature", ""))
    evaluation_signature = hashlib.sha1(
        (
            coefficient_signature
            + "|"
            + tracking_signature
            + "|"
            + repr(float(floor_fraction))
        ).encode()
    ).hexdigest()

    if target.is_file() and not force:
        with np.load(target, allow_pickle=False) as cached:
            stored = str(cached["evaluation_signature"].item())
            if stored == evaluation_signature:
                return {key: cached[key].copy() for key in cached.files}

    coordinates = tracking["coordinates"]
    completed = np.asarray(tracking["completed_turns"])
    survived = np.asarray(tracking["survived"])
    npoints = coordinates.shape[1]

    initial = nl.evaluate_invariant(
        result["Ix"],
        result["state"],
        coordinates[:, :, 0],
    ).reshape(-1)
    scale = float(np.nanmax(np.abs(initial)))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    floor = float(floor_fraction) * scale

    D = np.full(npoints, np.nan)
    log10D = np.full(npoints, np.nan)
    rms = np.full(npoints, np.nan)
    valid = np.zeros(npoints, dtype=bool)

    for i in range(npoints):
        ncomplete = int(completed[i])
        if ncomplete < 1 or not survived[i]:
            continue

        trajectory = coordinates[:, i, : ncomplete + 1]
        values = np.asarray(
            nl.evaluate_invariant(result["Ix"], result["state"], trajectory),
            dtype=float,
        ).reshape(-1)
        if len(values) < 2 or not np.all(np.isfinite(values)):
            continue

        relative = np.abs(values[1:] - values[0]) / max(abs(values[0]), floor)
        turn = np.arange(1, len(relative) + 1, dtype=float)
        rate = relative / turn
        D[i] = float(np.max(rate))
        log10D[i] = math.log10(max(D[i], 1e-300))
        rms[i] = float(np.sqrt(np.mean(relative ** 2)))
        valid[i] = True

    np.savez(
        target,
        method=np.asarray(method),
        coefficient_signature=np.asarray(coefficient_signature),
        evaluation_signature=np.asarray(evaluation_signature),
        D=D,
        log10D=log10D,
        rms=rms,
        valid=valid,
        initial=initial,
        floor=np.asarray(floor),
    )
    return {
        "method": np.asarray(method),
        "coefficient_signature": np.asarray(coefficient_signature),
        "evaluation_signature": np.asarray(evaluation_signature),
        "D": D,
        "log10D": log10D,
        "rms": rms,
        "valid": valid,
        "initial": initial,
        "floor": np.asarray(floor),
    }
