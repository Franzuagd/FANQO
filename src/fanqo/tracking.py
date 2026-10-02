"""Physical tracking cache for invariant studies.

The tracking and invariance definitions in this file intentionally follow the
working FANQO implementation from Development-0.3/main.

One physical patpass data set is stored and reused by every invariant method.
The FMA values are post-processed from those same saved trajectories using the
same split-window harmonic analysis used by
at.physics.frequency_maps.fmap_parallel_track.
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


# Increment this whenever the physical launch grid or trajectory convention
# changes. A mismatch forces one clean retracking run.
TRACKING_CACHE_VERSION = 3
FMA_VERSION = 3


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


# =============================================================================
# PHYSICAL RING -- kept equivalent to the proven FANQO FMA implementation
# =============================================================================

def _infer_ring_cells(lattice_cfg, parameters):
    magnets = lattice_cfg.define_magnets(parameters)
    by_name = {lin.magnet_field(e, "NAME"): e for e in magnets}

    bend = 0.0
    for name in lattice_cfg.CELL_NAMES:
        elem = by_name[name]
        if lin.magnet_field(elem, "TYPE") == "bending":
            bend += float(lin.magnet_field(elem, "ANGLE"))

    if abs(bend) < 1.0e-12:
        raise ValueError("Configured cell has zero net bend.")

    cells_float = 360.0 / abs(bend)
    cells = int(round(cells_float))
    if cells < 1 or not np.isclose(cells_float, cells, rtol=0.0, atol=1.0e-7):
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
            return at.Multipole(
                name, L, np.zeros_like(b), b, NumIntSteps=nsteps
            )
        return at.Quadrupole(name, L, k=K, NumIntSteps=nsteps)

    if typ == "sextupole":
        if K or O:
            b = np.array([0.0, K, S, O])
            return at.Multipole(
                name, L, np.zeros_like(b), b, NumIntSteps=nsteps
            )
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
    if not np.isclose(abs(native["total_bend_deg"]), 360.0, atol=1.0e-6):
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


# =============================================================================
# LAUNCH GRID -- same convention as AT fmap_parallel_track
# =============================================================================

def _fma_grid(coords_mm, steps):
    """Return the AT/FANQO FMA launch grid.

    TRACKING_STEPS is interpreted as the number of intervals, exactly as
    fmap_parallel_track does. Therefore [121,121] gives 122x122 launch points.
    """
    if len(coords_mm) != 4 or len(steps) != 2:
        raise ValueError(
            "Tracking grid needs coords=[xmin,xmax,ymin,ymax], steps=[nx,ny]."
        )

    xmin, xmax = sorted(map(float, coords_mm[:2]))
    ymin, ymax = sorted(map(float, coords_mm[2:]))
    nx, ny = map(int, steps)
    if nx <= 0 or ny <= 0:
        raise ValueError("TRACKING_STEPS must be positive.")

    xs = (
        np.array([xmin], dtype=float)
        if xmin == xmax
        else np.linspace(xmin, xmax, nx + 1)
    )
    ys = (
        np.array([ymin], dtype=float)
        if ymin == ymax
        else np.linspace(ymin, ymax, ny + 1)
    )
    return xs, ys


def _initial_row(xs_mm, y_mm, orbit, delta):
    """Initial conditions matching fmap_parallel_track.

    The 1 nm x/y offset is intentional and is present in AT's implementation
    to avoid exactly-zero ideal-lattice signals.
    """
    z = np.repeat(
        np.asarray(orbit, dtype=float).reshape(1, 6),
        len(xs_mm),
        axis=0,
    )
    z[:, 4] += float(delta)
    z[:, 0] += 1.0e-3 * np.asarray(xs_mm, dtype=float) + 1.0e-9
    z[:, 2] += 1.0e-3 * float(y_mm) + 1.0e-9
    return z


def delta_label(delta):
    """Filesystem-safe label for one momentum offset."""
    value = float(delta)
    sign = "p" if value >= 0.0 else "m"
    text = f"{abs(value):.8g}".replace(".", "p")
    return f"delta_{sign}{text}"


def _cache_signature(config, lattice_cfg, context, delta):
    payload = {
        "cache_version": TRACKING_CACHE_VERSION,
        "parameters": _jsonable(context["parameters"]),
        "cell_names": list(lattice_cfg.CELL_NAMES),
        "coords_mm": list(map(float, config.TRACKING_COORDS_MM)),
        "steps": list(map(int, config.TRACKING_STEPS)),
        "turns": int(config.TRACKING_TURNS),
        "delta": float(delta),
        "num_int_steps": int(config.TRACKING_NUM_INT_STEPS),
        "ring_cells": getattr(config, "TRACKING_PHYSICAL_RING_CELLS", None),
        "chromatic_correction": bool(config.CORRECT_CHROMATICITY),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha1(encoded).hexdigest(), payload


# =============================================================================
# FMA POST-PROCESSING -- copied mathematically from fmap_parallel_track
# =============================================================================

def _harmonic_tune(signal):
    """Run AT's default harmonic analysis without changing its mathematics."""
    signal = np.asarray(signal, dtype=float).reshape(-1)
    if signal.size < 2 or not np.all(np.isfinite(signal)):
        return np.nan

    signal = signal - np.mean(signal)

    try:
        from at.lattice import AtWarning
        from at.physics.harmonic_analysis import get_tunes_harmonic

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", AtWarning)
            frequency = np.asarray(
                get_tunes_harmonic(signal),
                dtype=float,
            ).reshape(-1)
    except (ImportError, AttributeError, TypeError, ValueError):
        return np.nan

    if frequency.size == 0 or not np.isfinite(frequency[0]):
        return np.nan
    return float(frequency[0])


def _fma_from_trajectory(trajectory, turns):
    """FMA row from a saved trajectory, matching AT frequency_maps.py.

    AT's fmap_parallel_track tracks 2*tns turns, analyzes raw x/y separately
    over the first and second tns windows, then computes

        0.5*log10((dnux^2 + dnuy^2)/tns)

    clipped to [-10,-2].
    """
    turns = int(turns)
    if turns < 2 or turns % 2:
        raise ValueError("TRACKING_TURNS must be even for split-window FMA.")

    tns = turns // 2
    trajectory = np.asarray(trajectory, dtype=float)

    # Column 0 in our cache is the initial condition; AT's zOUT begins at turn 1.
    z = trajectory[:, 1 : turns + 1]
    if z.shape[1] != turns or not np.all(np.isfinite(z)):
        return (np.nan,) * 5

    xfirst = z[0, :tns] - np.mean(z[0, :tns])
    xlast = z[0, tns : 2 * tns] - np.mean(z[0, tns : 2 * tns])
    yfirst = z[2, :tns] - np.mean(z[2, :tns])
    ylast = z[2, tns : 2 * tns] - np.mean(z[2, tns : 2 * tns])

    nux = _harmonic_tune(xfirst)
    nux_last = _harmonic_tune(xlast)
    nuy = _harmonic_tune(yfirst)
    nuy_last = _harmonic_tune(ylast)

    if not np.all(np.isfinite([nux, nux_last, nuy, nuy_last])):
        return (np.nan,) * 5

    dnux = nux_last - nux
    dnuy = nuy_last - nuy
    diffusion = 0.5 * np.log10(
        max((dnux * dnux + dnuy * dnuy) / float(tns), 1.0e-300)
    )
    diffusion = float(np.clip(diffusion, -10.0, -2.0))
    return nux, nuy, dnux, dnuy, diffusion


def _refresh_fma(folder):
    """Recompute AT-style FMA values from saved coordinates, no retracking."""
    folder = Path(folder).resolve()
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    turns = int(metadata["turns"])

    coordinates = np.load(folder / "coordinates.npy", mmap_mode="r")
    x_mm = np.load(folder / "x_mm.npy", mmap_mode="r")
    y_mm = np.load(folder / "y_mm.npy", mmap_mode="r")
    survived = np.load(folder / "survived.npy", mmap_mode="r")

    npoints = coordinates.shape[1]
    nux = np.full(npoints, np.nan)
    nuy = np.full(npoints, np.nan)
    dnux = np.full(npoints, np.nan)
    dnuy = np.full(npoints, np.nan)
    diffusion = np.full(npoints, np.nan)

    valid_rows = []
    for i in range(npoints):
        if not bool(survived[i]):
            continue

        values = _fma_from_trajectory(coordinates[:, i, :], turns)
        if not np.all(np.isfinite(values)):
            continue

        nux[i], nuy[i], dnux[i], dnuy[i], diffusion[i] = values
        valid_rows.append([
            float(x_mm[i]),
            float(y_mm[i]),
            nux[i],
            nuy[i],
            dnux[i],
            dnuy[i],
            diffusion[i],
        ])

    fmap = (
        np.asarray(valid_rows, dtype=float).reshape(-1, 7)
        if valid_rows
        else np.empty((0, 7), dtype=float)
    )

    np.save(folder / "nux.npy", nux)
    np.save(folder / "nuy.npy", nuy)
    np.save(folder / "dnux.npy", dnux)
    np.save(folder / "dnuy.npy", dnuy)
    np.save(folder / "fma_diffusion.npy", diffusion)
    np.save(folder / "fmap.npy", fmap)


# =============================================================================
# CACHE I/O
# =============================================================================

def default_cache_directory(config, config_directory):
    value = getattr(config, "TRACKING_CACHE", None)
    if value is None:
        value = Path(config.OUTPUT_DIRECTORY) / "tracking_cache"
    value = Path(value).expanduser()
    if not value.is_absolute():
        value = Path(config_directory) / value
    return value.resolve()


def load_tracking(cache_directory):
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
        "completed_turns": np.load(
            folder / "completed_turns.npy", mmap_mode="r"
        ),
        "nux": np.load(folder / "nux.npy", mmap_mode="r"),
        "nuy": np.load(folder / "nuy.npy", mmap_mode="r"),
        "dnux": np.load(folder / "dnux.npy", mmap_mode="r"),
        "dnuy": np.load(folder / "dnuy.npy", mmap_mode="r"),
        "fma_diffusion": np.load(
            folder / "fma_diffusion.npy", mmap_mode="r"
        ),
        "fmap": np.load(folder / "fmap.npy", mmap_mode="r"),
    }


def track(
    config,
    lattice_cfg,
    context,
    cache_directory,
    *,
    force=False,
    delta=None,
):
    """Track one configured momentum offset and cache every ring turn."""
    folder = Path(cache_directory).resolve()
    effective_delta = (
        float(config.TRACKING_DELTA)
        if delta is None
        else float(delta)
    )
    signature, payload = _cache_signature(
        config,
        lattice_cfg,
        context,
        effective_delta,
    )
    metadata_path = folder / "metadata.json"

    if metadata_path.is_file() and not force:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("signature") == signature:
            if int(metadata.get("fma_version", 0)) != FMA_VERSION:
                _refresh_fma(folder)
                metadata["fma_version"] = FMA_VERSION
                metadata_path.write_text(
                    json.dumps(metadata, indent=2),
                    encoding="utf-8",
                )
            return load_tracking(folder)

    folder.mkdir(parents=True, exist_ok=True)

    ring, orbit, native = _tracking_ring(config, lattice_cfg, context)
    xs, ys = _fma_grid(config.TRACKING_COORDS_MM, config.TRACKING_STEPS)
    turns = int(config.TRACKING_TURNS)
    if turns < 2 or turns % 2:
        raise ValueError("TRACKING_TURNS must be a positive even integer.")
    delta = effective_delta

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

    from at.tracking import patpass

    pool_size = getattr(config, "TRACKING_POOL_SIZE", None)

    for iy, y0 in enumerate(ys):
        start = iy * len(xs)
        stop = start + len(xs)
        z0 = _initial_row(xs, y0, orbit, delta)
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
            loss.get("islost", np.zeros(len(xs), dtype=bool)),
            dtype=bool,
        )

        for local in range(len(xs)):
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
            survived[global_index] = bool(
                ncomplete == turns and not lost[local]
            )

    coordinates.flush()
    np.save(folder / "x_mm.npy", x_mm)
    np.save(folder / "y_mm.npy", y_mm)
    np.save(folder / "survived.npy", survived)
    np.save(folder / "completed_turns.npy", completed)

    metadata = {
        "signature": signature,
        "tracking": payload,
        "n_cells": int(native["n_cells"]),
        "cell_bend_deg": float(native["cell_bend_deg"]),
        "total_bend_deg": float(native["total_bend_deg"]),
        "npoints": int(npoints),
        "turns": int(turns),
        "fma_turns_per_window": int(turns // 2),
        "shape": [6, int(npoints), int(turns + 1)],
        "fma_version": FMA_VERSION,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    _refresh_fma(folder)
    return load_tracking(folder)


# =============================================================================
# IX INVARIANCE -- same definition used by working FANQO branches
# =============================================================================

def _coefficient_signature(result):
    physical = nl.physical_coefficients(result["Ix"], result["state"])
    return hashlib.sha1(
        np.asarray(physical, dtype=np.float64).tobytes()
    ).hexdigest()


def invariance_metrics(
    result,
    tracking,
    *,
    floor_fraction=1.0e-12,
    force=False,
):
    """Evaluate the proven FANQO Ix drift metric on cached trajectories."""
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
            if str(cached["evaluation_signature"].item()) == evaluation_signature:
                return {key: cached[key].copy() for key in cached.files}

    coordinates = tracking["coordinates"]
    survived = np.asarray(tracking["survived"], dtype=bool)
    turns = int(tracking["metadata"]["turns"])
    npoints = coordinates.shape[1]

    ix0 = np.asarray(
        nl.evaluate_invariant(
            result["Ix"],
            result["state"],
            coordinates[:, :, 0],
        ),
        dtype=float,
    ).reshape(-1)

    reference = float(np.nanmax(np.abs(ix0))) if ix0.size else 1.0
    if reference == 0.0 or not np.isfinite(reference):
        reference = 1.0
    floor = float(floor_fraction) * reference

    ix_last = np.full(npoints, np.nan)
    max_abs = np.full(npoints, np.nan)
    max_relative = np.full(npoints, np.nan)
    rms_relative = np.full(npoints, np.nan)
    D = np.full(npoints, np.nan)
    log10D = np.full(npoints, np.nan)
    valid = np.zeros(npoints, dtype=bool)

    for i in range(npoints):
        if not survived[i]:
            continue

        part = coordinates[:, i, 1 : turns + 1]
        values = np.asarray(
            nl.evaluate_invariant(
                result["Ix"],
                result["state"],
                part,
            ),
            dtype=float,
        ).reshape(-1)

        if values.size != turns or not np.all(np.isfinite(values)):
            continue

        delta_ix = values - float(ix0[i])
        denominator = max(abs(float(ix0[i])), floor)
        relative = np.abs(delta_ix) / denominator
        rate = relative / np.arange(1, turns + 1, dtype=float)

        ix_last[i] = float(values[-1])
        max_abs[i] = float(np.max(np.abs(delta_ix)))
        max_relative[i] = float(np.max(relative))
        rms_relative[i] = float(np.sqrt(np.mean(relative ** 2)))
        D[i] = float(np.max(rate))
        log10D[i] = float(np.log10(max(D[i], 1.0e-300)))
        valid[i] = True

    np.savez(
        target,
        method=np.asarray(method),
        coefficient_signature=np.asarray(coefficient_signature),
        evaluation_signature=np.asarray(evaluation_signature),
        initial=ix0,
        last=ix_last,
        max_abs=max_abs,
        max_relative=max_relative,
        rms=rms_relative,
        D=D,
        log10D=log10D,
        valid=valid,
        floor=np.asarray(floor),
    )

    return {
        "method": np.asarray(method),
        "coefficient_signature": np.asarray(coefficient_signature),
        "evaluation_signature": np.asarray(evaluation_signature),
        "initial": ix0,
        "last": ix_last,
        "max_abs": max_abs,
        "max_relative": max_relative,
        "rms": rms_relative,
        "D": D,
        "log10D": log10D,
        "valid": valid,
        "floor": np.asarray(floor),
    }
