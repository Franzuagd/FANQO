"""Small container for FANQO's active user session.

There are three similarly named objects worth keeping separate while reading:

1. STATE
   Global user-session memory: what experiment is currently active.

2. STATE.context
   The mutable optimization context: lattice, linear data, parameters,
   nonlinear state, caches, and settings.

3. STATE.context["state"]
   The nonlinear polynomial representation: monomial basis, Gram matrix,
   Lie matrices, normalization, derivatives, and invariant convention.

The public API updates this singleton so the user does not have to manually
pass the context between load, compute, plot, optimize, and report calls.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RuntimeState:
    # Loaded experiment configuration modules.
    config: object | None = None
    config_path: str | None = None
    lattice_config: object | None = None

    # Main mutable accelerator experiment used by the optimization engine.
    context: dict | None = None

    # Invariant vectors currently exposed through the public API.
    Ix: np.ndarray | None = None
    Iy: np.ndarray | None = None
    invariant_details: dict | None = None

    # Results kept after longer operations finish.
    optimization_result: dict | None = None
    a_box_result: dict | None = None
    diagnostics: dict = field(default_factory=dict)

    # Human-readable origin of active parameters: unloaded/loaded/edited/optimized.
    source: str = "unloaded"


# One session-wide state object used by fanqo.api.
STATE = RuntimeState()
