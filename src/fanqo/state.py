"""Internal runtime state."""
from dataclasses import dataclass, field
import numpy as np
@dataclass
class RuntimeState:
    config: object | None = None
    config_path: str | None = None
    lattice_config: object | None = None
    context: dict | None = None
    Ix: np.ndarray | None = None
    Iy: np.ndarray | None = None
    invariant_details: dict | None = None
    optimization_result: dict | None = None
    diagnostics: dict = field(default_factory=dict)
    source: str = "unloaded"
STATE = RuntimeState()
