"""Runtime state for the invariant-construction-only FANQO branch."""

from dataclasses import dataclass, field


@dataclass
class RuntimeState:
    config: object | None = None
    config_path: str | None = None
    lattice_config: object | None = None

    # Fixed lattice/linear-optics data used by every invariant constructor.
    context: dict | None = None

    # Cached method -> {state, Ix, details, transfer}.
    invariants: dict = field(default_factory=dict)

    source: str = "unloaded"


STATE = RuntimeState()
