import fanqo


def test_version():
    assert fanqo.__version__ == "0.2.0"


def test_public_api():
    public_functions = (
        "load",
        "status",
        "linear_summary",
        "compute_invariants",
        "run_fma",
        "optimize",
    )
    for name in public_functions:
        assert callable(getattr(fanqo, name))
