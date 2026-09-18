import fanqo


def test_version():
    assert fanqo.__version__ == "0.2.1"


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


def test_save_only_plot_backend():
    import matplotlib
    from fanqo.plotting import get_pyplot

    plt = get_pyplot(False)
    assert str(matplotlib.get_backend()).lower() == "agg"
    fig, _ = plt.subplots()
    plt.close(fig)
