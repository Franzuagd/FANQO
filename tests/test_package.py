import fanqo


def test_version():
    assert fanqo.__version__ == "0.3.0.dev0"


def test_public_api_is_invariant_only():
    public_functions = (
        "load",
        "status",
        "available_methods",
        "construct",
        "coefficients",
        "polynomial",
        "construction_details",
        "compare",
    )
    for name in public_functions:
        assert callable(getattr(fanqo, name))

    assert fanqo.available_methods() == (
        "a_box",
        "hybrid",
        "eigen",
        "graded_ls",
        "a_box_y0",
    )

    assert not hasattr(fanqo, "optimize")
    assert not hasattr(fanqo, "optimize_a_box")


def test_save_only_plot_backend():
    import matplotlib
    from fanqo.plotting import get_pyplot

    plt = get_pyplot(False)
    assert str(matplotlib.get_backend()).lower() == "agg"
    fig, _ = plt.subplots()
    plt.close(fig)
