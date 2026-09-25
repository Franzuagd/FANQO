"""Internal Matplotlib backend selection.

Long FANQO runs are commonly executed in screen/SSH/DWService sessions with no
GUI. When SHOW_PLOTS is false we force Matplotlib's Agg backend so saving figures
does not depend on an X server. Interactive display is allowed only from the
main Python thread.
"""

from __future__ import annotations

import threading


def get_pyplot(show=False):
    """Return pyplot using a GUI-free backend when figures are not displayed."""
    show = bool(show)
    if show and threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Interactive plot display must run on the main Python thread. Set SHOW_PLOTS = False to save plots without displaying them.")

    import matplotlib

    if not show:
        matplotlib.use("Agg", force=True)

    import matplotlib.pyplot as plt

    if not show and str(matplotlib.get_backend()).lower() != "agg":
        plt.switch_backend("Agg")

    return plt
