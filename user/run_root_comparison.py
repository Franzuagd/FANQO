"""Construct root_geometry and a_box, then compare at delta=0 only.

Place beside your general_config.py and lattice file. No FMA or old Poincare
plots. --geometry-only never imports/uses the optional tracking dependency.
"""
import argparse
import os
from pathlib import Path

# EDITABLE SETTINGS: FANQO uses J=S, without the conventional factor 1/2.
# Used when an older general_config.py has no ACTION_BOX setting.
ACTION_BOX = {'Jx_max': 1.75e-5, 'Jy_max': 1.83e-5, 'delta_max': .01}
ACTION_FRACTIONS = (.05, .15, .3, .5, .7, .9)
ROOT_ACTION_FRACTIONS = (.05, .15, .3, .5, .7)
ROOT_RESIDUAL_FACTOR = 1.1
ROOT_RESIDUAL_FLOOR = 1e-8  # action-metric residual / norm of CS seed
ROOT_MAX_ITER = 60
ROOT_ANGLES = 64
TRACKING_TURNS = 1024
PLOT_ANGLES = 360


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).with_name('general_config.py')))
    parser.add_argument('--output', default='root_comparison')
    parser.add_argument('--order', type=int, default=8)
    parser.add_argument('--delta-order', type=int, choices=[0, 1], default=1)
    parser.add_argument('--jx', type=float, nargs='+', help='Explicit launch actions J=S [m].')
    parser.add_argument('--jx-max', type=float, help='Override ACTION_BOX Jx_max.')
    parser.add_argument('--turns', type=int, default=TRACKING_TURNS)
    parser.add_argument('--angles', type=int, default=PLOT_ANGLES)
    parser.add_argument('--root-iterations', type=int, default=ROOT_MAX_ITER)
    parser.add_argument('--residual-factor', type=float, default=ROOT_RESIDUAL_FACTOR)
    parser.add_argument('--residual-floor', type=float, default=ROOT_RESIDUAL_FLOOR)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--geometry-only', action='store_true')
    parser.add_argument('--quick', action='store_true', help='Small angular/turn budget; order unchanged.')
    args = parser.parse_args()
    if args.threads < 1: parser.error('--threads must be positive')
    for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[name] = str(args.threads)
    try:
        from fanqo import config_loader
        from fanqo.root_comparison import run_root_comparison
    except ModuleNotFoundError as exc:
        if exc.name in ('fanqo', 'fanqo.root_comparison'):
            raise SystemExit('Install FANQO branch root-geometry-0.3 in this environment first.') from exc
        raise
    cfg = config_loader.load_general_config(args.config, reload=True)
    box = dict(getattr(cfg, 'ACTION_BOX', ACTION_BOX))
    if args.jx_max is not None: box['Jx_max'] = args.jx_max
    run_root_comparison(args.config, output=args.output, order=args.order, delta_order=args.delta_order,
        action_box=box, launch_actions=args.jx, action_fractions=ACTION_FRACTIONS,
        angles=min(args.angles, 96) if args.quick else args.angles,
        turns=min(args.turns, 128) if args.quick else args.turns,
        geometry_only=args.geometry_only,
        root_options=dict(ROOT_ACTION_FRACTIONS=ROOT_ACTION_FRACTIONS,
            ROOT_ANGLES=24 if args.quick else ROOT_ANGLES,
            ROOT_MAX_ITER=min(args.root_iterations, 15) if args.quick else args.root_iterations,
            ROOT_RESIDUAL_FACTOR=args.residual_factor, ROOT_RESIDUAL_FLOOR=args.residual_floor))


if __name__ == '__main__':
    main()
