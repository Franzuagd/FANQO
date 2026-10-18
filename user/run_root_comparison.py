"""Direct action-angle least squares versus unchanged a_box, delta=0 only.

Put beside general_config.py and its lattice file. All-root diagrams cover
0<=phi_x<=2*pi. Red/blue invariance maps use the old FMA tracking metric.
"""
import argparse
import os
from pathlib import Path

# FANQO actions J=S have no factor 1/2. Config ACTION_BOX takes priority.
ACTION_BOX = {'Jx_max':1.75e-5, 'Jy_max':1.83e-5, 'delta_max':.01}
ACTION_FRACTIONS = (.05,.15,.3,.5,.7,.9)
# None: use the same numerical values as the launch actions for fixed C levels.
C_VALUES = None
TRACKING_TURNS = 1024
PLOT_ANGLES = 360
MAP_STEPS = (30,30)  # intervals: 31*31 launches; physical range from config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(Path(__file__).with_name('general_config.py')))
    p.add_argument('--output',default='action_angle_comparison')
    p.add_argument('--order',type=int,default=8)
    p.add_argument('--delta-order',type=int,choices=[0,1],default=1)
    p.add_argument('--jx',type=float,nargs='+',help='Horizontal launch actions J=S [m].')
    p.add_argument('--levels',type=float,nargs='+',default=C_VALUES,help='Fixed C levels; all real nonnegative roots are plotted.')
    p.add_argument('--jx-max',type=float,help='Override ACTION_BOX Jx_max.')
    p.add_argument('--turns',type=int,default=TRACKING_TURNS)
    p.add_argument('--angles',type=int,default=PLOT_ANGLES)
    p.add_argument('--map-grid',type=int,nargs=2,default=MAP_STEPS,metavar=('NX','NY'))
    p.add_argument('--map-turns',type=int,help='Invariance-map turns; defaults to --turns.')
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--geometry-only',action='store_true')
    p.add_argument('--quick',action='store_true',help='Small angular/tracking grids; m,d unchanged.')
    args=p.parse_args()
    if args.threads<1: p.error('--threads must be positive')
    for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
        os.environ[name]=str(args.threads)
    try:
        from fanqo import config_loader
        from fanqo.root_comparison import run_root_comparison
        import fanqo.core.action_angle_ls  # reject the old nonlinear-refinement package
    except ModuleNotFoundError as exc:
        if exc.name in ('fanqo','fanqo.root_comparison','fanqo.core.action_angle_ls'):
            raise SystemExit('Install the updated root-geometry-0.3 branch in this Python environment first.') from exc
        raise
    cfg=config_loader.load_general_config(args.config,reload=True)
    box=dict(getattr(cfg,'ACTION_BOX',ACTION_BOX))
    if args.jx_max is not None: box['Jx_max']=args.jx_max
    run_root_comparison(args.config,output=args.output,order=args.order,delta_order=args.delta_order,
        action_box=box,launch_actions=args.jx,action_fractions=ACTION_FRACTIONS,level_values=args.levels,
        angles=min(args.angles,96) if args.quick else args.angles,
        turns=min(args.turns,128) if args.quick else args.turns,
        map_steps=tuple(min(n,8) for n in args.map_grid) if args.quick else tuple(args.map_grid),
        map_turns=min(args.map_turns or args.turns,64) if args.quick else args.map_turns,
        geometry_only=args.geometry_only)


if __name__=='__main__':
    main()
