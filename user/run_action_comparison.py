"""Compare action_a_box, regular a_box and Sx-selected eigen on your lattice.

Install the action-a-box-0.3 branch first. Put this file beside your current
general_config.py and lattice file, then run: python run_action_comparison.py
Use --quick for a small diagnostic grid; polynomial order remains m=8, d=1.
"""
from pathlib import Path
import argparse
import os

# USER SETTINGS: J=S=gamma*q^2+2*alpha*q*p+beta*p^2, WITHOUT factor 1/2.
ACTION_BOX = {'Jx_max': 1.75e-5, 'Jy_max': 1.83e-5, 'delta_max': 0.01}
DELTAS = (0., -.01, .01, -.02, .02, -.03, .03)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(Path(__file__).with_name('general_config.py')))
    parser.add_argument('--output',default='action_comparison')
    parser.add_argument('--jx-max',type=float,default=ACTION_BOX['Jx_max'])
    parser.add_argument('--jy-max',type=float,default=ACTION_BOX['Jy_max'])
    parser.add_argument('--delta-max',type=float,default=ACTION_BOX['delta_max'])
    parser.add_argument('--deltas',type=float,nargs='+',default=DELTAS)
    parser.add_argument('--order',type=int,default=8)
    parser.add_argument('--delta-order',type=int,choices=[0,1],default=1)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--timing-only',action='store_true')
    args=parser.parse_args()
    if args.threads<1: parser.error('--threads must be positive')
    # Set before NumPy/FANQO import, including in spawned tracking workers.
    for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
        os.environ[name]=str(args.threads)
    try:
        from fanqo.action_comparison import run_action_comparison
    except ImportError as exc:
        raise SystemExit('Install FANQO from branch action-a-box-0.3 with tracking support first.') from exc
    run_action_comparison(args.config,action_box={
        'Jx_max':args.jx_max,'Jy_max':args.jy_max,'delta_max':args.delta_max},
        output=args.output,deltas=tuple(args.deltas),order=args.order,
        delta_order=args.delta_order,quick=args.quick,timing_only=args.timing_only)


if __name__=='__main__':
    main()
