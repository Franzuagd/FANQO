"""Run with the installed IxOptCompare-0.3 FANQO package.

Quick timing + tracking: python run_structured_comparison.py --small
Full m=8,d=1 comparison: python run_structured_comparison.py
Your lattice/config:    python run_structured_comparison.py --config general_config.py
Timings only:           python run_structured_comparison.py --small --timing-only
The embedded package reference lattice is used when --config is omitted.
"""
import argparse
import os

# Fixed thread count makes timing reproducible and avoids BLAS oversubscription.
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--small',action='store_true',help='m=4,d=1; 7x7 grid; 64 turns; delta=0')
    p.add_argument('--timing-only',action='store_true')
    p.add_argument('--config',help='Optional existing general_config.py (and its LATTICE_FILE)')
    p.add_argument('--order',type=int,default=None)
    p.add_argument('--delta-order',type=int,default=1)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--points',type=int,default=None)
    p.add_argument('--turns',type=int,default=None)
    p.add_argument('--deltas',type=float,nargs='+',default=None)
    p.add_argument('--extent-mm',type=float,default=10.)
    p.add_argument('--coupled-maxiter',type=int,default=200)
    p.add_argument('--output',default='structured_comparison_output')
    p.add_argument('--methods',nargs='+',default=None)
    args=p.parse_args()
    try:
        from fanqo.comparison import run_comparison, DEFAULT_METHODS
    except ImportError as exc:
        raise SystemExit('Update FANQO from IxOptCompare-0.3 first. '+str(exc)) from exc
    run_comparison(order=args.order or (4 if args.small else 8),delta_order=args.delta_order,
        repeats=args.repeats,points=args.points or (7 if args.small else 25),
        turns=args.turns or (64 if args.small else 512),
        deltas=args.deltas if args.deltas is not None else ((0.,) if args.small else (0.,-.01,.01)),
        extent_mm=args.extent_mm,config=args.config,output=args.output,
        methods=args.methods or DEFAULT_METHODS,coupled_maxiter=args.coupled_maxiter,
        timing_only=args.timing_only)


if __name__=='__main__':
    main()
