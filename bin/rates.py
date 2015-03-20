#!/usr/bin/env python

import argparse
import emcee
import numpy as np
import rates_tool.rate as rt
import plotutils.runner as pr
import scipy.optimize as so

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--outdir', default='.', help='Output directory (default %(default)s)')

    parser.add_argument('--coincs', required=True, help='Coinc file')
    parser.add_argument('--background', action='append', required=True, help='Single-detector background file')
    parser.add_argument('--foreground', required=True, help='All-detector foreground distribution')

    parser.add_argument('--nwalkers', default=128, type=int, help='Number of walkers (default %(default)s)')
    parser.add_argument('--nensembles', default=128, type=int, help='Number of independent ensembles (default %(default)s')

    args = parser.parse_args()

    fore = np.loadtxt(args.foreground)
    backs = [np.loadtxt(b) for b in args.background]
    coincs = np.loadtxt(args.coincs)

    print [b.shape for b in backs], fore.shape, coincs.shape

    ratepost = rt.RatePosterior(backs, fore, coincs)

    print ratepost.log_fg

    pbest = so.fmin_powell(lambda x: -ratepost(x), np.zeros(2))

    sampler = emcee.EnsembleSampler(args.nwalkers, 2, ratepost)
    runner = pr.EnsembleSamplerRunner(sampler, pbest + 1e-3*np.random.randn(args.nwalkers, 2))

    runner.run_to_neff(args.nensembles, savedir=args.outdir)
