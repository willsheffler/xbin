from contextlib import redirect_stdout
import numpy as np
import sys, os
import homog
with redirect_stdout(open(os.devnull,'w')):
    from xbin import *

ori_nside = int(sys.argv[1])
Nsamp = 1000000
Nptiles = 101
ptiles = np.concatenate([np.arange(Nptiles),
                         100 - 10**np.linspace(2, -6, Nptiles)])
estimate = np.zeros_like(ptiles)
coherent = np.zeros_like(ptiles)

while True:
    xforms = homog.rand_xform(Nsamp)
    xb = XformBinner(ori_nside=ori_nside)
    idx = xb.get_bin_index(xforms)
    cen = xb.get_bin_center(idx)
    ori_dist = homog.angle_of(homog.hinv(cen) @ xforms)
    ehat = np.percentile(ori_dist, ptiles)
    old = estimate
    estimate = np.maximum(estimate, ehat)
    if np.any(old != estimate):
        print('independent', Nsamp, ori_nside,
              str(estimate.tolist()).replace(',','')[1:-1])
        sys.stdout.flush()
    if np.sum(ehat) > np.sum(coherent):
        coherent = ehat
        print('coherent', Nsamp, ori_nside,
              str(coherent.tolist()).replace(',','')[1:-1])
        sys.stdout.flush()
