#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
f-stat function

@author: David Fee, dfee1@alaska.edu using some WATC/Szuberla codes  

"""

def fstatbland(dtmp, rij,fs,tau):
    """
    calculates the F-statistic based on Blandford's method

    Parameters
    ~~~~~~~~~~
    dtmp : array
        (m, n) time series with `m` samples from `n` traces as columns
    rij : array
        (d, n) `n` sensor coordinates as [northing, easting, {elevation}]
        column vectors in `d` dimensions
    fs : float or int
        sample rate [Hz]
    tau : array
        (n(n-1)//2, ) time delays of relative signal arrivals (TDOA) for all
        unique sensor pairings
        
    Returns
    ~~~~~~~
    fstat : array
        f-statistic
    snr : float
        SNR    
          
    Reference:
      Blandford, R. R., 1974, Geophysics, vol. 39, no. 5, p. 633-643
    
    Version
    ~~~~~~~
    1.0

    """
    fstatbland.__version__ = '1.0'

    import numpy as np
    from Z import phaseAlignIdx, phaseAlignData
    
    m,n=dtmp.shape
    wgt=np.ones(n)
    
    #individual trace offsets from arrival model shifts...zero for this
    Moffset=[0 for i in range(n)]
    
    # calculate beam delays
    beam_delays = phaseAlignIdx(tau, fs, wgt, 0)
    
    # apply shifts, resulting in a zero-padded array
    beamMatrix = phaseAlignData(dtmp, beam_delays, wgt, 0,m,Moffset)
    
    fnum = np.sum(np.sum(beamMatrix,axis=1)**2)
    term1 = np.sum(beamMatrix,axis=1)/n
    term1_0 = term1
    for i in range(1,n):
        term1 = np.vstack((term1,term1_0))
    fden = np.sum(np.sum((beamMatrix.T-term1)**2))
    fstat= (n-1)*fnum/(n*fden)
    
    #calculate snr based on fstat
    snr=np.sqrt((fstat-1)/n)
    
    return fstat,snr
    