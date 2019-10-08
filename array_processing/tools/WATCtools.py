
# -*- coding: utf-8 -*-
"""
WATCtools
=========

Provides
    1. Array processing methods applicable to geophysical time series
    2. Parameter estimation tools
    3. Support functions for digital signal processing

How to use the module
~~~~~~~~~~~~~~~~~~~~~
Documentation is available in docstrings provided with the code. The
docstring examples assume that `WATCtools` has been imported as `watc`::

    import WATCtools as watc

Code snippets are indicated by three greater-than signs::

    >>> x = 42
    >>> x = x + 1

Use the built-in ``help`` function to view a function's docstring::

    >>> help(watc.bpf)

Each of the module's methods may be called as::

    >>> watc.bpf(data, filt_band)

or imported individually and called as::

    >>> from WATCtools import bpf
    >>> bpf(data, filt_band)

Available methods
~~~~~~~~~~~~~~~~~
bpf
    Fourier bandpass-filter data matrix or vector
MCCMcalc
    Weighted mean (and median) of the cross-correlation maxima from wlsqva
psf
    Pure state filter data matrix
randc
    Colored noise generator (e.g., 1/f "pink" noise)
 srcLoc
    Estimate source location and trace velocity

Notes
~~~~~
Unless otherwise indicated, all of the methods in this module are assumed
written for use under Python 3.*

"""

# Archival note: per PEP 396, for a module with version-numbered methods,
# the module has no version number.  Each method in this module has a
# version-number embedded as an attribute per PEP 440 as (e.g.):
#       method.__version__ = '3.2.1'

def bpf(x, band):
    """
    Fourier bandpass-filter a data matrix or vector

    This function applies a zero/one mask in the frequency domain to bandpass
    filter a time series.

    Parameters
    ~~~~~~~~~~
    x : array
        Columnar 2D matrix or 1D vector of real time series data
    band : array_like
        Two element object of corner frequencies, relative to DC = 0.0
        and Nyquist = 1.0

    Returns
    ~~~~~~~
    x_bpf : array
        Real, bandpass-filtered version of `x`
    X_bpf : array
        Frequency-domain bandpass-filtered version of `x`

    See Also
    ~~~~~~~~
    ft & ift : from `Z.py`, Sentman-like normalizations of fft & ifft

    Notes
    ~~~~~
    This filter may cause "ringing" at the start and end of the
    filtered time series.

    Version
    ~~~~~~~
    1.0.1 -- 30 Jan 2017

    """
    bpf.__version__ = '1.0.1'
    # (c) 2017 Curt A. L. Szuberla
    # University of Alaska Fairbanks, all rights reserved
    #
    from numpy import array, round, append
    # transform time series
    X_bpf = ft(x)
    # sort band
    band = array(band)
    band.sort()
    # size up the data
    N = x.shape
    # separate odd/even tracks for treating the Nyquist freq
    if N[0]%2:  # N odd
        # convert band to integer-valued indices
        band = round(array(band)*(N[0]-1)/2,decimals=0).astype('int64')
        # Nyquist index (not really, since it would actually be delta_f/2
        # more; half an index) for setting corner frequencies
        Nyq = (N[0]+1)//2
        # negative frequency index for band[2]
        idx_temp = 2*Nyq-band[1]
    else:       # N even, same algorithm as odd block
        band = round(array(band)*N[0]/2,decimals=0).astype('int64')
        Nyq = N[0]//2
        idx_temp = 2*(Nyq)-band[1]+1
    # now odd/even require the same operations
    # corner indices in positive and symmetrically in negative frequencies
    idx_one = append(band,array([idx_temp, N[0]-band[0]+1])-1)
    # corner indices for zero mask
    idxx = (list(range(0,idx_one[0])) +
            list(range(idx_one[1]+1,idx_one[2])) +
            list(range(idx_one[3]+1,N[0])))
    # set Fourier components (and complex conjugate counterparts)
    # outside of band to zero
    X_bpf[idxx] = 0
    x_bpf = ift(X_bpf, allowComplex=False)  # guarantee real output
    return x_bpf, X_bpf


def co_array(rij):
    """
    Form co-array coordinates for given array coordinates

    Parameters
    ~~~~~~~~~~
    rij : array
        (d, n) `n` sensor coordinates as [northing, easting, {elevation}]
        column vectors in `d` dimensions

    Returns
    ~~~~~~~
    xij : array
        (d, n(n-1)//2) co-array, coordinates of the sensor pairing separations

    Version
    ~~~~~~~
    1.0 -- 13 Feb 2017

    """
    co_array.__version__ = '1.0.0'
    # (c) 2017 Curt A. L. Szuberla
    # University of Alaska Fairbanks, all rights reserved
    #
    idx = [(i, j) for i in range(rij.shape[1]-1)
                    for j in range(i+1,rij.shape[1])]
    return rij[:,[i[0] for i in idx]] - rij[:,[j[1] for j in idx]]


def ft(x, n=None, axis=0, norm=None):
    """
    Sentman-like normalization of numpy's fft

    This function calculates the fast Fourier transform of a time series.

    Parameters
    ~~~~~~~~~~
    x : array
        Array of data, can be complex
    n : int, optional
        Length of the transformed axis of the output.
        If `n` is smaller than the length of the input, the input is cropped.
        If it is larger, the input is padded with zeros.  If `n` is not given,
        the length of the input along the axis specified by `axis` is used.
    axis : int, optional
        Axis over which to compute the FFT.  Default is 0.
    norm : {None, "ortho"}, optional
        Normalization mode (see `numpy.fft`). Default is None.

    Returns
    ~~~~~~~
    out : complex array
        The truncated or zero-padded input, transformed along the
        indicated axis.

    See Also
    ~~~~~~~~
    ift : from `Z.py`, Sentman-like normalization of ifft
    numpy.fft : master fft functions

    Notes
    ~~~~~
    This function is just `numpy.fft.fft` with Dr. Davis Sentman
    normalization such that the DC components of the transform are the
    mean of the each column/row in the input time series.

    Version
    ~~~~~~~
    1.0.1 -- 3 Oct 2016

    """
    ft.__version__ = '1.0.1'
    # (c) 2016 Curt A. L. Szuberla
    # University of Alaska Fairbanks, all rights reserved
    #
    from numpy.fft import fft
    # get normalization constant
    N = x.shape
    return fft(x, n, axis, norm)/N[axis]


def ift(X, n=None, axis=0, norm=None, allowComplex=False):
    r"""
    Sentman-like normalization of numpy's ifft

    This function calculates the fast Fourier inverse transform of a
    frequency series.

    Parameters
    ~~~~~~~~~~
    X : array
        Array of data, can be complex
    n : int, optional
        Length of the transformed axis of the output.
        If `n` is smaller than the length of the input, the input is cropped.
        If it is larger, the input is padded with zeros.  If `n` is not given,
        the length of the input along the axis specified by `axis` is used.
    axis : int, optional
        Axis over which to compute the FFT.  Default is 0.
    norm : {None, "ortho"}, optional
        Normalization mode (see `numpy.fft`).  Default is None.
    allowComplex : boolean
        If False, will force real output.  This is useful for suppressing
        cases of spurious (machine precision) imaginary parts where the
        time series are known to be real.  Default is False.

    Returns
    ~~~~~~~
    out : complex array
        The truncated or zero-padded input, transformed along the
        indicated axis.

    See Also
    ~~~~~~~~
    ft : from `Z.py`, Sentman-like normalization of fft
    numpy.fft : master fft functions

    Notes
    ~~~~~
    This function is just `numpy.fft.ifft` with Dr. Davis Sentman
    normalization such that the DC components of the input are the
    mean of the each column/row in the output time series.

    To check on the small imaginary parts of an inverse transform
    one could insert the code below. ::

        from numpy.linalg import norm as Norm
        print(Norm(x.imag))}``

    Version
    ~~~~~~~
    1.0.1 -- 7 Oct 2016

    """
    ift.__version__ = '1.0.1'
    # (c) 2016 Curt A. L. Szuberla
    # University of Alaska Fairbanks, all rights reserved
    #
    from numpy import real
    from numpy.fft import ifft
    # get normalization constant
    N = X.shape
    x = ifft(X, n, axis, norm)*N[axis]
    # force real output, if requested
    if allowComplex:
        return x
    else:
        return real(x)


def MCCMcalc(cmax, wgt=None):
    r"""
    Weighted mean (and median) of the cross-correlation maxima of wlsqva

    This method calculates `MCCM` as the weighted arithmetic mean of the cross-
    correlation maxima output of `wlsqva`, such that channels given a weight
    identical to zero are excluded from the `MCCM` calculation.  Optionally, it
    will calculate `MdCCM` as the weighted median of same.

    Parameters
    ~~~~~~~~~~
    cmax : array
           (n(n-1)//2, ) cross-correlation maxima for `n` channels of data and
           each channel pairing in `tau` (output of `wlsqva`)
    wgt : array
           (n, ) weights corresponding each of the `n` data channels (input to
           `wlsqva`). Default is an array of ones (all channels equally
           weighted and included).

    Returns
    ~~~~~~~
    MCCM : float
           Weighted arithmetic mean of the cross-correlation maxima
    MdCCM : float
           Weighted median of the cross-correlation maxima

    Notes
    ~~~~~
    A typical use of the weights in `wlsqva` is a simple binary scheme (i.e.,
    zeros to exclude channels & ones to include).  If both `cmax` and `wgt`
    are passed to `MCCMcalc`, then the unweighted channels will not
    contribute to the value `MCCM`.  If, however, only `cmax` is passed in such
    a case, the value of `MCCM` will be lower than expected.  This is since
    `wlsqva` sets identically to zero any `cmax` element corresponding to a
    pairing with an unweighted channel.  If `wgt` was passed to `wlsqva`, use
    it in this method, too, in order to ensure the expected numerical behavior
    of `MCCM`.  Non-binary weights in `wgt` will be used to calculate the
    traditionally-defined, weighted arithmetic mean of `cmax`.

    In the calculation of `MdCCM`, the weights are converted to binary -- i.e,
    channels are simply either included or excluded.

    Version
    ~~~~~~~
    1.1 -- 3 Aug 2017

    """
    MCCMcalc.__version__ = '1.1'
    # (c) 2017 Curt A. L. Szuberla
    # University of Alaska Fairbanks, all rights reserved
    #
    from numpy import average, median
    if wgt is None:
        # default is to calculate using all channels & unity for weights
        MCCM = average(cmax)
        MdCCM = median(cmax)
    else:
        from numpy import array
        # first, standard weighted arithmetic mean, allows for
        # arbitrary weights
        Wgt = array([wgt[i]*wgt[j] for i in range(wgt.size-1)
                                       for j in range(i+1,wgt.size)])
        MCCM = average(cmax, weights=Wgt)
        # next, weighted median here only allows binary weights, so
        # just use non-zero weighted channels
        idx = [i for i, e in enumerate(Wgt) if e != 0]
        MdCCM = median(cmax[idx])
    return MCCM, MdCCM


def randc(N, beta=0.0):
    r"""
    Colored noise generator

    This function generates pseudo-random colored noise (power spectrum
    proportional to a power of frequency) via fast Fourier inversion of
    the appropriate amplitudes and complex phases.

    Parameters
    ~~~~~~~~~~
    N : int or tuple of int
        Shape of output array
    beta : float, optional
        Spectrum of output will be proportional to ``f**(-beta)``.
        Default is 0.0.

    Returns
    ~~~~~~~
    out : array
        Colored noise sequences as columns with shape `N`, each normalized
        to zero-mean and unit-variance.  Result is always real valued.

    See Also
    ~~~~~~~~
    ift : from `Z.py`, Sentman-like normalization of ifft
    numpy.fft : master fft functions

    Notes
    ~~~~~
    Spectrum of output will be :math:`\sim 1/f^\beta`.

    White noise is the default (:math:`\beta` = 0); others as pink
    (:math:`\beta` = 1) or brown/surf (:math:`\beta` = 2), ....

    Since the output is zero-mean, the DC spectral component(s) will
    be identically zero.

    Version
    ~~~~~~~
    1.0.1 -- 13 Feb 2017

    """
    randc.__version__ = '1.0.1'
    # (c) 2017 Curt A. L. Szuberla
    # University of Alaska Fairbanks, all rights reserved
    #
    from numpy import (inf, floor, pi, empty, zeros, hstack, arange, exp,
                       vstack, flipud, tile, std)
    from numpy.random import random_sample
    # catch scalar input & form tuple
    if type(N) is int:
        N = (N,)
    # ensure DC component of output will be zero (unless beta == 0, see below)
    if beta < 0:
        c0 = 0
    else:
        c0 = inf
    # catch the case of a 1D array in python, so dimensions act like a matrix
    if len(N) == 1:
        M = (N[0],1) # use M[1] any time # of columns is called for
    else:
        M = N
    # phase array with size (# of unique complex Fourier components,
    # columns of original data)
    n = int(floor((N[0]-1)/2)) # works for odd/even cases
    cPhase = random_sample((n, M[1]))*2*pi
    # Nyquist placeholders
    if N[0]%2:
        # odd case: Nyquist is 1/2 freq step between highest components
        # so it is empty
        cFiller = empty((0,))
        pFiller = empty((0,M[1]))
    else:
        # even case: we have a Nyquist component
        cFiller = N[0]/2
        pFiller = zeros((1,M[1]))
    # noise amplitudes are just indices (unit-offset!!) to be normalized
    # later, phases are arranged as Fourier conjugates
    r = hstack((c0, arange(1,n+1), cFiller, arange(n,0,-1)))
    phasor  = exp(vstack((zeros((1,M[1])), 1j*cPhase, pFiller,
                          -1j*flipud(cPhase))))
    # this is like my cols.m function in MATLAB
    r = tile(r,M[1]).reshape(M[1],N[0]).T**(-beta/2)
    # catch beta = 0 case here to ensure zero DC component
    if not beta:
        r[0] = 0
    # inverse transform to get time series as columns, ensuring real output
    r = ift(r*phasor, allowComplex=False)
    # renormalize r such that mean = 0 & std = 1 (MATLAB dof default used)
    # and return it in its original shape (i.e., a 1D vector, if req'd)
    return r.reshape(N)/std(r, ddof=1)


def srcLoc(rij, tau, nord=2, seedXY_size=0.05, seedV_size=0.3):
    r"""
    Estimate a geopgraphical source location and propagation velocity for an
    event recorded on an array of sensors
    
    Parameters
    ~~~~~~~~~~
    rij : list|array
        (d, n) `n` array coordinates as [easting, northing, {elevation}]
        column vectors in `d` dimensions
    tau : array
        (n*(n-1)/2, ) unique intersensor TDOA information (delays)
    nord : positive int|inf
        Order of the norm to calculate the cost function (optional, default 
        is 2 for the usual Euclidean L2 norm)
    seedXY_size : float
        Geographic seed value (optional, default is 0.05)
    seedXY_size : float
        Propagation velocity seed value (optional, default is 0.3)
        
    Returns
    ~~~~~~~
    Sxyc : array
        (d+1, ) optimized source location as geographic coordinates (same as 
        the columns of `rij`) and propagation speed
    Srtc : array
        (d+1, ) optimized source location as [range, azimuth, {elevation}, 
        propagation speed]

    Notes
    ~~~~~
    This is a Pythonic method for srcLoc that might've been dubbed srcLocLite.
    It takes a naive approach to the seed, ignoring Dr. Arnoult's spacetime
    approach, but takes into account the quirks of the Nelder-Mead optimization
    and prduces a fairly good (if not great) facsimilie of the MATLAB version.

    Version
    ~~~~~~~
    1.0.2 -- 19 Mar 2018
    """
    srcLoc.__version__ = '1.0.2'
    # (c) 2018 Curt A. L. Szuberla
    # University of Alaska Fairbanks, all rights reserved
    #
    from scipy.optimize import minimize as nmOpt
    from numpy.linalg import norm
    # cost function
    def minTau(xyv_trial, tau_o, rij):
        tau_trial = tauCalcSWxy(xyv_trial[-1], xyv_trial[:-1], rij)
        return norm(tau_o - tau_trial, nord)
    # copnstruct naive seed
    xyv_seed = [seedXY_size] * len(rij) + [seedV_size]
    for k in range(0, len(xyv_seed)-1, 2):
        xyv_seed[k] = -xyv_seed[k]
    # perform optimization
    costFn = lambda xyv_trial: minTau(xyv_trial, tau, rij)
    xyv_opt = nmOpt(costFn, xyv_seed, method='Nelder-Mead')
    return xyv_opt.x, rij2rTh(xyv_opt.x[:len(rij)])


