
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

    >>> help(watc.wlsqva)

Each of the module's methods may be called as::

    >>> watc.wlsqva(data, rij, Hz)

or imported individually and called as::

    >>> from WATCtools import wlsqva
    >>> wlsqva(data, rij, Hz)

Available methods
~~~~~~~~~~~~~~~~~
bpf
    Fourier bandpass-filter data matrix or vector
co_array
    Form co-array coordinates for given array coordinates
ft
    Fourier transform such that DC component is mean of data
ift
    Inverse Fourier transform complement to `ft`
MCCMcalc
    Weighted mean (and median) of the cross-correlation maxima from wlsqva
psf
    Pure state filter data matrix
randc
    Colored noise generator (e.g., 1/f "pink" noise)
wlsqva
    Weighted least squares solution for slowness

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


def psf(x, p=2, w=3, n=3, window=None):
    r"""
    Pure-state filter a data matrix

    This function uses a generalized coherence estimator to enhance coherent
    channels in an ensemble of time series.

    Parameters
    ~~~~~~~~~~
    x : array
        (m, d) array of real-valued time series data as columns
    p : float, optional
        Level of contrast in filter.  Default is 2.
    w : int, optional
        Width of smoothing window in frequency domain.  Default is 3.
    n : float, optional
        Number of times to smooth in frequency domain.  Default is 3.
    window : function, optional
        Type of smoothing window in frequency domain.  Default is None, which
        results in a triangular window.

    Returns
    ~~~~~~~
    x_ppf : array
        (m, d) real-valued, pure state-filtered version of `x`
    P : array
        (m//2+1, ) degree of polarization (generalized coherence estimate) in
        frequency components of `x` from DC to the Nyquist

    Notes
    ~~~~~
    See any of Samson & Olson's early 1980s papers, or Szuberla's 1997
    PhD thesis, for a full description of the underlying theory.  The code
    implementation's defaults reflect historical values for the smoothing
    window -- a more realistic `w` would be of order :math:`\sqrt{m}\;`
    combined with a smoother window, such as `np.hanning`.  Letting `n=3`
    is a reasonable choice for all window types to ensure confidence in the
    spectral estimates used to construct the filter.

    For `m` samples of `d` channels of data, a (d, d) spectral matrix
    :math:`\mathbf{S}[f]` can be formed at each of the ``m//2+1`` real
    frequency components from DC to the Nyquist.  The generalized coherence
    among all of the `d` channels at each frequency is estimated by

    .. math::

        P[f] = \frac{d \left(\text{tr}\mathbf{S}^2[f]\right) -
        \left(\text{tr}\mathbf{S}[f]\right)^2}
        {\left(d-1\right)\left(\text{tr}\mathbf{S}[f]\right)^2} \;,

    where :math:`\text{tr}\mathbf{S}[f]` is the trace of the spectral
    matrix at frequency :math:`f`.  The filter is constructed by applying
    the following multiplication in the frequency domain

    .. math::

        \hat{\mathbf{X}}[f] = P[f]^p\mathbf{X}[f] \;,

    where :math:`\mathbf{X}[f]` is the Fourier transform component of the all
    channels at frequency :math:`f` and `p` is the level of contrast.  The
    inverse Fourier transform of the matrix :math:`\hat{\mathbf{X}}` gives the
    filtered time series.

    The estimator :math:`\mathbf{P}[f] = 1`, identically, without smoothing in
    the spectral domain (a consequence of the variance in the raw Fourier
    components), but it is bound by :math:`\mathbf{P}[f]\in[0,1]` even with
    smoothing, hence its utility as a multiplicative filter in the frequency
    domain.  Similarly, this bound allows the contrast between channels to be
    enhanced based on their generalized coherence if :math:`p>1\;`.

    Data channels should be pre-processed to have unit-variance, since unlike
    the traditional two-channel magintude squared coherence estimators, the
    generalized coherence estimate can be biased by relative amplitude
    variations among the channels.  To mitigate the effects of smoothing
    complex values into the DC and Nyquist components, they are set to zero
    before computing the inverse transform of :math:`\hat{\mathbf{X}}`.

    Version
    ~~~~~~~
    1.0.2 -- 23 Feb 2017

    """
    psf.__version__ = '1.0.2'
    # (c) 2017 Curt A. L. Szuberla
    # University of Alaska Fairbanks, all rights reserved
    #
    # private functions up front
    def Ssmooth(S, w, n, window):
        # smooth special format of spectral matries as vectors
        from scipy.signal import convolve2d
        for k in range(n):
            # f@#$ing MATLAB treats odd/even differently with mode='full'
            # but the behavior below now matches conv2 exactly
            S = convolve2d(S, window(w).reshape(-1,1),
                           mode='full')[w//2:-w//2+1,:]
        return S
    def triang(N):
        # for historical reasons, the default window shape
        from numpy import bartlett
        return bartlett(N+2)[1:-1]
    # main code block
    from numpy import empty, outer, tile, vstack
    # size up input data
    N, d = x.shape
    # Fourier transform of data matrix by time series columns, retain only
    # the diagonal & above (unique spectral components)
    X = ft(x)
    X = X[:N//2+1, :]
    # form spectral matrix stack in reduced vector form (**significant**
    # speed improvement due to memory problem swith 3D tensor format -- what
    # was too slow in 1995 is still too slow in 2017!)
    # -- pre-allocate stack & fill it in
    Sidx = [(i, j) for i in range(d) for j in range(i, d)]
    S = empty((X.shape[0], d*(d+1)//2), dtype=complex)
    for i in range(X.shape[0]):
        # at each freq w, S_w is outer product of raw Fourier transforms
        # of each column at that freq; select unique components to store
        S_w = outer(X[i,:], X[i,:].conj())
        S[i,:] = S_w[[i[0] for i in Sidx], [j[1] for j in Sidx]]
    # smooth each column of S (i,e., in freq. domain)
    if not window:
        # use default window
        window = triang
    S = Ssmooth(S, w, n, window)
    # trace calculations (notation consistent w traceCalc.m in MATLAB), but
    # since results are positive, semi-definite, real -- return as such
    #  -- diagonal elements
    didx = [i for i in range(len(Sidx)) if Sidx[i][0]==Sidx[i][1]]
    #  -- traceS**2 of each flapjack (really a vector here)
    trS = sum(S[:,didx].real.T)**2
    #  -- trace of each flapjack (ditto, vector), here we recognize that
    #     trace(S@S.T) is just sum square magnitudes of all the
    #     non-redundant components of S, doubling squares of the non-diagonal
    #     elements
    S = (S*(S.conj())*2).real
    S[:,didx] /= 2
    trS2 = sum(S.T)
    # estimate samson-esque polarization estimate (if d==2, same as fowler)
    P = (d*trS2 - trS)/((d-1)*trS)
    # a litle trick here to handle odd/even number of samples and zero-out
    # both the DC & Nyquist (they're both complex-contaminated due to Ssmooth)
    P[0] = 0
    if N%2:
        # odd case: Nyquist is 1/2 freq step between highest components
        fudgeIdx = 0
    else:
        # even case: we have a Nyquist component
        fudgeIdx = 1
        P[-1] = 0
    # apply P as contrast agent to frequency series
    X *= tile(P**p, d).reshape(X.shape[::-1]).T
    # inverse transform X and ensure real output
    x_psf = ift(vstack((X[list(range(N//2+1))],
                          X[list(range(N//2-fudgeIdx,0,-1))].conj())),
                            allowComplex=False)
    return x_psf, P


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


def wlsqva(data, rij, Hz, wgt=None):
    r"""
    Weighted least squares solution for slowness of a plane wave crossing
    an array of sensors

    This function estimates the slowness vector associated with a signal
    passing across an array of sensors under the model assumption that the
    wavefront is planar and the sensor locations are known exactly. The
    slowness is estimated directly from the sensor traces and position
    coordinates. Weights may be applied to each trace to either deselect a
    trace or (de)emphasize its contribution to the least squares solution.

    Parameters
    ~~~~~~~~~~
    data : array
        (m, n) time series with `m` samples from `n` traces as columns
    rij : array
        (d, n) `n` sensor coordinates as [northing, easting, {elevation}]
        column vectors in `d` dimensions
    Hz : float or int
        sample rate
    wgt : list or array
        optional list|vector of relative weights of length `n`
        (0 == exclude trace). Default is None (use all traces with equal
        relative weights).

    Returns
    ~~~~~~~
    vel : float
        signal velocity across array
    az : float or array
        `d = 2`: back azimuth (float) from co-array coordinate origin (º CW
        from N); `d = 3`: back azimuth and elevation angle (array) from
        co-array coordinate origin (º CW from N, º from N-E plane)
    tau : array
        (n(n-1)//2, ) time delays of relative signal arrivals (TDOA) for all
        unique sensor pairings
    cmax : array
        (n(n-1)//2, ) cross-correlation maxima (e.g., for use in MCCMcalc) for
        each sensor pairing in `tau`
    sig_tau : float
        uncertainty estimate for `tau`, also estimate of plane wave model
        assumption violation for non-planar arrivals
    s : array
        (d, ) signal slowness vector via generalized weighted least squares
    xij : array
        (d, n(n-1)//2) co-array, coordinates of the sensor pairing separations

    Raises
    ~~~~~~
    ValueError
        If the input arguments are not consistent with least squares.
    IndexError
        If the input argument dimensions are not consistent.

    Notes
    ~~~~~
    Typical use provides sensor coordinates in km and sample rate in Hz.  This
    will give `vel` in km/s and `s` in s/km; `az` is always in º; `sig_tau`
    and `tau` in s; `xij` in km.

    Cross-correlation maxima in `cij` are normalized to unity for auto-
    correlations at zero lag and set to zero for any pairing with a zero-
    weight trace.

    For a 2D array, a minimum of 3 sensors are required; for 3D, 4 sensors.
    The `data` and `rij` must have a consistent number of sensors. If provided,
    `wgt` must be consistent with the number of sensors.

    This module is self-contained, in the sense that it only requires that
    the `numpy` package is availble, but need not have been imported into the
    workspace -- the module imports from `numpy` as required.

    Examples
    ~~~~~~~~
    Given an approppriate (m, 4) array ``data``, sampled at 20 Hz, the
    following would estimate the slowness using the entire array.

    >>> rij = np.array([[0, 1, 0.5, 0], [1, 0, 0.5, -1]])
    >>> vel, az, tau, cmax, sig_tau, s, xij = wlsqva(data, rij, 20)

    To eliminate the 3d trace from the slowness  estimation, a ``wgt`` list
    can be passed as an argument.

    >>> wgt = [1, 1, 0, 1]
    >>> vel, az, tau, cmax, sig_tau, s, xij = wlsqva(data, rij, 20, wgt)

    Similarly, if the 3d trace is suspect, but should not be completely removed
    from the slowness estimation, it can be given a smaller, relative, weight
    to the other traces.

    >>> wgt = [1, 1, 0.3, 1]
    >>> vel, az, tau, cmax, sig_tau, s, xij = wlsqva(data, rij, 20, wgt)

    Often, only `vel`, `az` are required, so the other returns may be combined
    with extended sequence unpacking in Python 3.X.

    >>> vel, az, *aux_vars = wlsqva(data, rij, 20)

    Version
    ~~~~~~~
    4.0.2 -- 28 Feb 2017

    """
    wlsqva.__version__ = '4.0.2'
    # (c) 2017 Curt A. L. Szuberla
    # University of Alaska Fairbanks, all rights reserved
    #
    # -- input error checking cell (no type checking, just dimensions to help
    #    user identify the problem)
    if data.shape[1] != rij.shape[1]:
        raise IndexError('data.shape[1] != ' + str(rij.shape[1]))
    if rij.shape[1] < rij.shape[0]+1:
        raise ValueError('rij.shape[1] < ' + str(rij.shape[0]+1))
    if rij.shape[0] < 2 or rij.shape[0] > 3:
        raise ValueError('rij.shape[0] != 2|3')
    # --
    from numpy import (array, diag, empty, correlate, argmax, pi, arctan2,
                       abs, eye, sqrt)
    from numpy.linalg import norm, inv
    # size things up
    m, nTraces = data.shape  # number of samples & stations
    dim = rij.shape[0]   # 2D or 3D array?
    # set default wgt, recast as an array, form W array
    if wgt is None:
        # all ones (any constant will do)
        wgt = array([1 for i in range(nTraces)])
        N_w = nTraces
        # -- no need for an error check since all channels kept
    else:
        wgt = array(wgt)
        # -- error checks on wgt done here since now in array form
        if len(wgt) != nTraces:
            raise IndexError('len(wgt) != ' + str(nTraces))
        N_w = sum(wgt != 0)
        if N_w < dim+1:
            raise ValueError('sum(wgt != 0) < ' + str(dim+1))
        # --
    # very handy list comprehension for array processing
    idx = [(i, j, wgt[i]*wgt[j]) for i in range(rij.shape[1]-1)
                    for j in range(i+1,rij.shape[1])]
    # -- co-array is now a one-liner
    xij = rij[:,[i[0] for i in idx]] - rij[:,[j[1] for j in idx]]
    # -- same for generalized weight array
    W = diag([i[2] for i in idx])
    # compute cross correlations across co-array
    N = xij.shape[1]           # number of unique inter-sensor pairs
    cij = empty((m*2-1, N))    # pre-allocated cross-correlation matrix
    for k in range(N):
        # MATLAB's xcorr w/ 'coeff' normalization: unit auto-correlations
        # and save a little time by only calculating on weighted pairs
        if W[k][k]:
            cij[:,k] = (correlate(data[:,idx[k][0]], data[:,idx[k][1]],
               mode='full') / sqrt(sum(data[:,idx[k][0]]*data[:,idx[k][0]]) *
                sum(data[:,idx[k][1]]*data[:,idx[k][1]])))
    # extract cross correlation maxima and associated delays
    cmax = cij.max(axis=0)
    cmax[[i for i in range(N) if W[i][i] == 0]] = 0  # set to zero if not Wvec
    delay = argmax(cij, axis=0)+1 # MATLAB-esque +1 offset here for tau
    # form tau vector
    tau = (m - delay)/Hz
    # form auxiliary arrays for general weighted LS
    X_p = W@xij.T
    tau_p = W@tau
    # calculate least squares slowness vector
    s_p = inv(X_p.T@X_p) @ X_p.T @ tau_p
    # re-cast slowness as geographic vel, az (and phi, if req'd)
    vel = 1/norm(s_p, 2)
    # this converts az from mathematical CCW from E to geographical CW from N
    az = (arctan2(s_p[0],s_p[1])*180/pi-360)%360
    if dim == 3:
        # if 3D, tack on elevation angle to azimuth
        from numpy import hstack
        az = hstack((az, arctan2(s_p[2], norm(s_p[:2], 2))*180/pi))
    # calculate sig_tau (note: moved abs function inside sqrt so that std.
    # numpy.sqrt can be used; only relevant in 3D case w nearly singular
    # solutions, where argument of sqrt is small, but negative)
    N_p = N_w*(N_w-1)/2
    sig_tau_p = sqrt(abs(tau_p @ (eye(N) - X_p @ inv(X_p.T @ X_p) @ X_p.T) @
                       tau_p / (N_p - dim)))
    return vel, az, tau_p, cmax, sig_tau_p, s_p, xij
