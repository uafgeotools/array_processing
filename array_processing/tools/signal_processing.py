import numpy as np
from scipy.signal import convolve2d


def randc(N, beta=0.0):
    """
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
    """

    # (c) 2017 Curt A. L. Szuberla
    # University of Alaska Fairbanks, all rights reserved
    #

    # catch scalar input & form tuple
    if type(N) is int:
        N = (N,)
    # ensure DC component of output will be zero (unless beta == 0, see below)
    if beta < 0:
        c0 = 0
    else:
        c0 = np.inf
    # catch the case of a 1D array in python, so dimensions act like a matrix
    if len(N) == 1:
        M = (N[0],1) # use M[1] any time # of columns is called for
    else:
        M = N
    # phase array with size (# of unique complex Fourier components,
    # columns of original data)
    n = int(np.floor((N[0] - 1) / 2)) # works for odd/even cases
    cPhase = np.random.random_sample((n, M[1])) * 2 * np.pi
    # Nyquist placeholders
    if N[0]%2:
        # odd case: Nyquist is 1/2 freq step between highest components
        # so it is empty
        cFiller = np.empty((0,))
        pFiller = np.empty((0, M[1]))
    else:
        # even case: we have a Nyquist component
        cFiller = N[0]/2
        pFiller = np.zeros((1, M[1]))
    # noise amplitudes are just indices (unit-offset!!) to be normalized
    # later, phases are arranged as Fourier conjugates
    r = np.hstack((c0, np.arange(1, n + 1), cFiller, np.arange(n, 0, -1)))
    phasor  = np.exp(np.vstack((np.zeros((1, M[1])), 1j * cPhase, pFiller,
                                      -1j * np.flipud(cPhase))))
    # this is like my cols.m function in MATLAB
    r = np.tile(r, M[1]).reshape(M[1], N[0]).T ** (-beta / 2)
    # catch beta = 0 case here to ensure zero DC component
    if not beta:
        r[0] = 0
    # inverse transform to get time series as columns, ensuring real output
    r = ift(r*phasor, allowComplex=False)
    # renormalize r such that mean = 0 & std = 1 (MATLAB dof default used)
    # and return it in its original shape (i.e., a 1D vector, if req'd)
    return r.reshape(N) / np.std(r, ddof=1)


def psf(x, p=2, w=3, n=3, window=None):
    """
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
    """

    # (c) 2017 Curt A. L. Szuberla
    # University of Alaska Fairbanks, all rights reserved
    #
    # private functions up front
    def Ssmooth(S, w, n, window):
        # smooth special format of spectral matries as vectors
        for k in range(n):
            # f@#$ing MATLAB treats odd/even differently with mode='full'
            # but the behavior below now matches conv2 exactly
            S = convolve2d(S, window(w).reshape(-1,1),
                           mode='full')[w//2:-w//2+1,:]
        return S
    def triang(N):
        # for historical reasons, the default window shape
        return np.bartlett(N + 2)[1:-1]
    # main code block


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
    S = np.empty((X.shape[0], d * (d + 1) // 2), dtype=complex)
    for i in range(X.shape[0]):
        # at each freq w, S_w is outer product of raw Fourier transforms
        # of each column at that freq; select unique components to store
        S_w = np.outer(X[i, :], X[i, :].conj())
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
    X *= np.tile(P ** p, d).reshape(X.shape[::-1]).T
    # inverse transform X and ensure real output
    x_psf = ift(np.vstack((X[list(range(N // 2 + 1))],
                              X[list(range(N//2-fudgeIdx,0,-1))].conj())),
                allowComplex=False)
    return x_psf, P
