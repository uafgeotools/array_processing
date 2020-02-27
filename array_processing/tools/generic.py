import numpy as np
from scipy.signal import convolve2d
from functools import reduce
from itertools import groupby
from operator import itemgetter


def array_thresh(mcthresh, az_volc, az_diff, mdccm, az, vel):
    r"""
    Find array processing values above multiple set thresholds for MCCM,
    back-azimuth, and trace velocity. Uses default 0.25–0.45 km/s for trace
    velocity thresholds. Also finds consecutive segments that meet thresholds,
    but these values are not currently returned.

    Args:
        mcthresh (float): MCCM or MdCCM threshold (0–1)
        az_volc (float): Back-azimuth to target volcano or source (0–359)
        az_diff (float): Tolerance for back-azimuth from `az_volc`
        mdccm: MdCCM or MCCM values from array processing (0–1)
        az: Back-azimuth values (0–359)
        vel: Trace-velocity values (km/s)

    Returns:
        Indices to time segments that meet set thresholds
    """

    # Use numpy to find where thresholds are exceeded
    mc_good = np.where(mdccm > mcthresh)[0]
    az_good = np.where((az >= az_volc - az_diff) & (az <= az_volc + az_diff))[0]
    vel_good = np.where((vel >= .25) & (vel <= .45))[0]
    igood = reduce(np.intersect1d, (mc_good, az_good, vel_good))

    # Find find number of consecutive values exceeded.
    ranges = []
    nconsec = []
    for k, g in groupby(enumerate(igood), lambda x: x[0]-x[1]):
        group = list(map(itemgetter(1), g))
        ranges.append((group[0], group[-1]))
        nconsec.append(group[-1]-group[0]+1)

    if len(nconsec) > 0:
        consecmax = max(nconsec)
    else:
        consecmax = 0
    print('%d above trheshold, %d consecutive\n' % (len(igood), consecmax))

    return igood


def beamForm(data, rij, Hz, azPhi, vel=0.340, r=None, wgt=None, refTrace=None,
             M=None, Moffset=None):
    r"""
    Form a "best beam" from the traces of an array.

    Args:
        data: ``(m, n)`` array; time series with ``m`` samples from ``n``
            traces as columns
        rij: ``(d, n)`` array; ``n`` sensor coordinates as [easting, northing,
            {elevation}] column vectors in ``d`` dimensions
        Hz (int or float): Sample rate
        azPhi: Back azimuth (float) from co-array coordinate origin (° CW from
            N); back azimuth and elevation angle (list) from co-array
            coordinate origin (° CW from N, ° from N-E plane)
        vel (float): Estimated signal velocity across array
        r (float): Range to source from co-array origin. Default is `None` (use
            plane wave arrival model), If not `None`, use spherical wave
            arrival model
        wgt: Vector of relative weights of length ``n`` (0 == exclude trace).
            Default is `None` (use all traces with equal relative weights ``[1
            for i in range(nTraces)]``)
        refTrace (int): Reference sensor for TDOA information. Default is
            `None` (use first non-zero-weighted trace)
        M (int): Length of best beam result in samples. Default is `None` (use
            ``m`` samples, same as input `data`)
        Moffset: Individual trace offsets from arrival model shifts. Default is
            `None` (use ``[0 for i in range(nTraces)]``)

    Returns:
        ``(M, )`` array of summed and weighted shifted traces to form a best
        beam

    Raises:
        IndexError: If the input argument dimensions are not consistent

    Notes:
        This beamformer handles planar- or spherical-model arrivals from
        arbitrarily elevated sources incident on 2- or 3-D arrays. Weights are
        relative and normalized upon beam output. The default value for `vel`
        assumes that `rij` is in units of km (e.g., the speed is in km/s).
    """

    # (c) 2017 Curt A. L. Szuberla
    # University of Alaska Fairbanks, all rights reserved
    #
    # size things up
    m, nTraces = data.shape
    # -- default parsing & error checking -----------------------------------
    # default weight is unity weight for all traces
    if wgt is None:
        wgt = [1 for i in range(nTraces)]
    else:
        if len(wgt) != nTraces:
            # catch dimension mismatch between tau & wgt
            raise IndexError('len(wgt) != ' + str(nTraces))
    wgt = np.array(wgt)    # require array form here for later operations
    # default refTrace is first non-zero wgt
    if refTrace is None:
        refTrace = np.min(np.where(wgt != 0)) # requires array wgt
    # default Moffset is zero for all traces
    if Moffset is None:
        Moffset = [0 for i in range(nTraces)]
    else:
        if len(Moffset) != nTraces:
            # catch dimension mismatch between tau & Moffset
            raise IndexError('len(Moffset) != ' + str(nTraces))
    # -- end default parsing & error checking -------------------------------
    # planar (far-field) or spherical (near-field) arrival?
    if r is None:
        tau = tauCalcPW(vel, azPhi, rij)
    else:


        # need to unpack & repack azPhi with care
        if np.isscalar(azPhi):
            tau = tauCalcSW(vel, [r, azPhi], rij)
        else:
            tau = tauCalcSW(vel, [r, azPhi[0], azPhi[1]], rij)
    # calculate shifts as samples
    beam_delays = phaseAlignIdx(tau, Hz, wgt, refTrace)
    # apply shifts, resulting in a zero-padded array
    beamMatrix = phaseAlignData(data, beam_delays, wgt, refTrace, M, Moffset)
    # linear algrebra to perform sum & then normalize by weights
    return beamMatrix@wgt / wgt.sum()


def phaseAlignData(data, delays, wgt, refTrace, M, Moffset, plotFlag=False):
    r"""
    Embeds ``n`` phase aligned traces in a data matrix.

    Args:
        data: ``(m, n)`` array; time series with ``m`` samples from ``n``
            traces as columns
        delays: ``(n, )`` array; vector of shifts as indices for embedding
            traces in an array, such that trace ``i`` will begin at index
            ``out[i]``
        wgt: Vector of relative weights of length ``n`` (0 == exclude trace by
            setting to padding value, see `plotFlag`)
        refTrace (int): Reference sensor for TDOA information
        M (int): Length of best beam result in samples (use ``m`` to let beam
            be same length as input traces)
        Moffset: Individual trace offsets from arrival model shifts (use ``[0
            for i in range(nTraces)]`` to skip this effect)
        plotFlag (bool): Flag to indicate output array will be used for
            plotting purposes. Default is `False` (pads shifts with zeros; pads
            with :data:`numpy.nan` if `True`)

    Returns:
        ``(M, n)`` array of shifted traces as columns

    Notes:
        The output of :func:`phaseAlignIdx` is used to calculate the input
        `delays`.
    """

    # (c) 2017 Curt A. L. Szuberla
    # University of Alaska Fairbanks, all rights reserved
    #
    # -- this is low level code w/out error checks or defaults, designed
    # --  to be called by wrappers that make use of the indices provided
    # size up data
    m, nTraces = data.shape
    # if plotting, embed in array of np.nan
    if plotFlag:
        nanOrOne = np.nan
    else:
        nanOrOne = 1
    # correct for negative Moffset elements
    # subtract this to ensure corrected delays is positive,
    # semi-definite and has (at least) one zero element
    maxNegMoffset = min(np.array(Moffset)[np.array(Moffset) <= 0])
    # apply Moffset & correction for negative elements of Moffset
    delays = delays + Moffset - maxNegMoffset
    # start with everything in the window as a default (trim||pad later)
    data_align = np.zeros((max(delays) + m, nTraces)) * nanOrOne
    # embed shifted traces in array
    for k in range(nTraces):
        if wgt[k]:
            data_align[delays[k]:delays[k]+m, k] = data[:, k] * wgt[k]
    # truncate|| pad data_align if M >< m, centered on refTrace
    mp = data_align.shape[0]  # new value for m
    if M is not None and M is not mp:
        alignBounds = [delays[refTrace] + m//2 - M//2,
                       delays[refTrace] + m//2 + M//2]
        # trap round-off errors and force (M, nTraces) data_align
        if alignBounds[1]-alignBounds[0] != M:
            alignBounds[1] += 1
            if not (alignBounds[1]-alignBounds[0])%2:
                alignBounds[0] -= 1
        #  -- LHS (graphically, but actually topside in array-land!)
        if alignBounds[0] < 0:
            # pad LHS of traces w zeros or np.nans
            data_align = np.vstack((np.zeros((-alignBounds[0], nTraces)) * nanOrOne,
                                    data_align))
        elif alignBounds[0] > 0:
            data_align = data_align[alignBounds[0]:]
        #  -- RHS (graphically, but actually bottom in array-land!)
        if alignBounds[1] > mp:
            # pad RHS of traces w zeros or np.nans

            data_align = np.vstack((data_align, np.zeros((alignBounds[1] - mp,
                                                          nTraces)) * nanOrOne))
        elif alignBounds[1] < mp:
            data_align = data_align[:M]
    return data_align


def phaseAlignIdx(tau, Hz, wgt, refTrace):
    r"""
    Calculate shifts required to phase align ``n`` traces in a data matrix.

    Args:
        tau: ``(n(n-1)//2, )`` array; time delays of relative signal arrivals
            (TDOA) for all unique sensor pairings
        Hz (int or float): Sample rate
        wgt: Vector of relative weights of length ``n`` (0 = exclude trace)
        refTrace (int): Reference sensor for TDOA information

    Returns:
        ``(n, )`` array; vector of shifts as indices for embedding traces in an
        array, such that trace ``i`` will begin at index ``out[i]``

    Notes:
        The output of this function is compatible with the inputs of
        :func:`phaseAlignData`.
    """

    # -- this is low level code w/out error checks or defaults, designed
    # --  to be called by wrappers that make use of the indices provided
    # solve for number of traces from pairings in tau
    nTraces = int(1 + np.sqrt(1 + 8 * len(tau))) // 2
    # calculate delays (samples) relative to refTrace for each trace
    #   -- first pass grabs delays starting with refTrace as i in ij
    delayIdx = (nTraces*refTrace - refTrace*(refTrace+1)//2,
                nTraces*(refTrace+1) - (refTrace+1)*(refTrace+2)//2)
    delays = np.hstack((0, (tau[range(delayIdx[0], delayIdx[1])] * Hz))).astype(int)
    # the std. rij list comprehension for unique inter-trace pairs
    tau_ij = [(i, j) for i in range(nTraces) for j in range(i+1, nTraces)]
    #  -- second pass grabs delays with refTrace as j in ij
    preRefTau_idx = [k for k in range(len(tau)) if tau_ij[k][1] == refTrace]
    delays = np.hstack((-tau[preRefTau_idx] * Hz, delays)).astype(int)
    # re-shift delays such that the first sample of the trace requiring the
    # largest shift left relative to the refTrace (hence, largest positive,
    # semi-definite element of delays) has an index of zero; i.e., all traces
    # have a valid starting index for embedding into an array (no padding)
    return -delays + max(delays)


def tauCalcPW(vel, azPhi, rij):
    r"""
    Calculates theoretical tau vector for a plane wave moving across an array
    of ``n`` elements.

    Args:
        vel (float): Signal velocity across array
        azPhi: Back azimuth (float) from co-array coordinate origin (° CW from
            N); back azimuth and elevation angle (array) from co-array
            coordinate origin (° CW from N, ° from N-E plane)
        rij: ``(d, n)`` array; ``n`` element coordinates as [easting, northing,
            {elevation}] column vectors in ``d`` dimensions

    Returns:
        ``(n(n-1)//2, )`` array; time delays of relative signal arrivals (TDOA)
        for all unique sensor pairings
    """

    dim, nTraces = rij.shape
    if dim == 2:
        rij = np.vstack((rij, np.zeros((1, nTraces))))
    idx = [(i, j) for i in range(rij.shape[1]-1)
           for j in range(i+1, rij.shape[1])]
    X = rij[:, [i[0] for i in idx]] - rij[:, [j[1] for j in idx]]
    if np.isscalar(azPhi):
        phi = 0
        az = azPhi
    else:
        phi = azPhi[1] / 180 * np.pi
        az = azPhi[0]
    az = np.pi * (-az / 180 + 0.5)
    s = np.array([np.cos(az), np.sin(az), np.sin(phi)])
    s[:-1] *= np.cos(phi)

    return X.T@(s/vel)


def tauCalcSW(vel, rAzPhi, rij):
    r"""
    Calculates theoretical tau vector for a spherical wave moving across an
    array of ``n`` elements.

    Args:
        vel (float): Signal velocity across array
        rAzPhi: Range to source and back azimuth from co-array coordinate
            origin (° CW from N); range to source, back azimuth and elevation
            angle from co-array coordinate origin (° CW from N, ° from N-E
            plane)
        rij: ``(d, n)`` array; ``n`` element coordinates as [easting, northing,
            {elevation}] column vectors in ``d`` dimensions

    Returns:
        ``(n(n-1)//2, )`` array; time delays of relative signal arrivals (TDOA)
        for all unique sensor pairings
    """

    dim, nTraces = rij.shape
    if dim == 2:
        rij = np.vstack((rij, np.zeros((1, nTraces))))
    if len(rAzPhi) == 3:
        phi = rAzPhi[2] / 180 * np.pi
    else:
        phi = 0
    idx = [(i, j) for i in range(rij.shape[1]-1)
           for j in range(i+1, rij.shape[1])]
    # aw, this is so convolutedly elegant that it must be saved in a
    # comment for posterity!, but the line below it is "simpler"
    # az = -( (rAzPhi[1]/180*pi - 2*pi)%(2*pi) - pi/2  )%(2*pi)
    az = np.pi * (-rAzPhi[1] / 180 + 0.5)
    source = rAzPhi[0] * np.array([np.cos(az), np.sin(az), np.sin(phi)])
    source[:-1] *= np.cos(phi)
    tau2sensor = np.linalg.norm(rij - np.tile(source, nTraces).reshape(nTraces, 3).T,
                                2, axis=0)/vel

    return tau2sensor[[j[1] for j in idx]] - tau2sensor[[i[0] for i in idx]]


def tauCalcSWxy(vel, xy, rij):
    r"""
    Calculates theoretical tau vector for a spherical wave moving across an
    array of ``n`` elements.

    Args:
        vel (float): Signal velocity across array
        xy: ``(d, )`` array; source location as 2-D [easting, northing] or 3-D
            [easting, northing, elevation] coordinates
        rij: ``(d, n)`` array; ``n`` element coordinates as [easting, northing,
            {elevation}] column vectors in ``d`` dimensions

    Returns:
        ``(n(n-1)//2, )`` array; time delays of relative signal arrivals (TDOA)
        for all unique sensor pairings
    """

    dim, nTraces = len(rij), len(rij[0])
    if dim == 2:
        rij = np.vstack((rij, [0] * nTraces))
    else:
        rij = np.vstack((rij,))
    if len(xy) == 2:
        xy0 = 0
    else:
        xy0 = []
    source = np.hstack((xy, xy0))
    idx = [(i, j) for i in range(rij.shape[1]-1)
           for j in range(i+1, rij.shape[1])]
    tau2sensor = np.linalg.norm(rij - np.tile(source, nTraces).reshape(nTraces, 3).T,
                                2, axis=0)/vel

    return tau2sensor[[j[1] for j in idx]] - tau2sensor[[i[0] for i in idx]]


def randc(N, beta=0.0):
    r"""
    Colored noise generator. This function generates pseudo-random colored
    noise (power spectrum proportional to a power of frequency) via fast
    Fourier inversion of the appropriate amplitudes and complex phases.

    Args:
        N (int or tuple): Shape of output array
        beta (float): Spectrum of output will be proportional to ``f**(-beta)``

    Returns:
        Colored noise sequences as columns with shape `N`, each normalized to
        zero-mean and unit-variance. Result is always real valued

    Notes:
        Spectrum of output will be :math:`\sim1/f^\beta`. White noise is the
        default (:math:`\beta = 0`); others are pink (:math:`\beta = 1`) or
        brown/surf (:math:`\beta = 2`). Since the output is zero-mean, the DC
        spectral component(s) will be identically zero.
    """

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
        M = (N[0], 1) # use M[1] any time # of columns is called for
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
    phasor = np.exp(np.vstack((np.zeros((1, M[1])), 1j * cPhase, pFiller,
                               -1j * np.flipud(cPhase))))
    # this is like my cols.m function in MATLAB
    r = np.tile(r, M[1]).reshape(M[1], N[0]).T ** (-beta / 2)
    # catch beta = 0 case here to ensure zero DC component
    if not beta:
        r[0] = 0
    # inverse transform to get time series as columns, ensuring real output
    X = r*phasor
    r = np.real(np.fft.ifft(X, axis=0)*X.shape[0])

    # renormalize r such that mean = 0 & std = 1 (MATLAB dof default used)
    # and return it in its original shape (i.e., a 1D vector, if req'd)
    return r.reshape(N) / np.std(r, ddof=1)


def psf(x, p=2.0, w=3, n=3.0, window=None):
    r"""
    Pure-state filter a data matrix. This function uses a generalized coherence
    estimator to enhance coherent channels in an ensemble of time series.

    Args:
        x: ``(m, d)`` array of real-valued time series data as columns
        p (float): Level of contrast in filter
        w (int): Width of smoothing window in frequency domain
        n (float): Number of times to smooth in frequency domain
        window: Type of smoothing window in frequency domain. Default is
            `None`, which results in a triangular window

    Returns:
        tuple: Tuple containing:

        - **x_psf** – ``(m, d)`` array; real-valued, pure state-filtered
          version of `x`
        - **P** – ``(m//2+1, )`` array; degree of polarization (generalized
          coherence estimate) in frequency components of `x` from DC to the
          Nyquist

    Notes:
        See any of Samson & Olson's early 1980s papers, or Szuberla's 1997 PhD
        thesis, for a full description of the underlying theory. The code
        implementation's defaults reflect historical values for the smoothing
        window — a more realistic `w` would be of order :math:`\sqrt{m}`
        combined with a smoother window, such as :func:`numpy.hanning`. Letting
        `n=3` is a reasonable choice for all window types to ensure confidence
        in the spectral estimates used to construct the filter.

        For :math:`m` samples of :math:`d` channels of data, a ``(d, d)``
        spectral matrix :math:`\mathbf{S}[f]` can be formed at each of the
        ``m//2+1`` real frequency components from DC to the Nyquist. The
        generalized coherence among all of the :math:`d` channels at each
        frequency is estimated by

        .. math::

            P[f] = \frac{d \left(\text{Tr}\,\mathbf{S}^2[f]\right) -
            \left(\text{Tr}\,\mathbf{S}[f]\right)^2}
            {\left(d-1\right)\left(\text{Tr}\,\mathbf{S}[f]\right)^2},

        where :math:`\text{Tr}\,\mathbf{S}[f]` is the trace of the spectral
        matrix at frequency :math:`f`. The filter is constructed by applying
        the following multiplication in the frequency domain

        .. math::

            \hat{\mathbf{X}}[f] = P[f]^p\mathbf{X}[f],

        where :math:`\mathbf{X}[f]` is the Fourier transform component of the
        all channels at frequency :math:`f` and :math:`p` is the level of
        contrast. The inverse Fourier transform of the matrix
        :math:`\hat{\mathbf{X}}` gives the filtered time series.

        The estimator :math:`\mathbf{P}[f] = 1`, identically, without smoothing
        in the spectral domain (a consequence of the variance in the raw
        Fourier components), but it is bound by
        :math:`\mathbf{P}[f]\in[0,1]` even withn smoothing, hence its
        utility as a multiplicative filter in the frequency domain. Similarly,
        this bound allows the contrast between channels to be enhanced based on
        their generalized coherence if :math:`p>1`.

        Data channels should be pre-processed to have unit-variance, since
        unlike the traditional two-channel magnitude squared coherence
        estimators, the generalized coherence estimate can be biased by
        relative amplitude variations among the channels. To mitigate the
        effects of smoothing complex values into the DC and Nyquist components,
        they are set to zero before computing the inverse transform of
        :math:`\hat{\mathbf{X}}`.
    """

    # private functions up front
    def Ssmooth(S, w, n, window):
        # smooth special format of spectral matries as vectors
        for k in range(n):
            # f@#$ing MATLAB treats odd/even differently with mode='full'
            # but the behavior below now matches conv2 exactly
            S = convolve2d(S, window(w).reshape(-1, 1),
                           mode='full')[w//2:-w//2+1, :]
        return S
    def triang(N):
        # for historical reasons, the default window shape
        return np.bartlett(N + 2)[1:-1]

    # size up input data
    N, d = x.shape
    # Fourier transform of data matrix by time series columns, retain only
    # the diagonal & above (unique spectral components)
    Nx = x.shape
    X = np.fft.fft(x, axis=0)/Nx[0]
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
        S[i, :] = S_w[[i[0] for i in Sidx], [j[1] for j in Sidx]]
    # smooth each column of S (i,e., in freq. domain)
    if not window:
        # use default window
        window = triang
    S = Ssmooth(S, w, n, window)
    # trace calculations (notation consistent w traceCalc.m in MATLAB), but
    # since results are positive, semi-definite, real -- return as such
    #  -- diagonal elements
    didx = [i for i in range(len(Sidx)) if Sidx[i][0] == Sidx[i][1]]
    #  -- traceS**2 of each flapjack (really a vector here)
    trS = sum(S[:, didx].real.T)**2
    #  -- trace of each flapjack (ditto, vector), here we recognize that
    #     trace(S@S.T) is just sum square magnitudes of all the
    #     non-redundant components of S, doubling squares of the non-diagonal
    #     elements
    S = (S*(S.conj())*2).real
    S[:, didx] /= 2
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
    XX = np.vstack((X[list(range(N // 2 + 1))],
                    X[list(range(N//2-fudgeIdx, 0, -1))].conj()))
    x_psf = np.real(np.fft.ifft(XX, axis=0)*XX.shape[0])

    return x_psf, P
