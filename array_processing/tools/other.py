import numpy as np


def beamForm(data, rij, Hz, azPhi, vel=0.340, r=None, wgt=None, refTrace=None,
             M=None, Moffset=None):
    """
    Form a "best beam" from the traces of an array

    Parameters
    ~~~~~~~~~~
    data : array
        (m, n) time series with `m` samples from `n` traces as columns
    rij : array
        (d, n) `n` sensor coordinates as [easting, northing, {elevation}]
        column vectors in `d` dimensions
    Hz : float or int
        sample rate
    azPhi : float or list|array
        back azimuth (float) from co-array coordinate origin (º CW from N);
        back azimuth and elevation angle (list) from co-array coordinate
        origin (º CW from N, º from N-E plane)
    vel : float
        optional estimated signal velocity across array. Default is 0.340.
    r : float
        optional range to source from co-array origin. Default is None
        (use plane wave arrival model), If not None, use spherical wave
        arrival model.
    wgt : list or array
        optional list|vector of relative weights of length `n`
        (0 == exclude trace). Default is None (use all traces with equal
        relative weights ``[1 for i in range(nTraces)]``).
    refTrace : int
        optional reference sensor for TDOA information. Default is None
        (use first non-zero-weighted trace).
    M : int
        optional length of best beam result in samples. Default is None
        (use `m` samples, same as input `data`)
    Moffset : list or array
        optional individual trace offsets from arrival model shifts. Default
        is None (use ``[0 for i in range(nTraces)]``)

    Returns
    ~~~~~~~
    beam : array
        (M, ) vector of summed and weighted shifted traces to form a best beam

    Raises
    ~~~~~~
    IndexError
        If the input argument dimensions are not consistent.

    Notes
    ~~~~~
    This beamformer handles planar- or spherical-model arrivals from
    arbitrarily elevated sources incident on 2- or 3D arrays.  Weights are
    relative and normalized upon beam output.  The default value for `vel`
    assumes rij is in units of km (e.g., the speed is in km/s).
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
    """
    Embeds `n` phase aligned traces in a data matrix

    Parameters
    ~~~~~~~~~~
    data : array
        (m, n) time series with `m` samples from `n` traces as columns
    delays : array
        (n, ) vector of shifts as indicies for embedding traces in an array,
        such that trace `i` will begin at index ``out[i]``
    wgt : list or array
       vector of relative weights of length `n` (0 == exclude trace by setting
       to padding value, see `plotFlag`)
    refTrace : int
        reference sensor for TDOA information
    M : int
        length of best beam result in samples (use `m` to let beam be same
        length as inpout traces)
    Moffset : list or array
        individual trace offsets from arrival model shifts (use
        ``[0 for i in range(nTraces)]`` to skip this effect)
    plotFlag : Boolean
        optional flag to indicate output array will be used for plotting
        purposes.  Default is False (pads shifts with zeros; pads with
        np.nan if True).

    Returns
    ~~~~~~~
    data_align : array
        (M, n) array of shifted traces as columns

    Notes
    ~~~~~
    The output of `phaseAlignIdx` is used to calculate the input `delays`.
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
    """
    Calculate shifts required to phase align `n` traces in a data matrix

    Parameters
    ~~~~~~~~~~
    tau : array
        (n(n-1)//2, ) time delays of relative signal arrivals (TDOA) for all
        unique sensor pairings
    Hz : float or int
        sample rate
    wgt : list or array
       vector of relative weights of length `n` (0 == exclude trace)
    refTrace : int
        reference sensor for TDOA information

    Returns
    ~~~~~~~
    delays : array
        (n, ) vector of shifts as indicies for embedding traces in an array,
        such that trace `i` will begin at index ``out[i]``

    Notes
    ~~~~~
    The output of this function is compatible with the inputs of
    `phaseAlignData`.
    """

    # (c) 2017 Curt A. L. Szuberla
    # University of Alaska Fairbanks, all rights reserved
    #
    # -- this is low level code w/out error checks or defaults, designed
    # --  to be called by wrappers that make use of the indices provided
    # solve for number of traces from pairings in tau
    nTraces = int(1 + np.sqrt(1 + 8 * len(tau))) // 2
    # calculate delays (samples) relative to refTrace for each trace
    #   -- first pass grabs delays starting with refTrace as i in ij
    delayIdx = (nTraces*refTrace - refTrace*(refTrace+1)//2,
            nTraces*(refTrace+1) - (refTrace+1)*(refTrace+2)//2)
    delays = np.hstack((0, (tau[range(delayIdx[0], delayIdx[1])] * Hz))
                          ).astype(int)
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
    """
    Calculates theoretical tau vector for a plane wave moving across an array
    of `n` elements

    Parameters
    ~~~~~~~~~~
    vel : float
        signal velocity across array
    azPhi : float or list|array
        back azimuth (float) from co-array coordinate origin (º CW from N);
        back azimuth and elevation angle (array) from co-array coordinate
        origin (º CW from N, º from N-E plane)
    rij : array
        (d, n) `n` element coordinates as [easting, northing, {elevation}]
        column vectors in `d` dimensions

    Returns
    ~~~~~~~
    tau : array
        (n(n-1)//2, ) time delays of relative signal arrivals (TDOA) for all
        unique sensor pairings
    """

    # (c) 2017 Curt A. L. Szuberla
    # University of Alaska Fairbanks, all rights reserved
    #
    dim, nTraces = rij.shape
    if dim == 2:
        rij = np.vstack((rij, np.zeros((1, nTraces))))
    idx = [(i, j) for i in range(rij.shape[1]-1)
                    for j in range(i+1,rij.shape[1])]
    X = rij[:,[i[0] for i in idx]] - rij[:,[j[1] for j in idx]]
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
    """
    Calculates theoretical tau vector for a spherical wave moving across an
    array of `n` elements

    Parameters
    ~~~~~~~~~~
    vel : float
        signal velocity across array
    rAzPhi : list|array
        range to source and back azimuth from co-array coordinate origin
        (º CW from N); range to source, back azimuth and elevation angle
        from co-array coordinate origin (º CW from N, º from N-E plane)
    rij : array
        (d, n) `n` element coordinates as [easting, northing, {elevation}]
        column vectors in `d` dimensions

    Returns
    ~~~~~~~
    tau : array
        (n(n-1)//2, ) time delays of relative signal arrivals (TDOA) for all
        unique sensor pairings
    """

    # (c) 2017 Curt A. L. Szuberla
    # University of Alaska Fairbanks, all rights reserved
    #

    dim, nTraces = rij.shape
    if dim == 2:
        rij = np.vstack((rij, np.zeros((1, nTraces))))
    if len(rAzPhi) == 3:
        phi = rAzPhi[2] / 180 * np.pi
    else:
        phi = 0
    idx = [(i, j) for i in range(rij.shape[1]-1)
                    for j in range(i+1,rij.shape[1])]
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
    """
    Calculates theoretical tau vector for a spherical wave moving across an
    array of `n` elements

    Parameters
    ~~~~~~~~~~
    vel : float
        signal velocity across array
    xy : list|array
        (d, ) source location as 2-D [easting, northing] or 3-D [easting,
        northing, elevation] coordinates
    rij : list|array
        (d, n) `n` element coordinates as [easting, northing, {elevation}]
        column vectors in `d` dimensions

    Returns
    ~~~~~~~
    tau : array
        (n(n-1)//2, ) time delays of relative signal arrivals (TDOA) for all
        unique sensor pairings
    """

    # (c) 2018 Curt A. L. Szuberla
    # University of Alaska Fairbanks, all rights reserved
    #

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
                    for j in range(i+1,rij.shape[1])]
    tau2sensor = np.linalg.norm(rij - np.tile(source, nTraces).reshape(nTraces, 3).T,
                      2, axis=0)/vel
    return tau2sensor[[j[1] for j in idx]] - tau2sensor[[i[0] for i in idx]]
