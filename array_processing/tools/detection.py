import numpy as np
from obspy.core import Stream
from .generic import phaseAlignIdx, phaseAlignData


def fstatbland(dtmp, fs, tau):
    r"""
    Calculates the F-statistic and SNR based on Blandford's method (see Notes).

    Args:
        dtmp: ``(m, n)`` array; time series with ``m`` samples from ``n``
            traces as columns
        fs (int or float): Sample rate [Hz]
        tau: ``(n(n-1)//2)`` array; time delays of relative signal arrivals
            (TDOA) for all unique sensor pairings

    Returns:
        tuple: Tuple containing:

        - **fstat** – F-statistic
        - **snr** – SNR

    References:
        Blandford, R. R., 1974. An automatic event detector at the Tonto
        Forest Seismic Observatory. Geophysics, vol. 39, no. 5,
        p. 633–643. `https://library.seg.org/doi/abs/10.1190/1.1440453
        <https://library.seg.org/doi/abs/10.1190/1.1440453>`__
    """

    m, n = dtmp.shape
    wgt = np.ones(n)

    #individual trace offsets from arrival model shifts. Zeros here
    m_offset = [0 for i in range(n)]

    # calculate beam delays
    beam_delays = phaseAlignIdx(tau, fs, wgt, 0)

    # apply shifts, resulting in a zero-padded array
    beam = phaseAlignData(dtmp, beam_delays, wgt, 0, m, m_offset)

    fnum = np.sum(np.sum(beam, axis=1)**2)
    term1 = np.sum(beam, axis=1)/n
    term1_0 = term1
    for i in range(1, n):
        term1 = np.vstack((term1, term1_0))
    fden = np.sum(np.sum((beam.T - term1)**2))
    fstat = (n-1) * fnum/(n * fden)

    #calculate snr based on fstat
    snr = np.sqrt((fstat-1)/n)

    return fstat, snr


def calculate_semblance(data_in):
    r"""
    Calculates the semblance, a measure of multi-channel coherence, following
    the definition of Neidell & Taner (1971). Assumes data are already
    time-shifted to construct the beam.

    Args:
        data_in: Time-shifted ObsPy Stream or time-shifted NumPy array

    Returns:
        Multi-channel coherence defined on :math:`[0, 1]`
    """

    if isinstance(data_in, Stream):
        # check that all traces have the same length
        if len(set([len(tr) for tr in data_in])) != 1:
            raise ValueError('Traces in stream must have same length!')

        n = len(data_in)

        beam = np.sum([tr.data for tr in data_in], axis=0) / n
        beampower = n * np.sum(beam**2)

        avg_power = np.sum(np.sum([tr.data**2 for tr in data_in], axis=0))

    elif isinstance(data_in, np.ndarray):
        n = data_in.shape[0]

        beam = np.sum(data_in, axis=0) / n
        beampower = n * np.sum(beam**2)

        avg_power = np.sum(np.sum([data_in**2], axis=0))

    semblance = beampower / avg_power

    return semblance
