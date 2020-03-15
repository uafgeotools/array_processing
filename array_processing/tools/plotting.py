import matplotlib.pyplot as plt
from matplotlib import dates
import numpy as np

from copy import deepcopy
from collections import Counter


def array_plot(st, t, mdccm, vel, baz, ccmplot=False,
               mcthresh=None, sigma_tau=None, stdict=None):
    r"""
    Creates plots for velocity--back-azimuth array processing.

    Args:
        st (:class:`~obspy.core.stream.Stream`): Filtered data. Assumes
            response has been removed.
        t: Array processing time vector.
        mdccm: Array of median cross-correlation maxima.
        vel: Array of trace velocity estimates.
        baz: Array of back-azimuth estimates.
        ccmplot (bool): Toggle plotting the mean/median cross-correlation
            maxima values on a separate subplot in addition to the color scale.
        mcthresh (float): Add a dashed line at this level in the ccmplot
            subplot.
        sigma_tau: Array of :math:`\sigma_\tau` values. If provided, will plot
            the values on a separate subplot.
        stdict (dict): Dropped station pairs from LTS processing. If provided, will plot
            the dropped station pairs on a separate subplot.

    Returns:
        tuple: Tuple containing:

        - **fig** (:class:`~matplotlib.figure.Figure`) – Figure handle.
        - **axs** (Array of :class:`~matplotlib.axes.Axes`) – Axis handles.
    """

    # Specify the colormap.
    cm = 'RdYlBu_r'
    # Colorbar/y-axis limits for MdCCM.
    cax = (0.2, 1)
    # Specify the time vector for plotting the trace.
    tvec = st[0].times('matplotlib')

    # Determine the number and order of subplots.
    num_subplots = 3
    vplot = 1
    bplot = 2
    splot = bplot
    if ccmplot:
        num_subplots += 1
        vplot += 1
        bplot += 1
        splot = bplot
    if sigma_tau is not None or stdict is not None:
        num_subplots += 1
        splot = bplot + 1

    # Start Plotting.
    # Initiate and plot the trace.
    fig, axs = plt.subplots(num_subplots, 1, sharex='col')
    fig.set_size_inches(10, 9)
    axs[0].plot(tvec, st[0].data, 'k')
    axs[0].axis('tight')
    axs[0].set_ylabel('Pressure [Pa]')

    # Plot MdCCM on its own plot.
    if ccmplot:
        sc = axs[1].scatter(t, mdccm, c=mdccm,
                             edgecolors='k', lw=0.3, cmap=cm)
        if mcthresh:
            axs[1].plot([t[0], t[-1]], [mcthresh, mcthresh], 'k--')
        axs[1].axis('tight')
        axs[1].set_xlim(t[0], t[-1])
        axs[1].set_ylim(cax)
        sc.set_clim(cax)
        axs[1].set_ylabel('MdCCM')

    # Plot the trace/apparent velocity.
    sc = axs[vplot].scatter(t, vel, c=mdccm, edgecolors='k', lw=0.3, cmap=cm)
    axs[vplot].set_ylim(0.25, 0.45)
    axs[vplot].set_xlim(t[0], t[-1])
    sc.set_clim(cax)
    axs[vplot].set_ylabel('Trace Velocity\n [km/s]')

    # Plot the back-azimuth.
    sc = axs[bplot].scatter(t, baz, c=mdccm, edgecolors='k', lw=0.3, cmap=cm)
    axs[bplot].set_ylim(0, 360)
    axs[bplot].set_xlim(t[0], t[-1])
    sc.set_clim(cax)
    axs[bplot].set_ylabel('Back-azimuth\n [deg]')

    # Plot sigma_tau if given.
    if sigma_tau is not None:
        if np.isnan(np.sum(sigma_tau)):
            print(r'Sigma_tau values are NaN!')
            axs[splot].scatter(np.array([t[0], t[-1]]),
                                np.array([0.01, 0.01]), c='w')
        else:
            sc = axs[splot].scatter(t, sigma_tau, c=mdccm,
                                     edgecolors='k', lw=0.3, cmap=cm)
        axs[splot].set_xlim(t[0], t[-1])
        sc.set_clim(cax)
        axs[splot].set_ylabel(r'$\sigma_\tau$')

    # Plot dropped station pairs from LTS if given.
    if stdict is not None:
        ndict = deepcopy(stdict)
        n = ndict['size']
        ndict.pop('size', None)
        tstamps = list(ndict.keys())
        tstampsfloat = [float(ii) for ii in tstamps]

        # Set the second colormap for station pairs.
        cm2 = plt.get_cmap('binary', (n-1))
        initplot = np.empty(len(t))
        initplot.fill(1)

        axs[splot].scatter(np.array([t[0], t[-1]]),
                            np.array([0.01, 0.01]), c='w')
        axs[splot].axis('tight')
        axs[splot].set_ylabel('Element [#]')
        axs[splot].set_xlabel('UTC Time')
        axs[splot].set_xlim(t[0], t[-1])
        axs[splot].set_ylim(0.5, n+0.5)
        axs[splot].xaxis_date()
        axs[splot].tick_params(axis='x', labelbottom='on')

        # Loop through the stdict for each flag and plot
        for jj in range(len(tstamps)):
            z = Counter(list(ndict[tstamps[jj]]))
            keys, vals = z.keys(), z.values()
            keys, vals = np.array(list(keys)), np.array(list(vals))
            pts = np.tile(tstampsfloat[jj], len(keys))
            sc2 = axs[splot].scatter(pts, keys, c=vals, edgecolors='k',
                                      lw=0.1, cmap=cm2, vmin=0.5, vmax=n-0.5)

        # Add the horizontal colorbar for station pairs.
        p3 = axs[splot].get_position().get_points().flatten()
        cbaxes2 = fig.add_axes([p3[0], 0.05, p3[2]-p3[0], 0.02])
        hc2 = plt.colorbar(sc2, orientation="horizontal",
                           cax=cbaxes2, ax=axs[splot])
        hc2.set_label('Number of Flagged Element Pairs')
        plt.subplots_adjust(right=0.87, hspace=0.12)

    axs[splot].xaxis_date()
    axs[splot].tick_params(axis='x', labelbottom='on')
    axs[splot].fmt_xdata = dates.DateFormatter('%HH:%MM')
    axs[splot].xaxis.set_major_formatter(dates.DateFormatter("%d-%H:%M"))
    axs[splot].set_xlabel('UTC Time')

    # Add the MdCCM colorbar.
    cbot = axs[splot].get_position().y0
    ctop = axs[1].get_position().y1
    cbaxes = fig.add_axes([0.92, cbot, 0.02, ctop-cbot])
    hc = plt.colorbar(sc, cax=cbaxes)
    hc.set_label('MdCCM')

    return fig, axs


def arraySigPlt(rij, sig, sigV, sigTh, impResp, vel, th, kvec, figName=None):
    r"""
    Plots output of
    :func:`~array_processing.tools.array_characterization.arraySig`.

    Args:
        rij: Coordinates (km) of sensors as eastings & northings in a
            ``(2, N)`` array
        sigLevel (float): Variance in time delays (s), typically
            :math:`\sigma_\tau`
        sigV: Uncertainties in trace velocity (°) as a function of trace
            velocity and back-azimuth as ``(NgridTh, NgridV)`` array
        sigTh: Uncertainties in trace velocity (km/s) as a function of trace
            velocity and back-azimuth as ``(NgridTh, NgridV)`` array
        impResp: Impulse response over grid as ``(NgridK, NgridK)`` array
        vel: Vector of trace velocities (km/s) for axis in ``(NgridV, )``
            array
        th: Vector of back-azimuths (°) for axis in ``(NgridTh, )`` array
        kvec: Vector wavenumbers for axes in :math:`k`-space in ``(NgridK, )`` array
        figName (str): Name of output file, will be written as ``figName.png``
    """

    # Specify output figure file type and plotting resolution.
    figFormat = 'png'
    figDpi = 600

    # Plot array geometry in lower RHS.
    fig = plt.figure()
    axRij = plt.subplot(2, 2, 4)
    for h in range(rij.shape[1]):
        axRij.plot(rij[0, h], rij[1, h], 'bp')
    plt.xlabel('km')
    plt.ylabel('km')
    axRij.axis('square')
    axRij.grid()

    # Plot impulse reponse on upper RHS.
    axImp = plt.subplot(2, 2, 2)
    plt.pcolormesh(kvec, kvec, impResp)
    plt.ylabel('k$_y$ (km$^{-1}$)')
    plt.xlabel('k$_x$ (km$^{-1}$)')
    axImp.axis('square')

    # Plot theta uncertainty on upper LHS.
    plt.subplot(2, 2, 1)
    meshTh = plt.pcolormesh(th, vel, sigTh)
    plt.ylabel('vel. (km/s)')
    plt.xlabel(r'$\theta (^\circ)$')
    cbrTh = plt.colorbar(meshTh, )
    cbrTh.set_label(r'$\delta\theta\;\;\sigma_\tau = $' + str(sig) + ' s')

    # Plot velocity uncertainty on lower LHS.
    plt.subplot(2, 2, 3)
    meshV = plt.pcolormesh(th, vel, sigV)
    plt.ylabel('vel. (km/s)')
    plt.xlabel(r'$\theta (\circ)$')
    cbrV = plt.colorbar(meshV, )
    cbrV.set_label(r'$\delta v$')

    # Prepare output & display in iPython workspace.
    plt.tight_layout()  # IGNORE renderer warning from script! It is fine.
    if figName:
        plt.savefig(figName + '.' + figFormat, format=figFormat, dpi=figDpi)

    return fig


def arraySigContourPlt(sigV, sigTh, vel, th, trace_v):
    r"""
    Plots output of
    :func:`~array_processing.tools.array_characterization.arraySig` onto a
    polar plot for a specified trace velocity.

    Args:
        sigV: Uncertainties in trace velocity (°) as a function of trace
            velocity and back-azimuth as ``(NgridTh, NgridV)`` array
        sigTh: Uncertainties in trace velocity (km/s) as a function of trace
            velocity and back-azimuth as ``(NgridTh, NgridV)`` array
        vel: Vector of trace velocities (km/s) for axis in ``(NgridV, )`` array
        th: Vector of back-azimuths (°) for axis in ``(NgridTh, )`` array
        trace_v (float): Specified trace velocity (km/s) for uncertainty plot

    Returns:
        :class:`~matplotlib.figure.Figure`: Figure handle
    """

    tvel_ptr = np.abs(vel - trace_v).argmin()
    sigV_cont = sigV[tvel_ptr, :]
    sigTh_cont = sigTh[tvel_ptr, :]
    theta = np.linspace(0, 2 * np.pi, len(sigV_cont))

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,
                                   subplot_kw={'projection': 'polar'})

    # Plot trace velocity uncertainty.
    ax1.set_theta_direction(-1)
    ax1.set_theta_offset(np.pi/2.0)
    ax1.plot(theta, sigV_cont, color='k', lw=1)
    ax1.set_rmax(sigV_cont.max()*1.1)
    ax1.yaxis.get_major_locator().base.set_params(nbins=6)
    ax1.set_rlabel_position(22.5)
    ax1.grid(True)
    ax1.set_title('Trace Velocity\n Uncertainty [km/s]\n v=%.2f km/s' % trace_v,
                  va='bottom', pad=20)

    # Plot back-azimuth uncertainty.
    ax2.set_theta_direction(-1)
    ax2.set_theta_offset(np.pi/2.0)
    ax2.plot(theta, sigTh_cont, color='b', lw=1)
    ax2.set_rmax(sigTh_cont.max()*1.1)
    ax2.yaxis.get_major_locator().base.set_params(nbins=6)
    ax2.set_rlabel_position(22.5)
    ax2.grid(True)
    ax2.set_title('Back-Azimuth\n Uncertainty [$^\circ$]\n v=%.2f km/s' % trace_v,
                  va='bottom', pad=20)

    # Adjust subplot spacing to prevent overlap.
    fig.subplots_adjust(wspace=0.4)

    return fig
