import matplotlib.pyplot as plt
from matplotlib import dates
import numpy as np

from copy import deepcopy
from collections import Counter


def array_plot(st, t, mdccm, vel, baz, ccmplot=False,
               mcthresh=None, sigma_tau=None, stdict=None):
    """
    Creates plots for velocity - back-azimuth array processing.

    Args:
        st: Filtered obspy stream. Assumes response has been removed.
        t: Array processing time vector.
        mdccm: Array of median cross-correlation maxima.
        vel: Array of trace velocity estimates.
        baz: Array of back-azimuth estimates.
        ccmplot: Boolean flag to plot the Mean/Median cross-correlation
            maxima values on the y-axis in addition to the color scale.
        mcthresh: Add a dashed line at the [float] level
            in the ccmplot subplot.
        sigma_tau: Array of sigma_tau values.
            The flag to add the sigma_tau subplot.
        stdict: Array of dropped station pairs from LTS processing.
            The flag to add the stdict subplot.

    Returns:
        fig1: Output figure.
        axs1: Output figure axes.

    Usage:
        fig1, axs1= array_plot(st, t, mdccm, vel, baz, ccmplot=False,
                       mcthresh=None, sigma_tau=None, stdict=None)
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
    fig1, axs1 = plt.subplots(num_subplots, 1, sharex='col')
    fig1.set_size_inches(10, 9)
    axs1[0].plot(tvec, st[0].data, 'k')
    axs1[0].axis('tight')
    axs1[0].set_ylabel('Pressure [Pa]')

    # Plot MdCCM on its own plot.
    if ccmplot:
        sc = axs1[1].scatter(t, mdccm, c=mdccm,
                             edgecolors='k', lw=0.3, cmap=cm)
        if mcthresh:
            axs1[1].plot([t[0], t[-1]], [mcthresh, mcthresh], 'k--')
        axs1[1].axis('tight')
        axs1[1].set_xlim(t[0], t[-1])
        axs1[1].set_ylim(cax)
        sc.set_clim(cax)
        axs1[1].set_ylabel('MdCCM')

    # Plot the trace/apparent velocity.
    sc = axs1[vplot].scatter(t, vel, c=mdccm, edgecolors='k', lw=0.3, cmap=cm)
    axs1[vplot].set_ylim(0.25, 0.45)
    axs1[vplot].set_xlim(t[0], t[-1])
    sc.set_clim(cax)
    axs1[vplot].set_ylabel('Trace Velocity\n [km/s]')

    # Plot the back-azimuth.
    sc = axs1[bplot].scatter(t, baz, c=mdccm, edgecolors='k', lw=0.3, cmap=cm)
    axs1[bplot].set_ylim(0, 360)
    axs1[bplot].set_xlim(t[0], t[-1])
    sc.set_clim(cax)
    axs1[bplot].set_ylabel('Back-azimuth\n [deg]')

    # Plot sigma_tau if given.
    if sigma_tau is not None:
        sc = axs1[splot].scatter(t, sigma_tau, c=mdccm,
                                 edgecolors='k', lw=0.3, cmap=cm)
        axs1[splot].set_xlim(t[0], t[-1])
        sc.set_clim(cax)
        axs1[splot].set_ylabel(r'$\sigma_\tau$')

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

        axs1[splot].scatter(np.array([t[0], t[-1]]),
                            np.array([0.01, 0.01]), c='w')
        axs1[splot].axis('tight')
        axs1[splot].set_ylabel('Element [#]')
        axs1[splot].set_xlabel('UTC Time')
        axs1[splot].set_xlim(t[0], t[-1])
        axs1[splot].set_ylim(0.5, n+0.5)
        axs1[splot].xaxis_date()
        axs1[splot].tick_params(axis='x', labelbottom='on')

        # Loop through the stdict for each flag and plot
        for jj in range(len(tstamps)):
            z = Counter(list(ndict[tstamps[jj]]))
            keys, vals = z.keys(), z.values()
            keys, vals = np.array(list(keys)), np.array(list(vals))
            pts = np.tile(tstampsfloat[jj], len(keys))
            sc2 = axs1[splot].scatter(pts, keys, c=vals, edgecolors='k',
                                      lw=0.1, cmap=cm2, vmin=0.5, vmax=n-0.5)

        # Add the horizontal colorbar for station pairs.
        p3 = axs1[splot].get_position().get_points().flatten()
        cbaxes2 = fig1.add_axes([p3[0], 0.05, p3[2]-p3[0], 0.02])
        hc2 = plt.colorbar(sc2, orientation="horizontal",
                           cax=cbaxes2, ax=axs1[splot])
        hc2.set_label('Number of Flagged Element Pairs')
        plt.subplots_adjust(right=0.87, hspace=0.12)

    axs1[splot].xaxis_date()
    axs1[splot].tick_params(axis='x', labelbottom='on')
    axs1[splot].fmt_xdata = dates.DateFormatter('%HH:%MM')
    axs1[splot].xaxis.set_major_formatter(dates.DateFormatter("%d-%H:%M"))
    axs1[splot].set_xlabel('UTC Time')

    # Add the MdCCM colorbar.
    cbot = axs1[splot].get_position().y0
    ctop = axs1[1].get_position().y1
    cbaxes = fig1.add_axes([0.92, cbot, 0.02, ctop-cbot])
    hc = plt.colorbar(sc, cax=cbaxes)
    hc.set_label('MdCCM')

    return fig1, axs1


def arraySigPlt(rij, sig, sigV, sigTh, impResp, vel, th, kvec, figName=None):
    r"""
    Plots output of arraySig method.

    Parameters
    ----------
    rij : array
        Coordinates (km) of sensors as eastings & northings in a (2, N) array.
    sigLevel : float
        Variance in time delays (s), typically :math:`\sigma_\tau`.
    sigV : array
        Uncertainties in trace velocity :math:`(^\circ)` as a function of trace
        velocity and back-azimuth as (NgridTh, NgridV) array.
    sigTh : array
        Uncertainties in trace velocity (km/s) as a function of trace velocity
        and back-azimuth as (NgridTh, NgridV) array.
    impResp : array
        Impulse response over grid as (NgridK, NgridK) array.
    vel : array
        Vector of trace velocities (km/s) for axis in (NgridV, ) array.
    th : array
        Vector of back-azimuths :math:`(^\circ)` for axis in (NgridTh, ) array.
    kvec : array
        Vector wavenumbers for axes in k-space in (NgridK, ) array.
    figName : str
        Name of output file, will be written as figName.png (optional).
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
    Plots output of arraySig method onto a polar plot for a specified trace
    velocity.

    Parameters
    ----------
    sigV : array
        Uncertainties in trace velocity :math:`(^\circ)` as a function of trace
        velocity and back-azimuth as (NgridTh, NgridV) array.
    sigTh : array
        Uncertainties in trace velocity (km/s) as a function of trace velocity
        and back-azimuth as (NgridTh, NgridV) array.
    vel : array
        Vector of trace velocities (km/s) for axis in (NgridV, ) array.
    th : array
        Vector of back-azimuths :math:`(^\circ)` for axis in (NgridTh, ) array.
    trace_v : float
        Specified trace velocity (km/s) for uncertainy plot.

    Returns
    ~~~~~~~
    fig : figure handle

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
    ax1.set_title('Trace Velocity Uncertainty,\nV=%.2f' % trace_v,
                  va='bottom', pad=20)

    # Plot back-azimuth uncertainty.
    ax2.set_theta_direction(-1)
    ax2.set_theta_offset(np.pi/2.0)
    ax2.plot(theta, sigTh_cont, color='b', lw=1)
    ax2.set_rmax(sigTh_cont.max()*1.1)
    ax2.yaxis.get_major_locator().base.set_params(nbins=6)
    ax2.set_rlabel_position(22.5)
    ax2.grid(True)
    ax2.set_title('Back-Azimuth Uncertainty,\nV=%.2f' % trace_v,
                  va='bottom', pad=20)

    # Adjust subplot spacing to prevent overlap.
    fig.subplots_adjust(wspace=0.4)

    return fig
