import matplotlib.pyplot as plt
from matplotlib import dates
import numpy as np


def array_plot(st, t, mdccm, vel, az, ccmplot=False,
               sigma_tau=False, mcthresh=None):
    """
    Generalized plotting script for velocity - back-azimuth array processing.

    @ Author: David Fee

    Args:
        st: Filtered obspy stream. Assumes response has been removed.
        t: Array processing time vector
        mdccm: Median cross-correlation maxima
        vel: Array of trace velocity estimates
        baz: Array of back-azimuth estimates
        ccmplot: Flag to plot the Mean/Median cross-correlation
            maxima values on the y-axis in addition to the color scale.
        sigma_tau: Array of sigma_tau values
            The flag to add the sigma_tau subplot
        mcthresh: Tuple. Bounds for mdccm.

    Returns:
        fig1: Output figure
        axs1: Output figure axes

    Usage:
        fig1, axs1= array_plot(st, t, mdccm, vel, az, ccmplot=None,
                       sigma_tau=None, mcthresh=None)
    """

    # Colormap
    cm = 'RdYlBu_r'
    # Colorbar/y-axis for MdCCM
    cax = 0.2, 1
    # Time vector
    tvec = dates.date2num(st[0].stats.starttime.datetime)+st[0].times()/86400

    # Determine the number and order of subplots
    num_subplots = 3
    vplot = 1
    bplot = 2
    splot = bplot
    if ccmplot is not False:
        num_subplots += 1
        vplot += 1
        bplot += 1
        splot = bplot  # I'm curious if this line is necessary.
    if sigma_tau is not False:
        num_subplots += 1
        splot = bplot + 1

    # Start Plotting
    # Plot Trace
    fig1, axarr1 = plt.subplots(num_subplots, 1, sharex='col')
    fig1.set_size_inches(10, 9)
    axs1 = axarr1.ravel()
    axs1[0].plot(tvec, st[0].data, 'k')
    axs1[0].axis('tight')
    axs1[0].set_ylabel('Pressure [Pa]')

    # Plot MdCCM on its own plot
    if ccmplot is not False:
        sc = axs1[1].scatter(t, mdccm, c=mdccm,
                             edgecolors='k', lw=0.3, cmap=cm)
        # axs1[1].plot([t[0], t[-1]], [mcthresh[0], mcthresh[1]], 'r--')
        axs1[1].axis('tight')
        axs1[1].set_xlim(t[0], t[-1])
        axs1[1].set_ylim(cax)
        sc.set_clim(cax)
        axs1[1].set_ylabel('MdCCM')

    # Plot the velocity
    sc = axs1[vplot].scatter(t, vel, c=mdccm, edgecolors='k', lw=0.3, cmap=cm)
    axs1[vplot].set_ylim(0.25, 0.45)
    axs1[vplot].set_xlim(t[0], t[-1])
    sc.set_clim(cax)
    axs1[vplot].set_ylabel('Trace Velocity\n [km/s]')

    # Plot the back-azimuth
    sc = axs1[bplot].scatter(t, az, c=mdccm, edgecolors='k', lw=0.3, cmap=cm)
    axs1[bplot].set_ylim(0, 360)
    axs1[bplot].set_xlim(t[0], t[-1])
    sc.set_clim(cax)
    axs1[bplot].set_ylabel('Back-azimuth\n [deg]')

    # Plot sigma_tau
    if sigma_tau is not False:
        sc = axs1[splot].scatter(t, az, c=mdccm,
                                 edgecolors='k', lw=0.3, cmap=cm)
        axs1[splot].set_ylim(0, 360)
        axs1[splot].set_xlim(t[0], t[-1])
        sc.set_clim(cax)
        axs1[splot].set_ylabel(r'$\sigma_\tau$')

    axs1[splot].xaxis_date()
    axs1[splot].tick_params(axis='x', labelbottom='on')
    axs1[splot].fmt_xdata = dates.DateFormatter('%HH:%MM')
    axs1[splot].xaxis.set_major_formatter(dates.DateFormatter("%d-%H:%M"))
    axs1[splot].set_xlabel('UTC Time')

    # Add the MdCCM colorbar
    cbot = axs1[splot].get_position().y0
    ctop = axs1[1].get_position().y1
    cbaxes = fig1.add_axes([0.92, cbot, 0.02, ctop-cbot])
    hc = plt.colorbar(sc, cax=cbaxes)
    hc.set_label('MdCCM')

    return fig1, axs1


def arraySigPlt(rij, sig, sigV, sigTh, impResp, vel, th, kvec, figName=None):
    r"""
    Plots output of arraySig method

    Parameters
    ----------
    rij : array
        Coorindates (km) of sensors as eastings & northings in a (2, N) array
    sigLevel : float
        Variance in time delays (s), typically :math:`\sigma_\tau`
    sigV : array
        Uncertainties in trace velocity :math:`(^\circ)` as a function of trace
        velocity and back azimuth as (NgridTh, NgridV) array
    sigTh : array
        Uncertainties in trace velocity (km/s) as a function of trace velocity
        and back azimuth as (NgridTh, NgridV) array
    impResp : array
        Impulse response over grid as (NgridK, NgridK) array
    vel : array
        Vector of trace velocities (km/s) for axis in (NgridV, ) array
    th : array
        Vector of back azimuths :math:`(^\circ)` for axis in (NgridTh, ) array
    kvec : array
        Vector wavenumbers for axes in k-space in (NgridK, ) array
    figName : str
        Name of output file, will be written as figName.png (optional)
    """

    # For plotting methods & scripts
    figFormat = 'png'       # MUCH faster than pdf!!
    figDpi = 600               # good resolution

    # Lower RHS is array geometry
    fig = plt.figure()
    axRij = plt.subplot(2, 2, 4)
    for h in range(rij.shape[1]):
        axRij.plot(rij[0, h], rij[1, h], 'bp')
    plt.xlabel('km')
    plt.ylabel('km')
    axRij.axis('square')
    axRij.grid()

    # Upper RHS is impulse reponse
    axImp = plt.subplot(2, 2, 2)
    plt.pcolormesh(kvec, kvec, impResp)
    plt.ylabel('k$_y$ (km$^{-1}$)')
    plt.xlabel('k$_x$ (km$^{-1}$)')
    axImp.axis('square')

    # Upper RHS is theta uncertainty
    plt.subplot(2, 2, 1)
    meshTh = plt.pcolormesh(th, vel, sigTh)
    plt.ylabel('vel. (km/s)')
    plt.xlabel(r'$\theta (^\circ)$')
    cbrTh = plt.colorbar(meshTh, )
    sigStr = str(sig)
    cbrTh.set_label(r'$\delta\theta\;\;\sigma_\tau = $' + sigStr + ' s')

    # Lower RHS is velocity uncertainty
    plt.subplot(2, 2, 3)
    meshV = plt.pcolormesh(th, vel, sigV)
    plt.ylabel('vel. (km/s)')
    plt.xlabel(r'$\theta (\circ)$')
    cbrV = plt.colorbar(meshV, )
    cbrV.set_label(r'$\delta v$')

    # Prepare output & display in iPython workspace
    plt.tight_layout()  # IGNORE renderer warning from script! It is fine.
    if figName is not None:
        plt.savefig(figName + '.' + figFormat, format=figFormat, dpi=figDpi)


def arraySigContourPlt(sigV, sigTh, vel, th, trace_v):
    r"""
    Plots output of arraySig method onto a polar plot for a specified trace
    velocity.

    Parameters
    ----------
    sigV : array
        Uncertainties in trace velocity :math:`(^\circ)` as a function of trace
        velocity and back azimuth as (NgridTh, NgridV) array
    sigTh : array
        Uncertainties in trace velocity (km/s) as a function of trace velocity
        and back azimuth as (NgridTh, NgridV) array
    vel : array
        Vector of trace velocities (km/s) for axis in (NgridV, ) array
    th : array
        Vector of back azimuths :math:`(^\circ)` for axis in (NgridTh, ) array
    trace_v : float
        Specified trace velocity (km/s) for uncertainy plot

    Returns
    ~~~~~~~
    fig : figure handle

    author: D. Fee

    """

    tvel_ptr = np.abs(vel - trace_v).argmin()
    sigV_cont = sigV[tvel_ptr, :]
    sigTh_cont = sigTh[tvel_ptr, :]
    theta = np.linspace(0, 2 * np.pi, len(sigV_cont))

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,
                                   subplot_kw={'projection': 'polar'})

    ax1.set_theta_direction(-1)
    ax1.set_theta_offset(np.pi/2.0)
    ax1.plot(theta, sigV_cont, color='k', lw=1)
    ax1.set_rmax(sigV_cont.max()*1.1)
    ax1.yaxis.get_major_locator().base.set_params(nbins=6)
    ax1.set_rlabel_position(22.5)
    ax1.grid(True)
    ax1.set_title('Trace Velocity Uncertainty, V=%.2f' % trace_v,
                  va='bottom', pad=20)

    ax2.set_theta_direction(-1)
    ax2.set_theta_offset(np.pi/2.0)
    ax2.plot(theta, sigTh_cont, color='b', lw=1)
    ax2.set_rmax(sigTh_cont.max()*1.1)
    ax2.yaxis.get_major_locator().base.set_params(nbins=6)
    ax2.set_rlabel_position(22.5)
    ax2.grid(True)
    ax2.set_title('Back-Azimuth Uncertainty, V=%.2f' % trace_v,
                  va='bottom', pad=20)

    return fig
