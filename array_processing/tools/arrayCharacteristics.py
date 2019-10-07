
# -*- coding: utf-8 -*-
r"""
arrayCharacteristics
--------------------
Provides
    1. Array characteristic methods applicable to geophysical sesnor arrays
    2. Support functions for array characteristic calculations
    
How to use the module
---------------------
Documentation is available in docstrings provided with the code. The 
docstring examples assume that `arrayCharacteristics` has been imported 
as `arrChar`::

    import arrayCharacteristics as arrChar

Code snippets are indicated by three greater-than signs::

    >>> x = 42
    >>> x = x + 1

Use the built-in ``help`` function to view a function's docstring::

    >>> help(arrChar.arraySig)

Each of the module's methods may be called as::

    >>> arrChar.impulseResp(dij, kmax, NgridK)

or imported individually and called as::

    >>> from arrayCharacteristics import co_array
    >>> dij = co_array(rij)

Available methods
-----------------
arraySig
    Estimate array uncertainties and impulse response
arraySigPlt
    Plots output of arraySig method
chi2
    Calculates value of :math:`\chi^2` for given confidence level
co_array
    Form co-array coordinates from array coordinates
cubicEqn
    Find roots of :math:`x^3 + ax^2 + bx + c = 0`
impulseResp
    Calculate impulse response of an array
quadraticEqn
    Find roots of :math:`ax^2 + bx + c = 0`
quarticEqn
    Find roots of :math:`x^4 + ax^3 + bx^2 + cx + d = 0`
rthEllipse
    Calculate angles subtending and extremal distances to an ellipse

Notes
-----
All of the methods in this module are written for use under Python 3.*

"""

# Archival note: per PEP 396, for a module with version-numbered methods, 
# the module has no version number.  Each method in this module has a 
# version-number embedded as an attribute per PEP 440 as (e.g.):
#       method.__version__ = '3.2.1'


"""
Created on Wed Sep 18 13:02:15 2019
@author: cas
      (c) 2019 Curt A. L. Szuberla
      University of Alaska Fairbanks, all rights reserved
"""

# unicode test & digital signature of this module
wolf = '\U0001F43A'
print(wolf)

def arraySig(rij, kmax, sigLevel, p=0.9, velLims=(0.27, 0.36), NgridV=100, 
             NgridTh=100, NgridK=100):
    r"""
    Estimate 2D array uncertainties in trace velocity and back azimuth, 
    calculate impulse response
    
    Parameters
    ----------
    rij :   array
        Coorindates (km) of sensors as eastings & northings in a (2, N) array
    kmax : float
        Impulse response will be calculated over the range [-kmax, kmax]
        in k-space (1/km)
    sigLevel : float
        Variance in time delays (s), typically :math:`\sigma_\tau`
    p : float
        Confidence limit in uncertainty estimates (optional, default is 0.9)
    velLims : tuple of float(s)
        Range of trace velocities (km/s) to estimate uncertainty over, single 
        value can be used (optional, default is (0.27, 0.36))
    NgridV : int
        Number of velocities to estimate uncertainties in range `velLims` 
        (optional, default is 100)
    NgridTh : int
        Number of angles to estimate uncertainties in range :math:`[0^\circ, 
        360^\circ)` (optional, default is 100)
    NgridK : int
        Number of k-space coordinates to calculate in each dimension (optional,
        default is 100)
        
    Returns
    -------   
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
    
            
    Version
    -------
    1.0 -- 20 Sep 2019

    """
    arraySig.__version__ = '2.4'
    # (c) 2019 Curt A. L. Szuberla
    # University of Alaska Fairbanks, all rights reserved
    
    from numpy import (linspace, pi, meshgrid, sin, cos, zeros, arctan2, array,
                       sqrt, abs, diff)
    from numpy.linalg import eig
                       
    
    # calculate uncertainties
    # preliminaries
    dij = co_array(rij)
    th = linspace(0, 360*(1-1/NgridTh), NgridTh) / 180*pi
    if len(velLims) == 1:
        vel = velLims
    else:
        vel = linspace(velLims[0], velLims[1], NgridV)
    Th, Vel = meshgrid(th, vel)
    S1 = sin(Th) / Vel
    S2 = cos(Th) / Vel
    sigTh = zeros(Th.shape)
    sigV = sigTh.copy()
    # single-pass calcs
    # calculate eigenvalues/vectors of design matrix (one-time shot)
    C = dij@dij.T
    cii, Ve = eig(C)
    thEigR = arctan2(Ve[1, 0], Ve[0, 0])
    R = array([[cos(thEigR), sin(thEigR)], [-sin(thEigR), cos(thEigR)]])
    # calculate chi2 for desired confidence level
    x2 = chi2(2, 1-p)
    sigS = sigLevel / sqrt(cii)
    # prep for loop
    a = sqrt(x2)*sigS[0]
    b = sqrt(x2)*sigS[1]
    N, M = Th.shape
    # froot loops
    for n in range(N):
        for m in range(M):
            # calculate elliptical extrema
            So = R @ [[S1[n, m]], [S2[n, m]]]
            eExtrm, eVec = rthEllipse(a, b, So[0][0], So[1][0])
            # rotate & recalculate
            eVec = eVec @ R
            # fix up angle calcs
            sigTh[n, m] = abs(diff(
                    (arctan2(eVec[2:, 1], eVec[2:, 0])*180/pi-360)%360
                    ))
            if sigTh[n, m] > 180:
                sigTh[n, m] = abs(sigTh[n, m] - 360)
            sigV[n, m] = abs(diff(1 / eExtrm[:2]))
    # prepare impulse response
    impResp, kvec = impulseResp(dij, kmax, NgridK)
    return sigV, sigTh, impResp, vel, th/pi*180, kvec


def impulseResp(dij, kmax, NgridK):
    """
    Calculate impulse response of a 2D array
    
    Parameters
    ~~~~~~~~~~
    dij : array
        Coorindates of coarray of N-sensors in a (2, (N*N-1)/2) array
    kmax : float
        Impulse response will be calculated over the range [-kmax, kmax]
        in k-space
    NgridK : int
        Number of k-space coordinates to calculate in each dimension
        
    Returns
    ~~~~~~~   
    d : array
        Impulse response over grid as (NgridK, NgridK) array
    kvec : array
        Vector wavenumbers for axes in k-space in (NgridK, ) array
            
    Version
    ~~~~~~~
    1.0 -- 20 Sep 2019

    """
    impulseResp.__version__ = '1.0'
    # (c) 2019 Curt A. L. Szuberla
    # University of Alaska Fairbanks, all rights reserved
    
    from numpy import linspace, meshgrid, vstack, cos, sum, reshape, sqrt
    
    
    # pre-allocate grid for k-space
    kvec  = linspace(-kmax, kmax, NgridK)
    Kx, Ky = meshgrid(kvec, kvec)
    N = dij.shape[1]
    K = vstack((Ky.flatten(), Kx.flatten())).T
    d = 2 * cos(K@dij)
    # last term adds in fact that cos(0)==1 for ignored self-delay terms
    d = reshape(sum(d, axis=1), (NgridK, NgridK)) + (1+sqrt(1 + 8*N))/2
    return d, kvec


def rthEllipse(a, b, x0, y0):
    """
    Calculate angles subtending, and extremal distances to, a coordinate-
    aligned ellipse from the origin
    
    Parameters
    ~~~~~~~~~~
    a : float
        semi-major axis of ellipse
    b : float
        semi-minor axis of ellipse
    x0 : float
        horizontal center of ellipse
    y0 : float
        vertical center of ellipse

    Returns
    ~~~~~~~   
    eExtrm : array
        Extremal parameters in (4, ) array as
        [min distance, max distance, min angle (degrees), max angle (degrees)]
    eVec : array
        Coordinates of extremal points on ellipse in (4, 2) array as
        [[x min dist., y min dist.], [x max dist., y max dist.], 
         [x max angle tangency, y max angle tangency], 
         [x min angle tangency, y min angle tangency]]
        
    Version
    ~~~~~~~
    1.1.1 -- 7 Oct 2019
    
    """
    rthEllipse.__version__ = '1.1.1'
    # (c) 2019 Curt A. L. Szuberla
    # University of Alaska Fairbanks, all rights reserved
    
    from numpy import (zeros, abs, finfo, float64, array, sqrt, where, real,
                       vstack, max, hstack, sort, arctan2, pi)
    
    # set constants
    A = 2/a**2
    B = 2*x0/a**2
    C = 2/b**2
    D = 2*y0/b**2
    E = (B*x0+D*y0)/2-1
    F = C-A
    G = A/2
    H = C/2
    eExtrm = zeros((4, ))
    eVec = zeros((4, 2))
    eps = finfo(float64).eps
    
    # some tolerances for numerical errors
    circTol = 1e8  # is it circular to better than circTol*eps?
    zeroTol = 1e4  # is center along a coord. axis to better than zeroTol*eps?
    magTol = 1e-5  # is a sol'n within ellipse*(1+magTol) (maginification)

    # pursue circular or elliptical solutions
    if abs(F) <= circTol * eps:
        # circle
        cent = sqrt(x0**2 + y0**2)
        eExtrm[0:2]  = cent + array([-a, a])
        eVec[0:2, :] = array([
                [x0-a*x0/cent, y0-a*y0/cent],
                [x0+a*x0/cent, y0+a*y0/cent]
                ])
    else:
        # ellipse
        # check for trivial distance sol'n
        if abs(y0) <  zeroTol * eps:
            eExtrm[0:2] = x0 + array([-a, a])
            eVec[0:2, :] = vstack( (eExtrm[0:2],  [0, 0]) ).T
        elif abs(x0) <  zeroTol * eps:
            eExtrm[0:2] = y0 + array([-b, b])
            eVec[0:2, :] = vstack( ([0, 0], eExtrm[0:2]) ).T
        else:
            # use dual solutions of quartics to find best, real-valued results
            # solve quartic for y
            fy = F**2*H
            y = quarticEqn(-D*F*(2*H+F)/fy, (B**2*(G+F)+E*F**2+D**2*(H+2*F))/fy, 
                           -D*(B**2+2*E*F+D**2)/fy, (D**2*E)/fy)
            y = array([y[i] for i in list(where(y == real(y))[0])])
            xy = B*y / (D-F*y)
            # solve quartic for x
            fx = F**2*G
            x = quarticEqn(B*F*(2*G-F)/fx, (B**2*(G-2*F)+E*F**2+D**2*(H-F))/fx,
                           B*(2*E*F-B**2-D**2)/fx, (B**2*E)/fx)
            x = array([x[i] for i in list(where(x == real(x))[0])])
            yx = D*x / (F*x+B)
            # combine both approaches
            distE = hstack(
                    (sqrt(x**2 + yx**2), sqrt(xy**2 + y**2))
                    )
            # trap real, but bogus sol's (esp. near Th = 180)
            distEidx = where(
                (distE <= sqrt(x0**2 + y0**2)+max([a, b])*(1+magTol)) & 
                (distE >= sqrt(x0**2 + y0**2)-max([a, b])*(1+magTol))
                )
            coords = hstack(((x, yx), (xy, y))).T
            coords = coords[distEidx,:][0]
            distE = distE[distEidx]
            eExtrm[0:2] = [distE.min(),  distE.max()]
            eVec[0:2, :] = vstack(
                    (coords[where(distE == distE.min()), :][0][0],
                     coords[where(distE == distE.max()), :][0][0])
                    )
    # angles subtended
    if x0 < 0:
        x0 = -x0
        y = -array(quadraticEqn(D**2 + B**2*H/G, 4*D*E, 4*E**2 - B**2*E/G))
        x = -sqrt(E/G - H/G*y**2)
    else:
        y = -array(quadraticEqn(D**2 + B**2*H/G, 4*D*E, 4*E**2 - B**2*E/G))
        x = sqrt(E/G - H/G*y**2)
    eVec[2:, :] = vstack((x, y)).T
    # various quadrant fixes
    if x0 == 0 or abs(x0) - a < 0:
        eVec[2, 0] = -eVec[2, 0]
    eExtrm[2:] = sort(arctan2(eVec[2:, 1],eVec[2:, 0])/pi*180)
    return eExtrm, eVec


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
    dij : array
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


def chi2(nu, alpha, funcTol=1e-10):
    
    chi2.__version__ = '1.0'

    if nu == 2:
        from numpy import log
        return -2 * log(alpha) 
    else:
        from scipy import optimize
        from scipy.special import gammainc 
        from numpy import abs
        gammaTest = lambda X2test: abs(gammainc(nu/2, 
                                                X2test/2) - (1-alpha))
        return optimize.fmin(func=gammaTest, x0=1, ftol=funcTol, disp=False)


def quadraticEqn(a,b,c):
    r"""
    Roots of quadratic equation in the form :math:`ax^2 + bx + c = 0`
    
    Parameters
    ~~~~~~~~~~
    a, b, c : int or float, can be complex 
        Scalar coefficients of quadratic equation in standard form
        
    Returns
    ~~~~~~~
    x : list
        Roots of quadratic equation in standard form
         
    See Also
    ~~~~~~~~
    numpy.roots : generic polynomial root finder
    
    Notes
    ~~~~~
    1) Stable solutions, even for :math:`b^2 >> ac` or complex coefficients, 
    per algorithm of NR 2d ed. :math:`\S` 5.6.
    
    Version
    ~~~~~~~
    1.0.1 -- 10 Feb 2017 

    """
    quadraticEqn.__version__ = '1.0.1'
    # (c) 2017 Curt A. L. Szuberla
    # University of Alaska Fairbanks, all rights reserved
    #
    from numpy import isreal, sign, real, sqrt, complex
    # real coefficient branch
    if isreal([a, b, c]).all():
        # note np.sqrt(-1) = nan, so force complex argument
        if b:
            # std. sub-branch
            q = -0.5*(b + sign(b)*sqrt(complex(b*b - 4*a*c)))
        else:
            # b = 0 sub-branch
            q = -sqrt(complex(-a*c))
    # complex coefficient branch
    else:
        from numpy import conj
        if real(conj(b)*sqrt(b*b - 4*a*c)) >= 0:
            q = -0.5*(b + sqrt(b*b - 4*a*c))
        else:
            q = -0.5*(b - sqrt(b*b - 4*a*c))
    # stable root solution
    x = [q/a, c/q]
    # parse real and/or int roots for tidy output
    for k in 0,1:
        if real(x[k]) == x[k]:
            x[k] = float(real(x[k]))
            if int(x[k]) == x[k]:
                x[k] = int(x[k])
    return x


def cubicEqn(a,b,c):
    """
    Roots of cubic equation in the form :math:`x^3 + ax^2 + bx + c = 0`
    
    Parameters
    ~~~~~~~~~~
    a, b, c : int or float, can be complex 
        Scalar coefficients of cubic equation in standard form
        
    Returns
    ~~~~~~~
    x : list
        Roots of cubic equation in standard form
         
    See Also
    ~~~~~~~~
    numpy.roots : generic polynomial root finder
    
    Notes
    ~~~~~
    1) Relatively stable solutions, with some tweaks by Dr. Z, per algorithm 
    of NR 2d ed. :math:`\S` 5.6.  Even np.roots can have some (minor) issues; 
    e.g., :math:`x^3 - 5x^2 + 8x - 4 = 0`.
    
    Version
    ~~~~~~~
    1.0.1 -- 11 Feb 2017 

    """
    # (c) 2017 Curt A. L. Szuberla
    # University of Alaska Fairbanks, all rights reserved
    #
    cubicEqn.__version__ = '1.0.1'
    from numpy import isreal, sqrt, real
    Q = a*a/9 - b/3 
    R = (3*c - a*b)/6 + a*a*a/27
    Q3 = Q*Q*Q
    R2 = R*R
    ao3 = a/3
    # Q & R are real
    if isreal([a, b, c]).all():
        # 3 real roots
        if R2 < Q3:
            from numpy import arccos, cos, pi
            sqQ = -2*sqrt(Q)
            theta = arccos(R/sqrt(Q3))
            # this solution first published in 1615 by Viète!
            x = [
                    sqQ*cos(theta/3) - ao3,
                    sqQ*cos((theta + 2*pi)/3) - ao3,
                    sqQ*cos((theta - 2*pi)/3) - ao3
                    ]
        # Q & R real, but 1 real, 2 complex roots
        else:
            # this is req'd since np.sign(0) = 0
            if R != 0:
                from numpy import sign
                A = -sign(R)*(abs(R) + sqrt(R2 - Q3))**(1/3)
            else:
                A = -sqrt(-Q3)**(1/3)
            if A==0:
                B = 0
            else:
                B = Q/A
            # one real root & two conjugate complex ones
            x = [
                    (A+B) - ao3,
                    -.5*(A+B) + 1j*sqrt(3)/2*(A-B) - ao3,
                    -.5*(A+B) - 1j*sqrt(3)/2*(A-B) - ao3
                    ]
    # Q & R complex, so also 1 real, 2 complex roots
    else:
        sqR2mQ3 = sqrt(R2-Q3)
        from numpy import conj
        if real(conj(R)*sqR2mQ3) >= 0:
            A = -(R+sqR2mQ3)**(1/3)
        else:
            A = -(R-sqR2mQ3)**(1/3)
        if A==0:
            B = 0 
        else: 
            B = Q/A
        # one real root & two conjugate complex ones
        x = [
                (A+B) - ao3,
                -.5*(A+B) + 1j*sqrt(3)/2*(A-B) - ao3,
                -.5*(A+B) - 1j*sqrt(3)/2*(A-B) - ao3
                ]
    # parse real and/or int roots for tidy output
    for k in range(0,3):
        if real(x[k]) == x[k]:
            x[k] = float(real(x[k]))
            if int(x[k]) == x[k]:
                x[k] = int(x[k])
    return x


def quarticEqn(a,b,c,d):
    """
    Roots of quartic equation in the form :math:`x^4 + ax^3 + bx^2 + 
    cx + d = 0`
    
    Parameters
    ~~~~~~~~~~
    a, b, c, d : int or float, can be complex 
        Scalar coefficients of quartic equation in standard form
        
    Returns
    ~~~~~~~
    x : list
        Roots of quartic equation in standard form
         
    See Also
    ~~~~~~~~
    numpy.roots : generic polynomial root finder
    
    Notes
    ~~~~~
    1) Stable solutions per algorithm of CRC  Std. Mathematical 
    Tables, 29th ed.
    
    Version
    ~~~~~~~
    1.0.1 -- 11 Feb 2017 

    """
    # (c) 2017 Curt A. L. Szuberla
    # University of Alaska Fairbanks, all rights reserved
    #
    quarticEqn.__version__ = '1.0.1'
    from numpy import sqrt, real
    # find *any* root of resolvent cubic
    a2 = a*a
    y = cubicEqn(-b, a*c - 4*d, (4*b - a2)*d - c*c)
    y = y[0]
    # find R
    R = sqrt(a2/4 - (1+0j)*b + y) # force complex in sqrt 
    foo = 3*a2/4 - R*R - 2*b
    if R != 0:
        D = sqrt(foo + (a*b - 2*c - a2*a/4)/R) # R is already complex here
        E = sqrt(foo - (a*b - 2*c - a2*a/4)/R) # ...
    else:
        sqrtTerm = 2*sqrt(y*y - (4+0j)*d) # force complex in sqrt
        D = sqrt(foo + sqrtTerm)
        E = sqrt(foo - sqrtTerm)
    x = [
            -a/4 + R/2 + D/2,
            -a/4 + R/2 - D/2,
            -a/4 - R/2 + E/2,
            -a/4 - R/2 - E/2
            ]
    # parse real and/or int roots for tidy output
    for k in range(0,4):
        if real(x[k]) == x[k]:
            x[k] = float(real(x[k]))
            if int(x[k]) == x[k]:
                x[k] = int(x[k])
    return x


def arraySigPlt(rij, sig, sigV, sigTh, impResp, vel, th, kvec, figName=None):
    """
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
    arraySigPlt.__version__ = '1.0'
    
    # std. (if inefficient) plotting import block
    import matplotlib.pyplot as plt
    
    # for plotting methods & scripts
    figFormat = 'png'       # MUCH faster than pdf!!
    figDpi = 600               # good resolution
    plt.rc('text', usetex=True)

    # lower RHS is array geometry
    axRij = plt.subplot(2, 2, 4)
    for h in range(rij.shape[1]):
        axRij.plot(rij[0, h], rij[1, h], 'bp')
    plt.xlabel('km')
    plt.ylabel('km')
    axRij.axis('square')
    axRij.grid()
    
    # upper RHS is impulse reponse
    axImp = plt.subplot(2, 2, 2)
    plt.pcolormesh(kvec, kvec, impResp)
    plt.ylabel('k$_y$ (km$^{-1}$)')
    plt.xlabel('k$_x$ (km$^{-1}$)')
    axImp.axis('square')
    
    # upper RHS is th uncertainty
    plt.subplot(2, 2, 1)
    meshTh = plt.pcolormesh(th, vel, sigTh)
    plt.ylabel('vel. (km/s)')
    plt.xlabel(r'$\theta (^\circ)$')
    cbrTh = plt.colorbar(meshTh, )
    sigStr = str(sig)
    cbrTh.set_label(r'$\delta\theta\;\;\sigma_\tau = $' + sigStr + ' s')

    # lower RHS is vel uncertainty
    plt.subplot(2, 2, 3)
    meshV = plt.pcolormesh(th, vel, sigV)
    plt.ylabel('vel. (km/s)')
    plt.xlabel(r'$\theta (\circ)$')
    cbrV = plt.colorbar(meshV, )
    cbrV.set_label('$\delta v$')
    
    # prepare output & display in iPython workspace
    plt.tight_layout() # IGNORE renderer warning from script! it does just fine
    if figName is not None:
        plt.savefig(figName + '.' + figFormat, format=figFormat, dpi=figDpi)


def arraySigContourPlt(sigV, sigTh, vel, th, trace_v):
    """
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
    import matplotlib.pyplot as plt
    import numpy as np

    tvel_ptr = np.abs(vel - trace_v).argmin()
    sigV_cont = sigV[tvel_ptr,:]
    sigTh_cont = sigTh[tvel_ptr,:]
    theta = np.linspace(0, 2 * np.pi, len(sigV_cont))
    
    
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, subplot_kw={'projection': 'polar'})
    
    ax1.set_theta_direction(-1)
    ax1.set_theta_offset(np.pi/2.0)
    ax1.plot(theta, sigV_cont, color='k', lw=1)
    ax1.set_rmax(sigV_cont.max()*1.1)
    ax1.yaxis.get_major_locator().base.set_params(nbins=6)
    ax1.set_rlabel_position(22.5)  
    ax1.grid(True)
    ax1.set_title('Trace Velocity Uncertainty, V=%.2f' % trace_v, va='bottom', pad=20)
    
    ax2.set_theta_direction(-1)
    ax2.set_theta_offset(np.pi/2.0)
    ax2.plot(theta, sigTh_cont, color='b', lw=1)
    ax2.set_rmax(sigTh_cont.max()*1.1)
    ax2.yaxis.get_major_locator().base.set_params(nbins=6)
    ax2.set_rlabel_position(22.5)  
    ax2.grid(True)
    ax2.set_title('Back-Azimuth Uncertainty, V=%.2f' % trace_v, va='bottom', pad=20)

    return fig