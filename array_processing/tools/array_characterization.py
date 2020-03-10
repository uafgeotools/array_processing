import numpy as np
from scipy import optimize
from scipy.special import gammainc


def arraySig(rij, kmax, sigLevel, p=0.9, velLims=(0.27, 0.36), NgridV=100,
             NgridTh=100, NgridK=100):
    r"""
    Estimate 2-D array uncertainties in trace velocity and back-azimuth, and
    calculate impulse response.

    Args:
        rij: Coordinates (km) of sensors as eastings & northings in a
            ``(2, N)`` array
        kmax (float): Impulse response will be calculated over the range
            [-`kmax`, `kmax`] in :math:`k`-space (1/km)
        sigLevel (float): Variance in time delays (s), typically
            :math:`\sigma_\tau`
        p (float): Confidence limit in uncertainty estimates
        velLims (tuple): Range of trace velocities (km/s) to estimate
            uncertainty over. A single value can be used, but the by default a
            range is used
        NgridV (int): Number of velocities to estimate uncertainties in range
            `velLims`
        NgridTh (int): Number of angles to estimate uncertainties in range
            :math:`[0^\circ, 360^\circ]`
        NgridK (int): Number of :math:`k`-space coordinates to calculate in
            each dimension

    Returns:
        tuple: Tuple containing:

        - **sigV** – Uncertainties in trace velocity (°) as a function of trace
          velocity and back-azimuth as ``(NgridTh, NgridV)`` array
        - **sigTh** – Uncertainties in trace velocity (km/s) as a function of
          trace velocity and back-azimuth as ``(NgridTh, NgridV)`` array
        - **impResp** – Impulse response over grid as ``(NgridK, NgridK)``
          array
        - **vel** – Vector of trace velocities (km/s) for axis in
          ``(NgridV, )`` array
        - **th** – Vector of back azimuths (°) for axis in ``(NgridTh, )``
          array
        - **kvec** – Vector wavenumbers for axes in :math:`k`-space in
          ``(NgridK, )`` array
    """

    # calculate uncertainties
    # preliminaries
    dij = co_array(rij)
    th = np.linspace(0, 360 * (1 - 1 / NgridTh), NgridTh) / 180 * np.pi
    if len(velLims) == 1:
        vel = velLims
    else:
        vel = np.linspace(velLims[0], velLims[1], NgridV)
    Th, Vel = np.meshgrid(th, vel)
    S1 = np.sin(Th) / Vel
    S2 = np.cos(Th) / Vel
    sigTh = np.zeros(Th.shape)
    sigV = sigTh.copy()
    # single-pass calcs
    # calculate eigenvalues/vectors of design matrix (one-time shot)
    C = dij@dij.T
    cii, Ve = np.linalg.eig(C)
    thEigR = np.arctan2(Ve[1, 0], Ve[0, 0])
    R = np.array([[np.cos(thEigR), np.sin(thEigR)],
                  [-np.sin(thEigR), np.cos(thEigR)]])
    # calculate chi2 for desired confidence level
    x2 = chi2(2, 1-p)
    sigS = sigLevel / np.sqrt(cii)
    # prep for loop
    a = np.sqrt(x2) * sigS[0]
    b = np.sqrt(x2) * sigS[1]
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
            sigTh[n, m] = np.abs(np.diff(
                (np.arctan2(eVec[2:, 1], eVec[2:, 0]) * 180 / np.pi - 360)
                % 360))
            if sigTh[n, m] > 180:
                sigTh[n, m] = np.abs(sigTh[n, m] - 360)
            sigV[n, m] = np.abs(np.diff(1 / eExtrm[:2]))

    # prepare impulse response
    impResp, kvec = impulseResp(dij, kmax, NgridK)

    return sigV, sigTh, impResp, vel, th / np.pi * 180, kvec


def impulseResp(dij, kmax, NgridK):
    r"""
    Calculate impulse response of a 2-D array.

    Args:
        dij: Coordinates of co-array of ``N`` sensors in a ``(2, (N*N-1)/2)``
            array
        kmax (float): Impulse response will be calculated over the range
            [-`kmax`, `kmax`] in :math:`k`-space
        NgridK (int): Number of :math:`k`-space coordinates to calculate in
            each dimension

    Returns:
        tuple: Tuple containing:

        - **d** – Impulse response over grid as ``(NgridK, NgridK)`` array
        - **kvec** - Vector wavenumbers for axes in :math:`k`-space in
          ``(NgridK, )`` array
    """

    # pre-allocate grid for :math:`k`-space
    kvec = np.linspace(-kmax, kmax, NgridK)
    Kx, Ky = np.meshgrid(kvec, kvec)
    N = dij.shape[1]
    K = np.vstack((Ky.flatten(), Kx.flatten())).T
    d = 2 * np.cos(K @ dij)
    # last term adds in fact that cos(0)==1 for ignored self-delay terms
    d = np.reshape(np.sum(d, axis=1), (NgridK, NgridK))
    + (1 + np.sqrt(1 + 8 * N)) / 2

    return d, kvec


def rthEllipse(a, b, x0, y0):
    r"""
    Calculate angles subtending, and extremal distances to, a
    coordinate-aligned ellipse from the origin.

    Args:
        a (float): Semi-major axis of ellipse
        b (float): Semi-minor axis of ellipse
        x0 (float): Horizontal center of ellipse
        y0 (float): Vertical center of ellipse

    Returns:
        tuple: Tuple containing:

        - **eExtrm** – Extremal parameters in ``(4, )`` array as

          .. code-block:: none

            [min distance, max distance, min angle (degrees), max angle (degrees)]

        - **eVec** – Coordinates of extremal points on ellipse in ``(4, 2)``
          array as

          .. code-block:: none

            [[x min dist., y min dist.],
             [x max dist., y max dist.],
             [x max angle tangency, y max angle tangency],
             [x min angle tangency, y min angle tangency]]
    """

    # set constants
    A = 2/a**2
    B = 2*x0/a**2
    C = 2/b**2
    D = 2*y0/b**2
    E = (B*x0+D*y0)/2-1
    F = C-A
    G = A/2
    H = C/2
    eExtrm = np.zeros((4,))
    eVec = np.zeros((4, 2))
    eps = np.finfo(np.float64).eps

    # some tolerances for numerical errors
    circTol = 1e8  # is it circular to better than circTol*eps?
    zeroTol = 1e4  # is center along a coord. axis to better than zeroTol*eps?
    magTol = 1e-5  # is a sol'n within ellipse*(1+magTol) (maginification)

    # pursue circular or elliptical solutions
    if np.abs(F) <= circTol * eps:
        # circle
        cent = np.sqrt(x0 ** 2 + y0 ** 2)
        eExtrm[0:2] = cent + np.array([-a, a])
        eVec[0:2, :] = np.array([
            [x0-a*x0/cent, y0-a*y0/cent],
            [x0+a*x0/cent, y0+a*y0/cent]])
    else:
        # ellipse
        # check for trivial distance sol'n
        if np.abs(y0) < zeroTol * eps:
            eExtrm[0:2] = x0 + np.array([-a, a])
            eVec[0:2, :] = np.vstack((eExtrm[0:2], [0, 0])).T
        elif np.abs(x0) < zeroTol * eps:
            eExtrm[0:2] = y0 + np.array([-b, b])
            eVec[0:2, :] = np.vstack(([0, 0], eExtrm[0:2])).T
        else:
            # use dual solutions of quartics to find best, real-valued results
            # solve quartic for y
            fy = F**2*H
            y = quarticEqn(-D*F*(2*H+F)/fy,
                           (B**2*(G+F)+E*F**2+D**2*(H+2*F))/fy,
                           -D*(B**2+2*E*F+D**2)/fy, (D**2*E)/fy)
            y = np.array([y[i] for i in list(np.where(y == np.real(y))[0])])
            xy = B*y / (D-F*y)
            # solve quartic for x
            fx = F**2*G
            x = quarticEqn(B*F*(2*G-F)/fx, (B**2*(G-2*F)+E*F**2+D**2*(H-F))/fx,
                           B*(2*E*F-B**2-D**2)/fx, (B**2*E)/fx)
            x = np.array([x[i] for i in list(np.where(x == np.real(x))[0])])
            yx = D*x / (F*x+B)
            # combine both approaches
            distE = np.hstack(
                (np.sqrt(x ** 2 + yx ** 2), np.sqrt(xy ** 2 + y ** 2)))
            # trap real, but bogus sol's (esp. near Th = 180)
            distEidx = np.where(
                (distE <= np.sqrt(x0 ** 2 + y0 ** 2)
                 + np.max([a, b]) * (1 + magTol))
                & (distE >= np.sqrt(x0 ** 2 + y0 ** 2)
                   - np.max([a, b]) * (1 + magTol)))
            coords = np.hstack(((x, yx), (xy, y))).T
            coords = coords[distEidx, :][0]
            distE = distE[distEidx]
            eExtrm[0:2] = [distE.min(), distE.max()]
            eVec[0:2, :] = np.vstack(
                (coords[np.where(distE == distE.min()), :][0][0],
                 coords[np.where(distE == distE.max()), :][0][0]))
    # angles subtended
    if x0 < 0:
        x0 = -x0
        y = -np.array(quadraticEqn(D ** 2 + B ** 2 * H / G, 4 * D * E,
                                   4 * E ** 2 - B ** 2 * E / G))
        x = -np.sqrt(E / G - H / G * y ** 2)
    else:
        y = -np.array(quadraticEqn(D ** 2 + B ** 2 * H / G, 4 * D * E,
                                   4 * E ** 2 - B ** 2 * E / G))
        x = np.sqrt(E / G - H / G * y ** 2)
    eVec[2:, :] = np.vstack((x, y)).T
    # various quadrant fixes
    if x0 == 0 or np.abs(x0) - a < 0:
        eVec[2, 0] = -eVec[2, 0]
    eExtrm[2:] = np.sort(np.arctan2(eVec[2:, 1], eVec[2:, 0]) / np.pi * 180)

    return eExtrm, eVec


def co_array(rij):
    r"""
    Form co-array coordinates for given array coordinates.

    Args:
        rij: ``(d, n)`` array; ``n`` sensor coordinates as [northing, easting,
            {elevation}] column vectors in ``d`` dimensions

    Returns:
        ``(d, n(n-1)//2)`` co-array, coordinates of the sensor pairing
        separations
    """

    idx = [(i, j) for i in range(rij.shape[1]-1)
           for j in range(i+1, rij.shape[1])]

    return rij[:, [i[0] for i in idx]] - rij[:, [j[1] for j in idx]]


def chi2(nu, alpha, funcTol=1e-10):
    r"""
    Calculate value of a :math:`\chi^2` such that a :math:`\nu`-dimensional
    confidence ellipsoid encloses a fraction :math:`1 - \alpha` of normally
    distributed variable.

    Args:
        nu (int): Degrees of freedom (typically embedding dimension of
            variable)
        alpha (float): Confidence interval such that :math:`\alpha \in [0, 1]`
        funcTol (float): Optimization function evaluation tolerance for
            :math:`\nu \ne 2`

    Returns:
        float: Value of a :math:`\chi^2` enclosing :math:`1 - \alpha`
        confidence region
    """

    if nu == 2:
        # this shorthand owing to Ken Arnoult
        return -2 * np.log(alpha)
    else:
        # but just in case we end up with a nu != 2 situation
        gammaTest = lambda X2test: np.abs(gammainc(nu / 2,
                                                   X2test / 2) - (1-alpha))
        return optimize.fmin(func=gammaTest, x0=1, ftol=funcTol, disp=False)


def cubicEqn(a, b, c):
    r"""
    Roots of cubic equation in the form :math:`x^3 + ax^2 + bx + c = 0`.

    Args:
        a (int or float): Scalar coefficient of cubic equation, can be
            complex
        b (int or float): Same as above
        c (int or float): Same as above

    Returns:
        list: Roots of cubic equation in standard form

    See Also:
        :func:`numpy.roots` — Generic polynomial root finder

    Notes:
        Relatively stable solutions, with some tweaks by Dr. Z,
        per algorithm of Numerical Recipes 2nd ed., :math:`\S` 5.6. Even
        :func:`numpy.roots` can have some (minor) issues; e.g.,
        :math:`x^3 - 5x^2 + 8x - 4 = 0`.
    """

    Q = a*a/9 - b/3
    R = (3*c - a*b)/6 + a*a*a/27
    Q3 = Q*Q*Q
    R2 = R*R
    ao3 = a/3

    # Q & R are real
    if np.isreal([a, b, c]).all():
        # 3 real roots
        if R2 < Q3:
            sqQ = -2 * np.sqrt(Q)
            theta = np.arccos(R / np.sqrt(Q3))
            # This solution first published in 1615 by Viète!
            x = [sqQ * np.cos(theta / 3) - ao3,
                 sqQ * np.cos((theta + 2 * np.pi) / 3) - ao3,
                 sqQ * np.cos((theta - 2 * np.pi) / 3) - ao3]
        # Q & R real, but 1 real, 2 complex roots
        else:
            # this is req'd since np.sign(0) = 0
            if R != 0:
                A = -np.sign(R) * (np.abs(R) + np.sqrt(R2 - Q3)) ** (1 / 3)
            else:
                A = -np.sqrt(-Q3) ** (1 / 3)
            if A == 0:
                B = 0
            else:
                B = Q/A
            # one real root & two conjugate complex ones
            x = [
                (A+B) - ao3,
                -.5 * (A+B) + 1j * np.sqrt(3) / 2 * (A - B) - ao3,
                -.5 * (A+B) - 1j * np.sqrt(3) / 2 * (A - B) - ao3]
    # Q & R complex, so also 1 real, 2 complex roots
    else:
        sqR2mQ3 = np.sqrt(R2 - Q3)
        if np.real(np.conj(R) * sqR2mQ3) >= 0:
            A = -(R+sqR2mQ3)**(1/3)
        else:
            A = -(R-sqR2mQ3)**(1/3)
        if A == 0:
            B = 0
        else:
            B = Q/A
        # one real root & two conjugate complex ones
        x = [
            (A+B) - ao3,
            -.5 * (A+B) + 1j * np.sqrt(3) / 2 * (A - B) - ao3,
            -.5 * (A+B) - 1j * np.sqrt(3) / 2 * (A - B) - ao3
                ]
    # parse real and/or int roots for tidy output
    for k in range(0, 3):
        if np.real(x[k]) == x[k]:
            x[k] = float(np.real(x[k]))
            if int(x[k]) == x[k]:
                x[k] = int(x[k])
    return x


def quadraticEqn(a, b, c):
    r"""
    Roots of quadratic equation in the form :math:`ax^2 + bx + c = 0`.

    Args:
        a (int or float): Scalar coefficient of quadratic equation, can be
            complex
        b (int or float): Same as above
        c (int or float): Same as above

    Returns:
        list: Roots of quadratic equation in standard form

    See Also:
        :func:`numpy.roots` — Generic polynomial root finder

    Notes:
        Stable solutions, even for :math:`b^2 >> ac` or complex coefficients,
        per algorithm of Numerical Recipes 2nd ed., :math:`\S` 5.6.
    """

    # real coefficient branch
    if np.isreal([a, b, c]).all():
        # note np.sqrt(-1) = nan, so force complex argument
        if b:
            # std. sub-branch
            q = -0.5*(b + np.sign(b) * np.sqrt(np.complex(b * b - 4 * a * c)))
        else:
            # b = 0 sub-branch
            q = -np.sqrt(np.complex(-a * c))
    # complex coefficient branch
    else:
        if np.real(np.conj(b) * np.sqrt(b * b - 4 * a * c)) >= 0:
            q = -0.5*(b + np.sqrt(b * b - 4 * a * c))
        else:
            q = -0.5*(b - np.sqrt(b * b - 4 * a * c))
    # stable root solution
    x = [q/a, c/q]
    # parse real and/or int roots for tidy output
    for k in 0, 1:
        if np.real(x[k]) == x[k]:
            x[k] = float(np.real(x[k]))
            if int(x[k]) == x[k]:
                x[k] = int(x[k])
    return x


def quarticEqn(a, b, c, d):
    r"""
    Roots of quartic equation in the form :math:`x^4 + ax^3 + bx^2 +
    cx + d = 0`.

    Args:
        a (int or float): Scalar coefficient of quartic equation, can be
            complex
        b (int or float): Same as above
        c (int or float): Same as above
        d (int or float): Same as above

    Returns:
        list: Roots of quartic equation in standard form

    See Also:
        :func:`numpy.roots` — Generic polynomial root finder

    Notes:
        Stable solutions per algorithm of CRC Std. Mathematical Tables, 29th
        ed.
    """

    # find *any* root of resolvent cubic
    a2 = a*a
    y = cubicEqn(-b, a*c - 4*d, (4*b - a2)*d - c*c)
    y = y[0]
    # find R
    R = np.sqrt(a2 / 4 - (1 + 0j) * b + y)  # force complex in sqrt
    foo = 3*a2/4 - R*R - 2*b
    if R != 0:
        # R is already complex.
        D = np.sqrt(foo + (a * b - 2 * c - a2 * a / 4) / R)
        E = np.sqrt(foo - (a * b - 2 * c - a2 * a / 4) / R)  # ...
    else:
        sqrtTerm = 2 * np.sqrt(y * y - (4 + 0j) * d)  # force complex in sqrt
        D = np.sqrt(foo + sqrtTerm)
        E = np.sqrt(foo - sqrtTerm)
    x = [-a/4 + R/2 + D/2,
         -a/4 + R/2 - D/2,
         -a/4 - R/2 + E/2,
         -a/4 - R/2 - E/2]
    # parse real and/or int roots for tidy output
    for k in range(0, 4):
        if np.real(x[k]) == x[k]:
            x[k] = float(np.real(x[k]))
            if int(x[k]) == x[k]:
                x[k] = int(x[k])

    return x
