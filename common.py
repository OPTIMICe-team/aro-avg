#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 14:59:45 2023

This module collects functions that are in general not specific for a
particular scattering method. Both DDA and TMM (and also others) are dependent
on the transformations implemented here. 

@author: dori
"""

import numpy as np
import matplotlib.pyplot as plt
from geometry3D import pol2cart as polar2cart

PRECISION = '{:1.3f}' # the floating point resolution of the parameters

def dB(x):
    """ returns x converted in decibels. Works on arrays.
        Does not perform checks on the input, negative values are not valid

    Parameters
    ----------
    x : float
        Linear quantity. Must be positive

    Returns
    -------
    float
        x in dB
    """
    return 10.0*np.log10(x)


def find_backward(th0, phi0):
    """ Finds the backward direction in polar coordinates in degrees

    Parameters
    ----------
    th0, phi0 : float (degrees)
        Polar and azimuth angles of the input vector

    Returns
    -------
    th, phi : floats (degrees)
        Polar and azimuth angle of the vector opposite to (th0, phi0)
    """
    
    th = (180.0 - th0) #% 180.0
    phi = (phi0 + 180.0) % 360.0
    
    return th, phi

"""
def polar2cart(th, ph, r=1.0):
    t = np.deg2rad(th)
    p = np.deg2rad(ph)
    z = r*np.cos(t)
    R = r*np.sin(t)
    x = R*np.cos(p)
    y = R*np.sin(p)
    return x, y, z

def cart2polar(x, y, z):
    r = np.sqrt(x*x + y*y + z*z)
    t = np.arccos(z/r)
    th = np.rad2deg(t)
    R = r*np.sin(t)
    p = np.arctan2(y/R, x/R)
    if p<0:
        p = 2.0*np.pi + p
    ph = np.rad2deg(p)
    return th, ph
"""


def plotDirection(th0, phi0):
    th, phi = find_backward(th0, phi0)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x0, y0, z0 = polar2cart(th0, phi0)
    x, y, z = polar2cart(th, phi)
    
    print("dot prod {}".format(x*x0+y*y0+z*z0))
    
    ax.plot([0, x0], [0, y0], [0, z0], label='fwd')
    ax.plot([0, x], [0, y], [0, z], label='bck')
    lim = 1.5
    lims = (-lim, lim)
    ax.set_xlim3d(lims)
    ax.set_ylim3d(lims)
    ax.set_zlim3d(lims)
    ax.set_xlabel('X')



def crop_params(m, scalars, angles, dipole_resolution=-1):
    """ 
    This function is more flexible than crop_parameters and accepts a variable
    number of parameters for both angles and general floats, but because of
    this it will leave out the computation of the backward direction.

    Parameters
    ----------
    m : complex
        refractive index. The imaginary part is treated differently
        with respect to all the quantities in scalars because, given the
        typical range of values for ice in the microwave a e-3 factor is
        assumed
    scalars : float
        All scalars that are not angles. Lengths are assumed to be millimeters
    angles : float (degrees)
        All angles in degrees.
    dipole_resolution : float, optional
        Also dipole resolution is treated differently. Since it is usually in
        in the order of micrometers and we took lengths in millimeters a factor
        of e-3 is also assumed here. The default is -1 which would return an
        error if used with DDA calculations.

    Returns
    -------
    strings : iterable
        contains string representation in fixed format for refractive index,
        all scalars, all angles and the dipole resolution. scalars and angles
        are returned in sub-iterables
    floats : iterable
        contains float representation in fixed format for refractive index,
        all scalars and all angles. scalars and angles are returned
        in sub-iterables
    """
    
    sca_str = [PRECISION.format(a) for a in scalars]
    ang_str = [PRECISION.format(a) for a in angles]

    # special inputs with different treatment
    mr = PRECISION.format(m.real)
    mi = PRECISION.format(m.imag*1.0e3) + 'e-3'
    drs = PRECISION.format(dipole_resolution*1.0e3) + 'e-3'

    strings = (mr, mi, drs, sca_str, ang_str)

    m = complex(float(mr), float(mi))
    res = float(drs)
    
    floats = (m, res,
              [float(a) for a in sca_str],
              [float(a) for a in ang_str])

    
#    thet, phi = find_backward(thet0, phi0) # to be done outside
    
    return strings, floats



# DEPRECATED
def crop_parameters(radius, wavelength, m, axis_ratio,
                    thet0, phi0,
                    alpha, beta,
                    dipole_resolution=-1):
    rs = PRECISION.format(radius)
    ws = PRECISION.format(wavelength)
    mr = PRECISION.format(m.real)
    mi = PRECISION.format(m.imag*1.0e3) + 'e-3'
    ars = PRECISION.format(axis_ratio)
    drs = PRECISION.format(dipole_resolution*1.0e3) + 'e-3'
    t0s = PRECISION.format(thet0)
    #ts = PRECISION.format(thet)
    p0s = PRECISION.format(phi0)
    #ps = PRECISION.format(phi)
    als = PRECISION.format(alpha)
    bes = PRECISION.format(beta)
    
    radius = float(rs)
    wl = float(ws)
    m = complex(float(mr), float(mi))
    axis_ratio = float(ars)
    res = float(drs)
    thet0 = float(t0s)
    #thet = np.deg2rad(float(ts))
    phi0 = float(p0s)
    #phi = np.deg2rad(float(ps))
    alpha = float(als)
    beta = float(bes)
    
    #print('cropped ', thet0, phi0, alpha, beta)
    
    thet, phi = find_backward(thet0, phi0)
    
    strings = (rs, ws, mr, mi, ars, drs, t0s, p0s, als, bes)
    angles = (thet0, thet, phi0, phi, alpha, beta)
    floats = (radius, wl, m, axis_ratio, angles, res)
    
    return strings, floats


def S2Z(S11, S12, S21, S22):
    """ Transformation of the amplitude matrix S into the scattering matrix Z
    as done within the notorious Mishchenko T-Matrix code
    https://www.giss.nasa.gov/staff/mmishchenko/t_matrix.html
    
    Parameters
    ----------
    S11, S12, S21, S22 : complex
        The 4 complex values of the amplitude matrix as defined in Mishchenko
        books and codes

    Returns
    -------
    Z : numpy 4x4 float
        The 4x4 real valued scattering matrix as defined in Mishchenko
        books and codes
    """

    S11c = S11.conjugate()
    S12c = S12.conjugate()
    S21c = S21.conjugate()
    S22c = S22.conjugate()
    
    Z11 = 0.5*( S11*S11c + S12*S12c + S21*S21c + S22*S22c).real
    Z12 = 0.5*( S11*S11c - S12*S12c + S21*S21c - S22*S22c).real
    Z13 = -(S11*S12c + S22*S21c).real
    Z14 = -(S11*S12c - S22*S21c).imag
    
    Z21 = 0.5*( S11*S11c + S12*S12c - S21*S21c - S22*S22c).real
    Z22 = 0.5*( S11*S11c - S12*S12c - S21*S21c + S22*S22c).real
    Z23 = -(S11*S12c - S22*S21c).real
    Z24 = -(S11*S12c + S22*S21c).imag
    
    Z31 = -(S11*S21c + S22*S12c).real
    Z32 = -(S11*S21c - S22*S12c).real
    Z33 = (S11*S22c + S12*S21c).real
    Z34 = (S11*S22c + S21*S12c).imag
    
    Z41 = -(S21*S11c + S22*S12c).imag
    Z42 = -(S21*S11c - S22*S12c).imag
    Z43 = (S22*S11c - S12*S21c).imag
    Z44 = (S22*S11c - S12*S21c).real
    
    Z = np.array([[Z11, Z12, Z13, Z14],
                  [Z21, Z22, Z23, Z24],
                  [Z31, Z32, Z33, Z34],
                  [Z41, Z42, Z43, Z44]])
    return Z


def lab2partRF2(th0L, th1L, ph0L, ph1L, alpha, beta, gamma=0.0):
    """ This function computes the transformation from laboratory to particle
    reference frame for the incident and scattering directions as detailed
    in Mishchenko, "Calculation of the amplitude matrix for a nonspherical 
    particle in a fixed orientation," Appl. Opt. 39, 1026-1031 (2000) 

    Parameters
    ----------
    th0L : float - degrees
        Polar angle of incident direction in Laboratory reference frame.
    th1L : float - degrees
        Polar angle of scattering direction in Laboratory reference frame.
    ph0L : float - degrees
        Azimuth angle of incident direction in Laboratory reference frame.
    ph1L : float - degrees
        Azimuth angle of scattering direction in Laboratory reference frame.
    alpha : float - degrees
        Alpha Euler zyz angle.
    beta : float - degrees
        Beta Euler zyz angle.
    gamma : float - degrees, optional
        Gamma Euler zyz angle. It is pointless for rotationally symmetric 
        particles The default is 0.0.

    Returns
    -------
    th0P : float - degrees
        Polar angle of the incident direction in Particle reference frame.
    th1P : float - degrees
        Polar angle of the scattering direction in Particle reference frame.
    ph0P : float - degrees
        Polar angle of the incident direction in Particle reference frame.
    ph1P : float - degrees
        Polar angle of the scattering direction in Particle reference frame.
    Rho : numpy 2x2 - float
        The Rho and Rho⁻1 2x2 real value transformation matrices that transform
        the amplitude matrix in the particle reference frame to the one in the
        laboratory reference frame. Definition of these matrices is detailed in
        is described in Mishchenko, "alculation of the amplitude matrix for a
        nonspherical particle in a fixed orientation," Appl. Opt. 39, 1026-1031
        (2000).
    """

    alp = np.deg2rad(alpha)
    bet = np.deg2rad(beta)
    gam = np.deg2rad(gamma)
    
    th0Lr = np.deg2rad(th0L)
    th1Lr = np.deg2rad(th1L)
    ph0Lr = np.deg2rad(ph0L)
    ph1Lr = np.deg2rad(ph1L)
       
    EPS = 1.0e-7  
    
    if (th0Lr < 0.5*np.pi):
        th0Lr += EPS
    if (th0Lr > 0.5*np.pi):
        th0Lr -= EPS
    if (th1Lr < 0.5*np.pi):
        th1Lr += EPS
    if (th1Lr > 0.5*np.pi):
        th1Lr -= EPS
    
    if (ph0Lr < np.pi):
        ph0Lr += EPS
    if (ph0Lr > np.pi):
        ph0Lr -= EPS
        
    if (ph1Lr < np.pi):
        ph1Lr += EPS
    if (ph1Lr > np.pi):
        ph1Lr -= EPS
    
    if ((bet <= 0.5*np.pi) and (2.0*np.pi-bet <= EPS)):
        bet -= EPS
    if ((bet > 0.5*np.pi) and (bet-2.0*np.pi <= EPS)):
        bet += EPS
    
    cosa = np.cos(alp)
    sina = np.sin(alp)

    cosb = np.cos(bet)
    sinb = np.sin(bet)
    
    cosg = np.cos(gam)
    sing = np.sin(gam)
    
    # Beta matrix (eq. 17) can already be calculated
    #Beta = _betaMatrix(alp, bet, gam)
    Beta = np.array([[cosa*cosb*cosg-sina*sing,  sina*cosb*cosg+cosa*sing, -sinb*cosg],
                     [-cosa*cosb*sing-sina*cosg, -sina*cosb*sing+cosa*cosg, sinb*sing],
                     [cosa*sinb, sina*sinb,  cosb]])
    invBeta = Beta.T

    cth0 = np.cos(th0Lr)
    sth0 = np.sin(th0Lr)

    cth1 = np.cos(th1Lr)
    sth1 = np.sin(th1Lr)
    
    # Alfa matrices (eq. 15) for the Lab. ref. frame can also be defined now
    cph0 = np.cos(ph0Lr)
    sph0 = np.sin(ph0Lr)

    Alpha0 = np.array([[cth0*cph0, -sph0],
                       [cth0*sph0,  cph0],
                       [-sth0, 0.0]])
    
    cph1 = np.cos(ph1Lr)
    sph1 = np.sin(ph1Lr)
    
    Alpha1 = np.array([[cth1*cph1, -sph1],
                       [cth1*sph1,  cph1],
                       [-sth1, 0.0]])

    cp0 = np.cos(ph0Lr - alp)
    sp0 = np.sin(ph0Lr - alp)

    cth0P = cth0*cosb + sth0*sinb*cp0  # Eq. 9
    theta0P = np.arccos(cth0P)
    cpp0 = cosb*cosg*sth0*cp0 + sing*sth0*sp0 - sinb*cosg*cth0 # (Eq. 10)*sth0P
    spp0 = -cosb*sing*sth0*cp0 + cosg*sth0*sp0 + sinb*sing*cth0# (Eq. 11)*sth0P
    
    phi0P = np.arctan2(spp0, cpp0)
    if (phi0P<0.0):
        phi0P+=2.0*np.pi    

    cp1 = np.cos(ph1Lr - alp)
    sp1 = np.sin(ph1Lr - alp)
    cth1P = cth1*cosb + sth1*sinb*cp1  # Eq. 9
    theta1P = np.arccos(cth1P)

    cpp1 = cosb*cosg*sth1*cp1 + sing*sth1*sp1 - sinb*cosg*cth1 # (Eq. 10)*sth0P
    spp1 = -cosb*sing*sth1*cp1 + cosg*sth1*sp1 + sinb*sing*cth1# (Eq. 11)*sth0P
    
    phi1P = np.arctan2(spp1, cpp1)
    if (phi1P<0.0):
        phi1P+=2.0*np.pi
        
    # inverse Alfa matrices (eq. 16) for the Part. ref. frame
    # I use the same definition as the Alfa matrix (eq. 15) with transposition
    # for easier reading
    cpp0 = np.cos(phi0P)
    spp0 = np.sin(phi0P)
    
    invAlpha0 = np.array([[cth0P*cpp0, -spp0],
                          [cth0P*spp0,  cpp0],
                          [-np.sin(theta0P), 0.0]]).T
    
    cpp1 = np.cos(phi1P)
    spp1 = np.sin(phi1P)
    
    invAlpha1 = np.array([[cth1P*cpp1, -spp1],
                          [cth1P*spp1,  cpp1],
                          [-np.sin(theta1P), 0.0]]).T
    
    th0P = np.rad2deg(theta0P)
    ph0P = np.rad2deg(phi0P)
    th1P = np.rad2deg(theta1P)
    ph1P = np.rad2deg(phi1P)
    
    Rho = invAlpha0.dot(Beta).dot(Alpha0)
    #invRho = np.linalg.inv(invAlpha1.dot(Beta).dot(Alpha1))
    invRho = (Alpha1.T).dot(invBeta).dot(invAlpha1.T)
    
    return th0P, th1P, ph0P, ph1P, Rho, invRho


def lab2partRF(th0L, th1L, ph0L, ph1L, alpha, beta, gamma=0.0):
    """ This function computes the transformation from laboratory to particle
    reference frame for the incident and scattering directions as detailed
    in Mishchenko, "Calculation of the amplitude matrix for a nonspherical 
    particle in a fixed orientation," Appl. Opt. 39, 1026-1031 (2000) 

    Parameters
    ----------
    th0L : float - degrees
        Polar angle of incident direction in Laboratory reference frame.
    th1L : float - degrees
        Polar angle of scattering direction in Laboratory reference frame.
    ph0L : float - degrees
        Azimuth angle of incident direction in Laboratory reference frame.
    ph1L : float - degrees
        Azimuth angle of scattering direction in Laboratory reference frame.
    alpha : float - degrees
        Alpha Euler zyz angle.
    beta : float - degrees
        Beta Euler zyz angle.
    gamma : float - degrees, optional
        Gamma Euler zyz angle. It is pointless for rotationally symmetric 
        particles The default is 0.0.

    Returns
    -------
    th0P : float - degrees
        Polar angle of the incident direction in Particle reference frame.
    th1P : float - degrees
        Polar angle of the scattering direction in Particle reference frame.
    ph0P : float - degrees
        Polar angle of the incident direction in Particle reference frame.
    ph1P : float - degrees
        Polar angle of the scattering direction in Particle reference frame.
    Rho : numpy 2x2 - float
        The Rho and Rho⁻1 2x2 real value transformation matrices that transform
        the amplitude matrix in the particle reference frame to the one in the
        laboratory reference frame. Definition of these matrices is detailed in
        is described in Mishchenko, "alculation of the amplitude matrix for a
        nonspherical particle in a fixed orientation," Appl. Opt. 39, 1026-1031
        (2000).
    """

    alp = np.deg2rad(alpha)
    bet = np.deg2rad(beta)
    gam = np.deg2rad(gamma)
    
    th0Lr = np.deg2rad(th0L)
    th1Lr = np.deg2rad(th1L)
    ph0Lr = np.deg2rad(ph0L)
    ph1Lr = np.deg2rad(ph1L)
    
    #print("ph0Lr-alp {} ph1Lr-alp {}".format(ph0Lr-alp, ph1Lr-alp))
    #cp0 = np.cos(ph0Lr - alp)
    #sp0 = np.sin(ph0Lr - alp)
    #cp1 = np.cos(ph1Lr - alp)
    #sp1 = np.sin(ph1Lr - alp)
    
    EPS = 1.0e-7  
    
    if (th0Lr < 0.5*np.pi):
        th0Lr += EPS
    if (th0Lr > 0.5*np.pi):
        th0Lr -= EPS
    if (th1Lr < 0.5*np.pi):
        th1Lr += EPS
    if (th1Lr > 0.5*np.pi):
        th1Lr -= EPS
    
    if (ph0Lr < np.pi):
        ph0Lr += EPS
    if (ph0Lr > np.pi):
        ph0Lr -= EPS
        
    if (ph1Lr < np.pi):
        ph1Lr += EPS
    if (ph1Lr > np.pi):
        ph1Lr -= EPS
    
    if ((bet <= 0.5*np.pi) and (2.0*np.pi-bet <= EPS)):
        bet -= EPS
    if ((bet > 0.5*np.pi) and (bet-2.0*np.pi <= EPS)):
        bet += EPS
        
    #print("ph0Lr-alp {} ph1Lr-alp {}".format(ph0Lr-alp, ph1Lr-alp))
    
    cosa = np.cos(alp)
    sina = np.sin(alp)

    cosb = np.cos(bet)
    sinb = np.sin(bet)
    
    cosg = np.cos(gam)
    sing = np.sin(gam)
    
    # Beta matrix (eq. 17) can already be calculated
    #Beta = _betaMatrix(alp, bet, gam)
    Beta = np.array([[cosa*cosb*cosg-sina*sing,  sina*cosb*cosg+cosa*sing, -sinb*cosg],
                     [-cosa*cosb*sing-sina*cosg, -sina*cosb*sing+cosa*cosg, sinb*sing],
                     [cosa*sinb, sina*sinb,  cosb]])
    #print("\n\n Beta matrix \n {}".format(Beta))

    cth0 = np.cos(th0Lr)
    sth0 = np.sin(th0Lr)

    cth1 = np.cos(th1Lr)
    sth1 = np.sin(th1Lr)
    
    # Alfa matrices (eq. 15) for the Lab. ref. frame can also be defined now
    cph0 = np.cos(ph0Lr)
    sph0 = np.sin(ph0Lr)
    #Alpha0 = _alfaMatrix(th0Lr, ph0Lr)
    Alpha0 = np.array([[cth0*cph0, -sph0],
                       [cth0*sph0,  cph0],
                       [-sth0, 0.0]])
    #print("\n\n Alpha0 matrix  \n  {}".format(Alpha0))
    
    cph1 = np.cos(ph1Lr)
    sph1 = np.sin(ph1Lr)
    #Alpha1 = _alfaMatrix(th1Lr, ph1Lr)
    Alpha1 = np.array([[cth1*cph1, -sph1],
                       [cth1*sph1,  cph1],
                       [-sth1, 0.0]])
    #print("\n\n Alpha1 matrix  \n {}".format(Alpha1))

    cp0 = np.cos(ph0Lr - alp)
    sp0 = np.sin(ph0Lr - alp)
    #print("cp0 {}   sp0 {}".format(cp0, sp0))

    cth0P = cth0*cosb + sth0*sinb*cp0  # Eq. 9
    theta0P = np.arccos(cth0P)
    cpp0 = cosb*cosg*sth0*cp0 + sing*sth0*sp0 - sinb*cosg*cth0 # (Eq. 10)*sth0P
    spp0 = -cosb*sing*sth0*cp0 + cosg*sth0*sp0 + sinb*sing*cth0# (Eq. 11)*sth0P
    
    #print("cpp0 {}   spp0 {}".format(cpp0, spp0))
    
    """
    phi0P = np.arctan(spp0/cpp0)
    
    #print("phi0P {}".format(phi0P))
    if ((phi0P>0.0) and (sp0<0.0)):
        phi0P+=np.pi
    if ((phi0P<0.0) and (sp0>0.0)):
        phi0P+=np.pi
    if (phi0P<0.0):
        phi0P+=2.0*np.pi
    """
    #print("phi0P norm {}".format(phi0P))
    phi0P = np.arctan2(spp0, cpp0)
    if (phi0P<0.0):
        phi0P+=2.0*np.pi
    

    cp1 = np.cos(ph1Lr - alp)
    sp1 = np.sin(ph1Lr - alp)
    #print("cp1 {}   sp1 {}".format(cp1, sp1))
    cth1P = cth1*cosb + sth1*sinb*cp1  # Eq. 9
    theta1P = np.arccos(cth1P)

    cpp1 = cosb*cosg*sth1*cp1 + sing*sth1*sp1 - sinb*cosg*cth1 # (Eq. 10)*sth0P
    spp1 = -cosb*sing*sth1*cp1 + cosg*sth1*sp1 + sinb*sing*cth1# (Eq. 11)*sth0P
    
    """
    phi1P = np.arctan(spp1/cpp1);
    
    #print("cpp1 {}   spp1 {}".format(cpp1, spp1))
    #print("phi1P {}".format(phi1P))

    if ((phi1P>0.0) and (sp1<0.0)):
#        print('positive')
        phi1P+=np.pi
    if ((phi1P<0.0) and (sp1>0.0)):
#        print('negative')
        phi1P+=np.pi
    if (phi1P<0.0):
#        print('normalize')
        phi1P+=2.0*np.pi
    #print("phi1P norm {}".format(phi1P))
    """
    phi1P = np.arctan2(spp1, cpp1)
    if (phi1P<0.0):
        phi1P+=2.0*np.pi
        
    # inverse Alfa matrices (eq. 16) for the Part. ref. frame
    # I use the same definition as the Alfa matrix (eq. 15) with transposition
    # for easier reading
    cpp0 = np.cos(phi0P)
    spp0 = np.sin(phi0P)
    #print("\n cpp0 {} spp0 {} \n".format(cpp0, spp0))
    #invAlpha0 = _alfaMatrix(theta0P, phi0P).T
    invAlpha0 = np.array([[cth0P*cpp0, -spp0],
                          [cth0P*spp0,  cpp0],
                          [-np.sin(theta0P), 0.0]]).T
    #print("\n\n invAlpha0 matrix  \n {}".format(invAlpha0))
    
    cpp1 = np.cos(phi1P)
    spp1 = np.sin(phi1P)
    #print("\n cpp1 {} spp1 {} \n".format(cpp1, spp1))
    #invAlpha1 = _alfaMatrix(theta1P, phi1P).T
    invAlpha1 = np.array([[cth1P*cpp1, -spp1],
                          [cth1P*spp1,  cpp1],
                          [-np.sin(theta1P), 0.0]]).T
    #print("\n\n invAlpha1 matrix  \n {}".format(invAlpha1))
    
    th0P = np.rad2deg(theta0P)
    ph0P = np.rad2deg(phi0P)
    th1P = np.rad2deg(theta1P)
    ph1P = np.rad2deg(phi1P)
    
    Rho = invAlpha0.dot(Beta).dot(Alpha0)
    invRho = np.linalg.inv(invAlpha1.dot(Beta).dot(Alpha1))
    
    return th0P, th1P, ph0P, ph1P, Rho, invRho


def lab2partRF_old(th0L, th1L, ph0L, ph1L, alpha, beta, gamma=0.0):
    """ This function computes the transformation from laboratory to particle
    reference frame for the incident and scattering directions as detailed
    in Mishchenko, "Calculation of the amplitude matrix for a nonspherical 
    particle in a fixed orientation," Appl. Opt. 39, 1026-1031 (2000) 

    Parameters
    ----------
    th0L : float - degrees
        Polar angle of incident direction in Laboratory reference frame.
    th1L : float - degrees
        Polar angle of scattering direction in Laboratory reference frame.
    ph0L : float - degrees
        Azimuth angle of incident direction in Laboratory reference frame.
    ph1L : float - degrees
        Azimuth angle of scattering direction in Laboratory reference frame.
    alpha : float - degrees
        Alpha Euler zyz angle.
    beta : float - degrees
        Beta Euler zyz angle.
    gamma : float - degrees, optional
        Gamma Euler zyz angle. It is pointless for rotationally symmetric 
        particles The default is 0.0.

    Returns
    -------
    th0P : float - degrees
        Polar angle of the incident direction in Particle reference frame.
    th1P : float - degrees
        Polar angle of the scattering direction in Particle reference frame.
    ph0P : float - degrees
        Polar angle of the incident direction in Particle reference frame.
    ph1P : float - degrees
        Polar angle of the scattering direction in Particle reference frame.
    Rho : numpy 2x2 - float
        The Rho and Rho⁻1 2x2 real value transformation matrices that transform
        the amplitude matrix in the particle reference frame to the one in the
        laboratory reference frame. Definition of these matrices is detailed in
        is described in Mishchenko, "alculation of the amplitude matrix for a
        nonspherical particle in a fixed orientation," Appl. Opt. 39, 1026-1031
        (2000).
    """

    alp = np.deg2rad(alpha)
    bet = np.deg2rad(beta)
    gam = np.deg2rad(gamma)
    
    th0Lr = np.deg2rad(th0L)
    th1Lr = np.deg2rad(th1L)
    ph0Lr = np.deg2rad(ph0L)
    ph1Lr = np.deg2rad(ph1L)
    
    EPS = 1.0e-7
    if (th0Lr < 2.0*np.pi):
        th0Lr += EPS
    if (th0Lr > 2.0*np.pi):
        th0Lr -= EPS
    if (th1Lr < 2.0*np.pi):
        th1Lr += EPS
    if (th1Lr > 2.0*np.pi):
        th1Lr -= EPS
    
    if (ph0Lr < np.pi):
        ph0Lr += EPS
    if (ph0Lr > np.pi):
        ph0Lr -= EPS
    if (ph1Lr < np.pi):
        ph1Lr += EPS
    if (ph1Lr > np.pi):
        ph1Lr -= EPS
    
    if ((bet <= 2.0*np.pi) and (2.0*np.pi-bet <= EPS)):
        bet -= EPS
    if ((bet > 2.0*np.pi) and (bet-2.0*np.pi <= EPS)):
        bet += EPS
      
    cosa = np.cos(alp)
    sina = np.sin(alp)

    cosb = np.cos(bet)
    sinb = np.sin(bet)
    
    cosg = np.cos(gam)
    sing = np.sin(gam)
    
    # Beta matrix (eq. 17) can already be calculated
    Beta = _betaMatrix(alp, bet, gam)
    """
    Beta = np.array([[cosa*cosb*cosg-sina*sing,  sina*cosb*cosg+cosa*sing,
                      -sinb*cosg],
                     [-cosa*cosb*sing-sina*cosg, -sina*cosb*sing+cosa*cosg,
                      sinb*sing],
                     [cosa*sinb, sina*sinb, cosb]])
    """
    cth0 = np.cos(th0Lr)
    sth0 = np.sin(th0Lr)

    cth1 = np.cos(th1Lr)
    sth1 = np.sin(th1Lr)
    
    # Alfa matrices (eq. 15) for the Lab. ref. frame can also be defined now
    cph0 = np.cos(ph0Lr)
    sph0 = np.sin(ph0Lr)
    Alpha0 = _alfaMatrix(th0Lr, ph0Lr)
    """
    Alpha0 = np.array([[cth0*cph0, -sph0],
                       [cth0*sph0,  cph0],
                       [-sth0, 0.0]])
    """
    #print("Alpha0")
    #print(Alpha0)
    
    cph1 = np.cos(ph1Lr)
    sph1 = np.sin(ph1Lr)
    Alpha1 = _alfaMatrix(th1Lr, ph1Lr)
    """
    Alpha1 = np.array([[cth1*cph1, -sph1],
                       [cth1*sph1,  cph1],
                       [-sth1, 0.0]])
    """
    cp0 = np.cos(ph0Lr - alp)
    sp0 = np.sin(ph0Lr - alp)
    #print("Alpha1")
    #print(Alpha1)

    cth0P = cth0*cosb + sth0*sinb*cp0  # Eq. 9
    theta0P = np.arccos(cth0P)
    cpp0 = cosb*cosg*sth0*cp0 + sing*sth0*sp0 - sinb*cosg*cth0      # (Eq. 10)*sth0P
    spp0 = -cosb*sing*sth0*cp0 + cosg*sth0*sp0 + sinb*sing*cth0# (Eq. 11)*sth0P
    phi0P = np.arctan(spp0/cpp0)
    if ((phi0P>0.0) and (sp0<0.0)):
        phi0P+=np.pi
    if ((phi0P<0.0) and (sp0>0.0)):
        phi0P+=np.pi
    if (phi0P<0.0):
        phi0P+=2.0*np.pi

    cp1 = np.cos(ph1Lr - alp)
    sp1 = np.sin(ph1Lr - alp)
    cth1P = cth1*cosb + sth1*sinb*cp1  # Eq. 9
    theta1P = np.arccos(cth1P)

    cpp1 = cosb*cosg*sth1*cp1 + sing*sth1*sp1 - sinb*cosg*cth1      # (Eq. 10)*sth0P
    spp1 = -cosb*sing*sth1*cp1 + cosg*sth1*sp1 + sinb*sing*cth1# (Eq. 11)*sth0P
    phi1P = np.arctan(spp1/cpp1);
    if ((phi1P>0.0) and (sp1<0.0)):
#        print('positive')
        phi1P+=np.pi
    if ((phi1P<0.0) and (sp1>0.0)):
#        print('negative')
        phi1P+=np.pi
    if (phi1P<0.0):
#        print('normalize')
        phi1P+=2.0*np.pi
    
    # inverse Alfa matrices (eq. 16) for the Part. ref. frame
    # I use the same definition as the Alfa matrix (eq. 15) with transposition
    # for easier reading
    cpp0 = np.cos(phi0P)
    spp0 = np.sin(phi0P)
    #invAlpha0 = _alfaMatrix(theta0P, phi0P).T
    
    invAlpha0 = np.array([[cth0P*cpp0, -spp0],
                          [cth0P*spp0,  cpp0],
                          [-np.sin(theta0P), 0.0]]).T
    
    #print('invAlpha0')
    #print(invAlpha0)
    
    cpp1 = np.cos(phi1P)
    spp1 = np.sin(phi1P)
    invAlpha1 = _alfaMatrix(theta1P, phi1P).T
    """
    invAlpha1 = np.array([[cth1P*cpp1, -spp1],
                          [cth1P*spp1,  cpp1],
                          [-np.sin(theta1P), 0.0]]).T
    """
    #print('invAlpha1')
    #print(invAlpha1)
    
    th0P = np.rad2deg(theta0P)
    ph0P = np.rad2deg(phi0P)
    th1P = np.rad2deg(theta1P)
    ph1P = np.rad2deg(phi1P)
    
    Rho = invAlpha0.dot(Beta).dot(Alpha0)
    invRho = np.linalg.inv(invAlpha1.dot(Beta).dot(Alpha1))
    
    return th0P, th1P, ph0P, ph1P, Rho, invRho


def alfaMatrix(th, ph):
    """ Public version of _alfaMatrix, it works with degrees
    Computes the transformation Alfa matrix as in eq (15) of Mishchenko, 
    "Calculation of the amplitude matrix for a nonspherical particle in a fixed
    orientation," Appl. Opt. 39, 1026-1031 (2000).
    The inverse of this matrix (equation 16) can be obtained by transposition
    Since this is intended for internal use of the module, input angles are
    already in radians

    Parameters
    ----------
    th : float - degrees
        Polar angle of the propagation direction of the electromagnetic wave.
    ph : float - degrees
        Azimuth angle of the propagation direction of the electromagnetic wave.

    Returns
    -------
    Alfa : 3x2 matrix float
        Alfa transformation matrix
    """
    
    Alfa = _alfaMatrix(np.deg2rad(th), np.deg2rad(ph))
    
    return Alfa


def _alfaMatrix(th, ph):
    """ Computes the transformation Alfa matrix as in eq (15) of Mishchenko, 
    "Calculation of the amplitude matrix for a nonspherical particle in a fixed
    orientation," Appl. Opt. 39, 1026-1031 (2000).
    The inverse of this matrix (equation 16) can be obtained by transposition
    Since this is intended for internal use of the module, input angles are
    already in radians

    Parameters
    ----------
    th : float - radians
        Polar angle of the propagation direction of the electromagnetic wave.
    ph : float - radians
        Azimuth angle of the propagation direction of the electromagnetic wave.

    Returns
    -------
    Alfa : 3x2 matrix float
        Alfa transformation matrix
    """
    
    ct = np.cos(th)
    st = np.sin(th)
    cp = np.cos(ph)
    sp = np.sin(ph)
    Alfa = np.array([[ct*cp, -sp],
                     [ct*sp, cp],
                     [-st, 0.0]])
    
    return Alfa


def betaMatrix(alpha, beta, gamma):
    """ Public version of _betaMatrix, it works with degrees
    Computes the rotation Beta matrix as in eq (17) of Mishchenko, 
    "Calculation of the amplitude matrix for a nonspherical particle in a fixed
    orientation," Appl. Opt. 39, 1026-1031 (2000).
    Since this is intended for internal use of the module, input angles are
    already in radians
    

    Parameters
    ----------
    alpha : float - degrees
        Alpha Euler zyz angle.
    beta : float - degrees
        Beta Euler zyz angle.
    gamma : float - degress
        Gamma Euler zyz angle.

    Returns
    -------
    Beta : 3x3 matrix float
        Beta rotation matrix.
    """
    Beta = _betaMatrix(np.deg2rad(alpha), np.deg2rad(beta), np.deg2rad(gamma))
    
    return Beta

def _betaMatrix(alpha, beta, gamma):
    """ Computes the rotation Beta matrix as in eq (17) of Mishchenko, 
    "Calculation of the amplitude matrix for a nonspherical particle in a fixed
    orientation," Appl. Opt. 39, 1026-1031 (2000).
    Since this is intended for internal use of the module, input angles are
    already in radians
    

    Parameters
    ----------
    alpha : float - radians
        Alpha Euler zyz angle.
    beta : float - radians
        Beta Euler zyz angle.
    gamma : float - radians
        Gamma Euler zyz angle.

    Returns
    -------
    Beta : 3x3 matrix float
        Beta rotation matrix.
    """
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    cb = np.cos(beta)
    sb = np.sin(beta)
    
    cg = np.cos(gamma)
    sg = np.sin(gamma)
    
    Beta = np.array([[ca*cb*cg-sa*sg,  sa*cb*cg+ca*sg, -sb*cg],
                     [-ca*cb*sg-sa*cg, -sa*cb*sg+ca*cg, sb*sg],
                     [ca*sb, sa*sb,  cb]])
    
    return Beta


def Rampl(eta):
    """ Public version of _Rampl, it works with degrees
    
    Parameters
    ----------
    eta : float - degrees
        Angle between the V (or H) polarization planes that are defined
        with respect to the z-axis in either the Laboratory or the Particle
        reference frame.

    Returns
    -------
    R : 2x2 matrix float
        Simple 2D rotation matrix.
    """
    
    R = _Rampl(np.deg2rad(eta))
    
    return R


def _Rampl(eta):
    """
    Basic rotation matrix that takes into account the rotation of the
    polarization plane when passing from Laboratory to Particle reference frame
    The angle eta between the 2 polarization axis is computed from th0, th, 
    ph0, ph, alpha, beta, gamma. See Mishchenko (2000) book for details.

    Parameters
    ----------
    eta : float - radians
        Angle between the V (or H) polarization planes that are defined
        with respect to the z-axis in either the Laboratory or the Particle
        reference frame.

    Returns
    -------
    R : 2x2 matrix float
        Simple 2D rotation matrix.
    """
    ce = np.cos(eta)
    se = np.sin(eta)
    R = np.array([[ce, se],
                  [-se, ce]])
    return R


def _Lscatt(eta):
    """
    Rotation matrix that transforms the Stokes parameters derived for a certain
    orientation of the polarization vector to those for a rotated polarization
    plane. The eta angle is the angle between the 2 polarization axis. 
    See Mishchenko (2000) book for details.

    Parameters
    ----------
    eta : float - radians
        Angle between the V (or H) polarization planes that are defined
        with respect to the z-axis in either the Laboratory or the Particle
        reference frame.

    Returns
    -------
    L : 4x4 matrix float
        The 4x4 rotation L matrix for the Stokes parameters

    """
    c2e = np.cos(2.0*eta)
    s2e = np.sin(2.0*eta)
    L = np.array([[0.0, 0.0, 0.0, 0.0],
                  [0.0, c2e,-s2e, 0.0],
                  [0.0, s2e, c2e, 0.0],
                  [0.0, 0.0, 0.0, 0.0]])
    return L