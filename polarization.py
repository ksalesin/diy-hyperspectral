"""
Functions implementing Jones and Mueller calculus for polarization.

Written by Kate, 2020
"""

import math
import cmath
import numpy as np

def phase_shift(wavelength, thickness, dior):
    """ 
    Calculate phase shift.

    * wavelength:    wavelength of light, nm
    * thickness:     thickness of tape, nm
    * dior:          birefringence of tape, unitless
    """
    return (2 * math.pi * dior * thickness) / wavelength

def intensity_jones(j):
    """ Calculate intensity of Jones vector. """
    rex = j[0].real
    imx = j[0].imag
    rey = j[1].real
    imy = j[1].imag
    return rex * rex + imx * imx + rey * rey + imy * imy

def rotation_jones(theta):
    """ Create Jones rotation matrix at angle theta. """
    cost = math.cos(theta)
    sint = math.sin(theta)
    return np.array([[cost, sint],
                    [-sint, cost]])

def linear_polarizer_jones(theta):
    """ Create Jones linear polarizer at angle theta. """
    cost = math.cos(theta)
    sint = math.sin(theta)
    sint_cost = cost * sint

    return np.array([[cost * cost, sint_cost],
                     [sint_cost, sint * sint]])

def waveplate_jones(gamma, alpha):
    """ Create Jones waveplate with phase shift gamma at angle alpha. """
    cos_g2 = math.cos(gamma / 2)
    sin_g2 = math.sin(gamma / 2)
    cos_2t = math.cos(2 * alpha)
    sin_2t = math.sin(2 * alpha)

    return np.array([[cos_g2 + 1j * sin_g2 * cos_2t, 1j * sin_g2 * sin_2t],
                     [1j * sin_g2 * sin_2t, cos_g2 - 1j * sin_g2 * cos_2t]])

def waveplate_mueller(gamma, alpha):
    cos2a = np.cos(2 * alpha)
    sin2a = np.sin(2 * alpha)
    cosg = np.cos(gamma)
    sing = np.sin(gamma)
    cosg_ = 1 - cosg

    w11 = cos2a * cos2a + sin2a * sin2a * cosg
    w12 = cos2a * sin2a * cosg_
    w13 = sin2a * sing
    w21 = cos2a * sin2a * cosg_
    w22 = cos2a * cos2a * cosg + sin2a * sin2a
    w23 = -cos2a * sing
    w31 = -sin2a * sing
    w32 = cos2a * sing
    w33 = cosg

    return np.array([[1.,  0.,  0.,  0.],
                     [0., w11, w12, w13],
                     [0., w21, w22, w23],
                     [0., w31, w32, w33]])

def transmission_jones(gamma, alpha, theta):
    """ 
    Evaluate transmission of outgoing light given a 
    polarizer-waveplate-analyzer configuration.
    Uses Jones calculus.

    * gamma: phase shift
    * alpha: angle of waveplate w.r.t. polarizer, rad
    * theta: angle of analyzer w.r.t. polarizer, rad
    """ 

    wp = waveplate_jones(gamma, alpha)
    an = linear_polarizer_jones(theta)

    # Jones vector of incoming light (renormalized after passing through 1st polarizer)
    j_in = np.array([[1.], [0.]])

    # Jones vector of transmitted light
    j_out = an @ wp @ j_in

    return intensity_jones(j_out)

def transmission_mueller(gamma, alpha, theta):
    """ 
    Evaluate transmission of outgoing light given a 
    polarizer-waveplate-analyzer configuration.
    Uses Mueller calculus.

    * gamma: phase shift
    * alpha: angle of waveplate w.r.t. polarizer, rad
    * theta: angle of analyzer w.r.t. polarizer, rad
    """ 
    
    # added 'and len(gamma) > 1' to account for using only 1 wavelength (such as when using a laser as only illuminant)
    
    # Waveplate(s)
    if isinstance(gamma, (list, np.ndarray)) and len(gamma) > 1:
        wp = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])
        
        for (g, a) in zip(gamma, alpha):
            wp = waveplate_mueller(g, a) @ wp
    else:
        wp = waveplate_mueller(gamma, alpha)
    
    cos2t = np.cos(2 * theta)
    sin2t = np.sin(2 * theta)

    # Linear polarizer (analyzer, final layer)
    az = 0.5 * np.array([[1., cos2t, sin2t, 0.],
                         [cos2t, cos2t * cos2t, sin2t * cos2t, 0.],
                         [sin2t, sin2t * cos2t, sin2t * sin2t, 0.],
                         [0., 0., 0., 0.]],
                         dtype=np.float32)

    # Stokes vector of incoming light (renormalized after passing through first linear polarizer)
    s_i = np.array([[1.], 
                    [1.], 
                    [0.], 
                    [0.]], 
                    dtype=np.float32)

    # Stokes vector of transmitted light
    s_o = az @ wp @ s_i

    # Intensity of transmitted light
    return s_o[0]
