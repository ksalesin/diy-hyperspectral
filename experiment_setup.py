""" Import this file in order to use the default settings 
    for running reconstruction experiments. """

import numpy as np

import color_utils
import sensor_data
import munsell_data
import optimization as opt
from experiment import *

# Set up experiment
wavelengths = sensor_data.NIKON_WAVELENGTHS
wavelengths = wavelengths[np.where((wavelengths > 399.) & (wavelengths < 701.))]

cc_spectra = color_utils.spec2spec(munsell_data.COLOR_CHECKER_WAVELENGTHS, munsell_data.COLOR_CHECKER_SQUARES, wavelengths)
sensor_response = color_utils.spec2spec(sensor_data.NIKON_WAVELENGTHS, sensor_data.NIKON_RESPONSE, wavelengths)

# Create basis functions for reflectance, illumination, and sensor responses
reflectance_spectra = color_utils.spec2spec(munsell_data.MUNSELL_CHIPS_WAVELENGTHS, munsell_data.MUNSELL_CHIPS, wavelengths)
illum_spectra = color_utils.spec2spec(munsell_data.ILLUMINANT_WAVELENGTHS, munsell_data.ILLUMINANTS, wavelengths)
sensor_spectra_r = color_utils.spec2spec(sensor_data.SENSOR_DATABASE_WAVELENGTHS, sensor_data.SENSOR_DATABASE_RESPONSE_R, wavelengths)
sensor_spectra_g = color_utils.spec2spec(sensor_data.SENSOR_DATABASE_WAVELENGTHS, sensor_data.SENSOR_DATABASE_RESPONSE_G, wavelengths)
sensor_spectra_b = color_utils.spec2spec(sensor_data.SENSOR_DATABASE_WAVELENGTHS, sensor_data.SENSOR_DATABASE_RESPONSE_B, wavelengths)

# illum_spectra = np.vstack((illum_spectra, sensor_spectra_r, sensor_spectra_g, sensor_spectra_b))

n_basis_r = 8
B = opt.basis(reflectance_spectra, n_basis_r)

n_basis_e = 12
E = opt.basis(illum_spectra, n_basis_e)

n_basis_s = 12
S_R = opt.basis(sensor_spectra_r, n_basis_s)
S_G = opt.basis(sensor_spectra_g, n_basis_s)
S_B = opt.basis(sensor_spectra_b, n_basis_s)

def white_balance(meas_wb, sensor_known = False, illum_known = False, gt_illum_file = None):
    """ Return the sensor response curves and illuminant coefficients
        to use for reconstruction. These may be unchanged from the input 
        (if the input is assumed to be known) or estimated. """

    # --------------------------------------------------------------------
    # --------- ASSUME SENSOR RESPONSES AND ILLUMINANT ARE KNOWN ---------
    # --------------------------------------------------------------------
    if sensor_known and illum_known:
        # Recover illuminant scale factor

        # Load ground truth illuminant
        gt_illum = np.load(gt_illum_file)[:,1]
        gt_illum = color_utils.spec2spec(munsell_data.SPECTROMETER_WAVELENGTHS, gt_illum, wavelengths)

        exp_wb = Experiment(wavelengths, sensor_response, np.array([0]), meas_wb, B, E, S_R, S_G, S_B, None)

        illuminant_scale = opt.recover_illum_known(exp_wb, cc_spectra, sensor_response, gt_illum)

        # Get best-fit coefs
        illuminant_coefs, _, _, _ = np.linalg.lstsq(E.transpose(), gt_illum * illuminant_scale, rcond = None)

        return sensor_response, illuminant_coefs

    # --------------------------------------------------------------------
    # ----- ASSUME SENSOR RESPONSES ARE KNOWN, ILLUMINANT IS UNKNOWN -----
    # --------------------------------------------------------------------
    elif sensor_known and not illum_known:
        exp_wb = Experiment(wavelengths, sensor_response, np.array([0]), meas_wb, B, E, None)
        illuminant_coefs = opt.recover_illum(exp_wb, cc_spectra, sensor_response, alpha = 1.)

        return sensor_response, illuminant_coefs

    # --------------------------------------------------------------------
    # ----- ASSUME SENSOR RESPONSES ARE UNKNOWN, ILLUMINANT IS KNOWN -----
    # --------------------------------------------------------------------
    elif not sensor_known and illum_known:
        # Load ground truth illuminant
        gt_illum = np.load(gt_illum_file)[:,1]
        gt_illum = color_utils.spec2spec(munsell_data.SPECTROMETER_WAVELENGTHS, gt_illum, wavelengths)

        # Initial sensor guess (coefs of basis functions)
        c0_r, _, _, _ = np.linalg.lstsq(S_R.transpose(), sensor_response[0,:], rcond = None)
        c0_g, _, _, _ = np.linalg.lstsq(S_G.transpose(), sensor_response[1,:], rcond = None)
        c0_b, _, _, _ = np.linalg.lstsq(S_B.transpose(), sensor_response[2,:], rcond = None)

        # c0_r = opt.fit_basis_nonzero(sensor_response[0,:], S_R.transpose(), alpha = 0.1)
        # c0_g = opt.fit_basis_nonzero(sensor_response[1,:], S_G.transpose(), alpha = 0.1)
        # c0_b = opt.fit_basis_nonzero(sensor_response[2,:], S_B.transpose(), alpha = 0.1)

        exp_wb = Experiment(wavelengths, np.array([0]), np.array([0]), meas_wb, B, E, np.array([S_R, S_G, S_B]))
        c, illuminant_scale = opt.recover_sensor(exp_wb, cc_spectra, gt_illum, c0 = np.column_stack((c0_r, c0_g, c0_b)), alpha = 0.1)
        print(illuminant_scale)
        
        s_r = S_R.transpose() @ c[:,0]
        s_g = S_G.transpose() @ c[:,1]
        s_b = S_B.transpose() @ c[:,2]

        sensor_response_wb = np.row_stack((s_r, s_g, s_b))

        return sensor_response_wb, gt_illum * illuminant_scale

    # --------------------------------------------------------------------
    # -------- ASSUME SENSOR RESPONSES AND ILLUMINANT ARE UNKNOWN --------
    # --------------------------------------------------------------------
    elif not sensor_known and not illum_known:
        exp_wb = Experiment(wavelengths, np.array([0]), np.array([0]), meas_wb, B, E, np.array([S_R, S_G, S_B]))

        # Initial sensor guess (coefs of basis functions)
        c0_r, _, _, _ = np.linalg.lstsq(S_R.transpose(), sensor_response[0,:], rcond = None)
        c0_g, _, _, _ = np.linalg.lstsq(S_G.transpose(), sensor_response[1,:], rcond = None)
        c0_b, _, _, _ = np.linalg.lstsq(S_B.transpose(), sensor_response[2,:], rcond = None)

        # c0_r = opt.fit_basis_nonzero(sensor_response[0,:], S_R.transpose(), alpha = 0.1)
        # c0_g = opt.fit_basis_nonzero(sensor_response[1,:], S_G.transpose(), alpha = 0.1)
        # c0_b = opt.fit_basis_nonzero(sensor_response[2,:], S_B.transpose(), alpha = 0.1)

        # Initial illuminant guess (coefs of basis functions)
        # e0, _, _, _ = np.linalg.lstsq(E.transpose(), illum_spectra[2,:], rcond = None)
        e0 = np.zeros(n_basis_e)

        c, illuminant_coefs = opt.recover_sensor_illum(exp_wb, cc_spectra, c0 = np.column_stack((c0_r, c0_g, c0_b)), e0 = e0, alpha = .1, beta = 1.)

        s_r = S_R.transpose() @ c[:,0]
        s_g = S_G.transpose() @ c[:,1]
        s_b = S_B.transpose() @ c[:,2]

        sensor_response_wb = np.row_stack((s_r, s_g, s_b))

        return sensor_response_wb, illuminant_coefs