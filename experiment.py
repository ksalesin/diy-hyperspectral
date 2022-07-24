"""
    Base class for experiments. A data structure to hold all information relevant to an experiment
    that can be passed around to different functions during optimization.
"""

import rawpy
import numpy as np
import matplotlib.pyplot as plt

import color_utils
from polarization import phase_shift, transmission_mueller

class Waveplate:
    def __init__(self, thickness, birefringence):
        self.thickness = thickness          # nm
        self.birefringence = birefringence  # unitless

    def __repr__(self):
        return "Waveplate(th = %s, bf = %s)" % (self.thickness, self.birefringence)

    def __str__(self):
        return "Waveplate(th = %s, bf = %s)" % (self.thickness, self.birefringence)


class Config:
    def __init__(self, alphas, theta, waveplates):
        self.alphas = alphas
        self.theta = theta
        self.waveplates = waveplates
        self.n_wp = len(self.waveplates)

    def transmission(self, wavelengths):
        if isinstance(wavelengths, (list, np.ndarray)):
            tr = np.zeros_like(wavelengths)
            for w in range(wavelengths.shape[0]):
                gammas = [phase_shift(wavelengths[w], self.waveplates[t].thickness, self.waveplates[t].birefringence) for t in range(self.n_wp)]
                tr[w] = transmission_mueller(gammas, np.deg2rad(self.alphas), np.deg2rad(self.theta))
        
        elif isinstance(wavelengths, float):
            wav = wavelengths
            gammas = [phase_shift(wav, self.waveplates[t].thickness, self.waveplates[t].birefringence) for t in range(self.n_wp)]
            tr = transmission_mueller(gammas, np.deg2rad(self.alphas), np.deg2rad(self.theta))
            tr = tr[0][0]

        else:
            pass

        return tr

    def __repr__(self):
        return "Config(alphas = %s, theta = %s, waveplates = %s)" % (self.alphas, self.theta, self.waveplates)

    def __str__(self):
        return "Config(alphas = %s, theta = %s, waveplates = %s)" % (self.alphas, self.theta, self.waveplates)

class Experiment:
    def __init__(self, wavelengths, sensor_response, filters, measurements, reflectance_basis, illum_basis, sensor_basis = None, mosaic = False):
        self.wavelengths = wavelengths
        self.sensor_response = sensor_response
        self.filters = filters

        self.reflectance_basis = reflectance_basis
        self.n_basis_r = reflectance_basis.shape[0]

        self.illum_basis = illum_basis
        self.n_basis_e = illum_basis.shape[0]

        if sensor_basis is not None:
            self.sensor_basis = sensor_basis
            self.n_basis_s = sensor_basis.shape[1] if sensor_basis.ndim == 3 else sensor_basis.shape[0]

        self.n_wav = wavelengths.shape[0]
        self.n_ftr = filters.shape[0]

        if mosaic:
            ref_shape = measurements[0].data.shape
            n_mosaic = ref_shape[0]
            assert(len(filters) == len(measurements) * n_mosaic), "Number of measurements and number of filters do not match!"

            self.dim = 1
            self.n_pix = 1
            self.measurements = np.zeros((len(measurements) * n_mosaic, 1, 3))

            for i, m in enumerate(measurements):
                assert(m.data.shape == ref_shape), "Measurements must have the same dimensions."

                for j in range(n_mosaic):
                    self.measurements[i * n_mosaic + j, 0, :] = m.data[j, :]  # n_ftr x n_pix x 3

        else:
            ref_shape = measurements[0].data.shape
            self.dim = len(ref_shape[:-1])
            self.measurements = np.zeros((len(measurements),) + ref_shape)

            for i, m in enumerate(measurements):
                assert(m.data.shape == ref_shape), "Measurements must have the same dimensions."

                if self.dim == 1:
                    self.n_pix = ref_shape[0]
                    self.measurements[i, :, :] = m.data             # n_ftr x n_pix x 3
                else:
                    self.n_pix = ref_shape[0] * ref_shape[1]
                    self.measurements[i, :, :, :] = m.data          # n_ftr x w x h x 3

    def pixel_major(self):
        """ Reshape measurements into a numpy array of pixels 
        in the shape w x h x (n_ftr * 3) or n_pix x (n_ftr * 3), 
        where every measurement of one pixel is in the last axis. """
        if self.dim == 1:
            pixels = np.transpose(self.measurements, axes=(1, 0, 2))      # n_pix x n_ftr x 3
        else:
            pixels = np.transpose(self.measurements, axes=(1, 2, 0, 3))   # w x h x n_ftr x 3
        return pixels.reshape(pixels.shape[:-2] + (self.n_ftr * 3,))      # n_pix x (n_ftr * 3) if 1D or w x h x (n_ftr * 3) if 2D

class Measurement:
    def __init__(self, data, filter_id):
        self.data = data
        self.filter_id = filter_id

class FileMeasurement(Measurement):
    def __init__(self, file, filter_id = 0):
        with rawpy.imread(file) as raw:
            img = raw.postprocess(gamma=(1,1), no_auto_bright=True, use_camera_wb=False, use_auto_wb=False) / 255.
            self.img = raw.postprocess(use_camera_wb=True) / 255.

        super().__init__(img, filter_id)

    def show(self):
        plt.imshow(self.data)
        plt.show()

def make_filters(configs, tapes, wavelengths, dior = 0.01):
    """ Generate transmission spectra of all filters 
        given optical element rotation angles and params using Mueller calculus."""

    nt = tapes.shape[0]                  # Number of tape layers
    nw = wavelengths.shape[0]            # Number of wavelengths
    nf = configs.shape[0] // (nt + 1)    # Number of measurements / filters

    # Generate transmission spectra from filter configurations using Mueller calculus
    filters = np.zeros((nf, nw))

    for f in range(nf):
        for w in range(nw):
            alphas = []
            gammas = []

            for t in range(nt):
                alphas.append(np.deg2rad(configs[f * (nt + 1) + t]))
                gammas.append(phase_shift(wavelengths[w], tapes[t], dior))

            theta = np.deg2rad(configs[f * (nt + 1) + nt])

            filters[f, w] = transmission_mueller(gammas, alphas, theta)

    return filters

def measure(spectra_, filters, sensor_response, wavelengths, illuminant = None, noise_std = 0.0):
    """ Measure RGB response of camera to given spectra after filtering."""
    nf = filters.shape[0]
    nw = wavelengths.shape[0]
    
    dim = spectra_.shape
    spectra = spectra_.reshape(-1, nw)

    ns = spectra.shape[0]

    if illuminant is None:
        measurements = filters.repeat(ns, axis = 0) * np.tile(spectra, reps=(nf, 1))
    else:
        measurements = filters.repeat(ns, axis = 0) * np.tile(spectra, reps = (nf, 1)) * illuminant.repeat(ns * nf, axis = 0)

    # Convert to RGB using the camera spectral response curves
    colors = sensor_response @ measurements.transpose()

    if spectra_.ndim == 2:
        colors = colors.transpose().reshape(nf, dim[0], 3)
    else:
        colors = colors.transpose().reshape(nf, dim[0], dim[1], 3)

    # colors = np.clip(colors, 0., 1.)

    # Add noise
    if noise_std != 0:
        noise = np.random.normal(loc = 0.0, scale = noise_std, size = colors.shape)
        colors += noise
        colors = np.clip(colors, 0., None)

    return colors

def measure_single(wavelength, value, filters, sensor_response, sensor_wavelengths, noise_std = 0.0):
    """ Measure RGB response of camera to a single wavelength (e.g. laser illumination)
       after filtering. Assumes filters is a 1D array of length n of filter transmission values 
       for a single wavelength for n filters."""
    nf = filters.shape[0]

    # Find sensor response at single wavelength
    sensor_wav = color_utils.spec2spec(sensor_wavelengths, sensor_response, wavelength)

    filters_tmp = np.repeat(filters, repeats = 3, axis = 1)
    sensor_tmp  = np.tile(sensor_wav, reps=(nf, 1))
    value_tmp   = np.tile(value, reps=(nf, 3))

    # Measure total response of filters, sensor, and illuminant value
    colors = filters_tmp * sensor_tmp * value_tmp
    
    # colors = np.clip(colors, 0., 1.)

    # Add noise
    if noise_std != 0:
        noise = np.random.normal(loc = 0.0, scale = noise_std, size = colors.shape)
        colors += noise
        colors = np.clip(colors, 0., None)

    return colors
