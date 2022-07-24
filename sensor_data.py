import numpy as np

NIKON_WAVELENGTHS = np.linspace(400., 720., 33)
NIKON_RESPONSE = np.load('data/nikon5100.npy')

SENSOR_DATABASE_WAVELENGTHS = np.linspace(400., 720., 33)
SENSOR_DATABASE_RESPONSE_R = np.load('data/sensor_spectral_responses_r.npy')
SENSOR_DATABASE_RESPONSE_G = np.load('data/sensor_spectral_responses_g.npy')
SENSOR_DATABASE_RESPONSE_B = np.load('data/sensor_spectral_responses_b.npy')

TEST_RESPONSE = np.ones_like(NIKON_WAVELENGTHS)
