import math
from colormath.color_objects import SpectralColor, sRGBColor, HSVColor, LabColor
from colormath.color_conversions import convert_color
from scipy.interpolate import interp1d as interp1d
import numpy as np

COLORMATH_WAVELENGTHS = np.array([
    340,350,360,370,380,390,400,410,420,430,440,450,460,470,480,490,500,510, \
    520,530,540,550,560,570,580,590,600,610,620,630,640,650,660,670,680,690, \
    700,710,720,730,740,750,760,770,780,790,800,810,820,830
], dtype=np.float)

def planckSpectrumModel(wav, T):
    h = 6.626e-34
    c = 3.0e+8
    k = 1.38e-23
    wav_nm = wav * 1e-9
    a = 2.0 * h * c**2
    b = h * c / (wav_nm * k * T)
    intensity = a / ((wav_nm**5) * (np.exp(b) - 1.0))
    return intensity

def list2spec(wavelengths, values):
    # Interpolate spectrum to wavelengths that colormath expects
    interp_spec = spec2spec(wavelengths, values, COLORMATH_WAVELENGTHS)
    return SpectralColor(*interp_spec, observer='2', illuminant='e')

def spec2spec(wavelengths1, values1, wavelengths2):
    """ Interpolate a spectrum sampled with intervals wavelengths1 to intervals wavelengths2. """
    interp_spec = interp1d(wavelengths1, values1, kind='cubic', bounds_error=False, fill_value='extrapolate')
    return interp_spec(wavelengths2)

def spec2rgb(wavelengths, values):
    spec = list2spec(wavelengths, values)
    rgb = convert_color(spec, sRGBColor, native_illuminant='e')
    return clamp_rgb(rgb.get_value_tuple())

def rgb2lab(rgb_):
    rgb = sRGBColor(rgb_[0], rgb_[1], rgb_[2])
    lab = convert_color(rgb, LabColor, target_illuminant='e')
    return lab.get_value_tuple()

def rgb2hex(rgb):
    return '#' + f'{rgb[0]:0>2x}' + f'{rgb[1]:0>2x}' + f'{rgb[2]:0>2x}'

def clamp(val, minval, maxval):
    return min(maxval, max(minval, val))

def clamp_rgb(color):
    return (clamp(color[0], 0., 1.), clamp(color[1], 0., 1.), clamp(color[2], 0., 1.))
