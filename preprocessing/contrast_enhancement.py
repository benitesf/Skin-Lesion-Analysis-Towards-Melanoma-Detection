from scipy.ndimage.filters import median_filter
from preprocessing import shadding_attenuation as shatt
from math import floor, sqrt
from skimage.color import rgb2hsv, hsv2rgb
import numpy as np



def median_filter_(img):
    M, N = img.shape[0:2]
    n = floor(5 * sqrt((M/768) * (N/512)))
    filtered = np.zeros(img.shape, dtype='uint8')

    filtered[:,:,0] = median_filter(img[:,:,0], size=n)
    filtered[:,:,1] = median_filter(img[:,:,1], size=n)
    filtered[:,:,2] = median_filter(img[:,:,2], size=n)

    return filtered


def shading_attenuation_method(image, extract, margin):
    """
    Apply the shading attenuation method to an image

    Parameters
    ----------
    image: 3D array
        The image source
    extract: scalar
        Number of pixel to extract, extract x extract
    margin: scalar
        Margin from the borders
    """

    hsv = rgb2hsv(image)
    V = np.copy(hsv[:, :, 2])

    shape = image.shape[0:2]

    """
    Sampling pixels
    ---------------
    """
    Yc, Xc = shatt.sampling_from_corners(margin=margin, extract=extract, shape=shape)
    Yf, Xf = shatt.sampling_from_frames(margin=margin, extract=extract, shape=shape)

    Zc = np.zeros((Xc.shape))
    Zf = np.zeros((Xf.shape))

    for j in range(0, Zc.shape[0]):
        Zc[j] = np.copy(V[Yc[j], Xc[j]])

    for j in range(0, Zf.shape[0]):
        Zf[j] = np.copy(V[Yf[j], Xf[j]])

    """
    Quadratic and cubic polynomial coefficients
    -------------------------------------------
    """
    Ac2 = shatt.quadratic_polynomial_function(Yc, Xc)
    Af2 = shatt.quadratic_polynomial_function(Yf, Xf)

    Ac3 = shatt.cubic_polynomial_function(Yc, Xc)
    Af3 = shatt.cubic_polynomial_function(Yf, Xf)

    """
    Fitting polynomial
    ------------------
    """
    coeffc2 = np.linalg.lstsq(Ac2, Zc)[0]
    coefff2 = np.linalg.lstsq(Af2, Zf)[0]

    coeffc3 = np.linalg.lstsq(Ac3, Zc)[0]
    coefff3 = np.linalg.lstsq(Af3, Zf)[0]

    """
    Processed
    ---------
    """
    Vprocc2 = shatt.apply_quadratic_function(V, coeffc2)
    Vprocf2 = shatt.apply_quadratic_function(V, coefff2)
    Vprocc3 = shatt.apply_cubic_function(V, coeffc3)
    Vprocf3 = shatt.apply_cubic_function(V, coefff3)

    # Convert Value into the range 0-1
    Vprocc2 = shatt.in_range(Vprocc2)
    Vprocf2 = shatt.in_range(Vprocf2)
    Vprocc3 = shatt.in_range(Vprocc3)
    Vprocf3 = shatt.in_range(Vprocf3)

    # Retrieve true color to skin
    muorig = V.mean()
    Vnewc2 = shatt.retrieve_color(Vprocc2, muorig)
    Vnewf2 = shatt.retrieve_color(Vprocf2, muorig)
    Vnewc3 = shatt.retrieve_color(Vprocc3, muorig)
    Vnewf3 = shatt.retrieve_color(Vprocf3, muorig)

    # Convert Value into the range 0-1
    Vnewc2 = shatt.in_range(Vnewc2)
    Vnewf2 = shatt.in_range(Vnewf2)
    Vnewc3 = shatt.in_range(Vnewc3)
    Vnewf3 = shatt.in_range(Vnewf3)

    # Select the image which have least entropy
    values = [V, Vnewc2, Vnewf2, Vnewc3, Vnewf3]
    entropy = [shatt.entropy_ratio(v) for v in values]
    index = entropy.index(min(entropy))

    hsv[:, :, 2] = np.copy(values[index])
    attenuated = hsv2rgb(hsv)

    return attenuated


"""
def plot_img_and_hist(img, axes, bins=256):
    img = img_as_float(img)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(img, cmap=plt.cm.gray)
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(img.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(img, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf
"""