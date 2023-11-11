import numpy as np
from scipy.io import loadmat, savemat
from scipy.stats import median_abs_deviation
from scipy.fft import fft2, fftshift, ifft2
import matplotlib.pyplot as plt

def decorrelate(input_file, output_file, f_x, f_y, m_x, m_y, cf):
    # Load complex SAR image
    # complex_SAR = loadmat(input_file)
    inquad = input_file[1]
    inphase = input_file[0]
    img_complex = inphase + 1j * inquad

    r, c = img_complex.shape

    # Calculate intensity image
    intensity_img = np.abs(img_complex)**2
    median_value = np.median(intensity_img)
    threshold = cf * median_value

    # Thresholding
    index_nonpoints = intensity_img < threshold
    index_points = intensity_img >= threshold

    # Replacement of point targets
    n_nonpoints = np.sum(index_nonpoints)
    n_points = np.sum(index_points)
    var = np.sum(intensity_img[index_nonpoints]) / n_nonpoints

    new_points = np.sqrt(var / 2) * (np.random.randn(n_points) + 1j * np.random.randn(n_points))

    img_complex_new = img_complex.copy()
    img_complex_new[index_points] = new_points

    # Equalization
    cout, W = equalizeIQ(img_complex_new, m_x, m_y, f_x, f_y)

    # Plot frequency spectrum
    fC = fft2(cout)
    S = np.real(fC * np.conj(fC))

    temp1 = np.sqrt(np.mean(fftshift(S), axis=0))
    temp1 /= np.max(temp1)
    x1 = np.linspace(-1, 1, r)
    plt.figure()
    plt.plot(x1, temp1, 'o')

    temp1 = np.sqrt(np.mean(fftshift(S), axis=1))
    temp1 /= np.max(temp1)
    x1 = np.linspace(-1, 1, c)
    plt.figure()
    plt.plot(x1, temp1, 'o')

    # Replacement of point targets (Undo operation)
    cout[index_points] = img_complex[index_points]

    # Save result
    output_data = {'cout': cout}
    savemat(output_file, output_data)

def equalizeIQ(C, mx, my, rx, ry):
    r, c = C.shape
    M = np.exp(-1j * np.pi * np.outer(np.arange(r), my)) * np.exp(-1j * np.pi * np.outer(np.arange(c), mx))
    C2 = M * C

    fC = fft2(C2)
    S = np.real(fC * np.conj(fC))
    R = ifft2(S)

    rho_x = np.abs(R[0, 1] / R[0, 0])
    rho_y = np.abs(R[1, 0] / R[0, 0])

    x1 = np.arange(c)
    y1 = np.sqrt(np.mean(fftshift(S), axis=0))
    clipping = np.percentile(y1, 99)
    y1[y1 > clipping] = clipping
    y1 /= np.max(y1)

    p = np.polyfit(x1[1 + round(c * rx): -round(c * rx)], y1[1 + round(c * rx): -round(c * rx)], 70)
    yi1 = np.polyval(p, x1)

    plt.figure()
    plt.plot(x1, y1, '*', x1[1 + round(c * rx): -round(c * rx)], yi1[1 + round(c * rx): -round(c * rx)], 'o')

    x2 = np.arange(r)
    y2 = np.sqrt(np.mean(fftshift(S), axis=1))
    clipping = np.percentile(y2, 99)
    y2[y2 > clipping] = clipping
    y2 /= np.max(y2)

    p = np.polyfit(x2[1 + round(r * ry): -round(r * ry)], y2[1 + round(r * ry): -round(r * ry)], 70)
    yi2 = np.polyval(p, x2)

    plt.figure()
    plt.plot(x2, y2, '*', x2[1 + round(r * ry): -round(r * ry)], yi2[1 + round(r * ry): -round(r * ry)], 'o')

    G = np.outer(yi2, yi1)
    mask = np.zeros((r, c))
    mask[1 + round(r * ry): -round(r * ry), 1 + round(c * rx): -round(c * rx)] = 1
    G = G * mask
    G = G / np.linalg.norm(G.flatten())
    W = 1 / G / np.sqrt(np.sum(mask.flatten()))
    W[~mask.astype(bool)] = 1

    W = fftshift(W)

    h = np.zeros((r, c))
    r1 = round(c * rx)
    r2 = round(r * ry)
    h[1 + r2: -r2, 1 + r1: -r1] = 1
    h = fftshift(h)

    fC = fC * h
    fCE = fC * W
    CE = ifft2(fCE)

    RE = ifft2(S * W**2)
    rho_xe = np.abs(RE[0, 1] / RE[0, 0])
    rho_ye = np.abs(RE[1, 0] / RE[0, 0])

    return CE, W

def inspect_spectrum(input_file):
    inquad = input_file[1]
    inphase = input_file[0]
    img_complex = inphase + 1j * inquad

    r, c = img_complex.shape

    fC = fft2(img_complex)
    S = np.real(fC * np.conj(fC))

    temp1 = np.sqrt(np.mean(fftshift(S), axis=0))
    temp1 /= np.max(temp1)
    x1 = np.linspace(-1, 1, r)
    plt.figure()
    plt.plot(x1, temp1, 'o')

    temp1 = np.sqrt(np.mean(fftshift(S), axis=1))
    temp1 /= np.max(temp1)
    x1 = np.linspace(-1, 1, c)
    plt.figure()
    plt.plot(x1, temp1, 'o')

    plt.show()

# # Example usage
# input_file = 'your_input_file.mat'
# output_file = 'your_output_file.mat'
# f_x = 0  # Replace with your values
# f_y = 0  # Replace with your values
# m_x = 0  # Replace with your values
# m_y = 0  # Replace with your values
# cf = 1.5  # Replace with your values

# decorrelate(input_file, output_file, f_x, f_y, m_x, m_y, cf)