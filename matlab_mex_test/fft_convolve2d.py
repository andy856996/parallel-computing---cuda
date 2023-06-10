import numpy as np

def fft_conv2d(image, kernel):
    # Pad the kernel to the size of the image
    kernel_padded = np.zeros_like(image)
    kh, kw = kernel.shape
    kernel_padded[:kh, :kw] = kernel
    
    # Compute FFT of image and kernel
    image_fft = np.fft.fft2(image)
    kernel_fft = np.fft.fft2(kernel_padded)
    
    # Compute product of FFTs and inverse FFT
    result = np.fft.ifft2(image_fft * kernel_fft)
    
    # Take real part of result (imaginary part should be very small)
    return np.real(result)

