#----------------------------------------------#
# create by Shun-Pin,Yeh
# Date : 2023 5 23
# National Taitung Unv. IPGIT
# Email : 10922127@gm.nttu.edu.tw
#----------------------------------------------#
import numpy as np

def fft_conv2d(image, kernel):
    # Compute FFT of image and kernel
    image_fft = np.fft.fft2(image)
    kernel_fft = np.fft.fft2(kernel)
    
    # Compute product of FFTs and inverse FFT
    result = np.fft.ifft2(image_fft * kernel_fft)
    
    # Take real part of result (imaginary part should be very small)
    return np.real(result)

