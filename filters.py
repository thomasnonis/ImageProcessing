'''
    Author: Thomas Nonis
'''

import numpy as np
import cv2 as cv

class Kernel:
    '''Class for filters represented by a kernel matrix, that can be computed via convolution.'''

    def __init__(self, kernel, sep_kernel_x = None, sep_kernel_y = None):
        '''Constructs the Kernel object from a kernel matrix and, optionally, from its separated form.

        If the separated kernel components are not passed and the kernel is separable,
        the constructor will generate the components by applying SVD decomposition.

        Args:
            kernel (list | array): Matrix of the kernel
            sep_kernel_x (list | array, optional): X component of the separable kernel. Defaults to None.
            sep_kernel_y (list | array, optional): Y component of the separable kernel. Defaults to None.
        '''
        self.kernel = kernel
        self.is_auto_separated = False  #to keep track of whether the user provided separated filter
        if(sep_kernel_x is None and sep_kernel_y is None and self.is_separable()):
            self.sep_kernel_x, self.sep_kernel_y = self.decompose()
            self.is_auto_separated = True
        else:
            self.sep_kernel_x = sep_kernel_x
            self.sep_kernel_y = sep_kernel_y
            

    def is_separable(self):
        '''Check whether the kernel is separable

        Returns:
            bool: True if the kernel is separable, False otherwise
        '''
        if(np.linalg.matrix_rank(self.kernel) > 1):
            return False
        return True

    def decompose(self):
        '''Computes the SVD decomposition of the matrix,returning the 2 components
        of the separated kernel.

        Note: Because of the irrelevance of the eigenvalues' sign, the sign of the returned matrices
        might not be always correct. Filter directionality might be affected.

        Returns:
            (list, list): 2 element tuple with the x and y separated kernel components respectively.
        '''
        U, S, V = np.linalg.svd(self.kernel)
        sep_kernel_x = -V[0,:] * np.sqrt(S[0])
        sep_kernel_y = -U[:,0][:,None] * np.sqrt(S[0])  #[:,None] does the transposition
        return sep_kernel_x, sep_kernel_y

    ############################################
    ########### OPERATOR OVERLOADS #############
    ############################################

    def __mul__(self, coeff: float):
        '''Multiplication operator overload

        Multiplies the kernel matrices by the coefficient passed as argument.
        Note: Both the kernel and its separated components are multiplied by the same amount.

        Args:
            coeff (float): Multiplication coefficient

        Returns:
            Kernel: self
        '''
        self.kernel *= coeff
        if self.sep_kernel_y is not None and self.sep_kernel_x is not None:
            self.sep_kernel_x *= coeff
            self.sep_kernel_y *= coeff
        return self

    ############################################
    ######### STANDARD FILTER KERNELS ##########
    ############################################

    @classmethod
    def LPF(cls, size: int):
        '''LPF or MA filter

        Generates an instance of a LPF kernel of the specified size

        Args:
            size (int): Size of the square kernel

        Returns:
            Kernel: LPF Kernel
        '''
        if (size % 2 == 0):
            print("Warning: Kernel size is even (" + str(size) + ")")

        kernel = np.ones( (size,size), np.float32 ) / size ** 2
        sep_kernel_x = np.ones( size, np.float32 ) / size
        sep_kernel_y = sep_kernel_x[:,None]
        return cls(kernel, sep_kernel_x, sep_kernel_y)

    @classmethod
    def HPF(cls, size: int):
        '''HPF filter

        Generates an instance of a HPF kernel of the specified size

        Args:
            size (int): Size of the square kernel

        Returns:
            Kernel: HPF Kernel
        '''
        if (size % 2 == 0):
            print("Warning: Kernel size is even (" + str(size) + ")")

        one_mtx = np.zeros( (size, size), np.float32 )
        one_mtx[int((size/2))][int((size/2))] = 1

        return cls(one_mtx - cls.LPF(size).kernel)

    @classmethod
    def gaussian(cls, size: int, sigma: float):
        '''Gaussian LPF filter

        Generates an instance of a gaussian LPF kernel of the specified size

        Args:
            size (int): Size of the square kernel
            sigma (float): Standard deviation of the gaussian curve

        Returns:
            Kernel: Gaussian LPF Kernel
        '''
        if (size % 2 == 0):
            print("Warning: Kernel size is even (" + str(size) + ")")

        # getGaussianKernel returns 1D vector of already separated kernel
        sep_kernel_y = cv.getGaussianKernel(size, sigma)
        sep_kernel_x = sep_kernel_y.transpose((1,0))

        return cls(np.matmul(sep_kernel_y, sep_kernel_x), sep_kernel_x, sep_kernel_y)
    
    @classmethod
    def crispening(cls, size: int):
        '''Crispening filter

        Generates an instance of a crispening filter kernel of the specified size

        Args:
            size (int): Size of the square kernel

        Returns:
            Kernel: Crispening filter Kernel
        '''
        if (size % 2 == 0):
            print("Warning: Kernel size is even (" + str(size) + ")")
        center_mtx = np.zeros( (size, size), np.float32 )
        center_mtx[int((size/2))][int((size/2))] = 2
        return cls(center_mtx - cls.LPF(size).kernel)

    ############################################
    ########## EDGE DETECTION KERNELS ##########
    ############################################

    @classmethod
    def prewitt(cls, normalize: bool = False):
        '''Prewitt edge detection filter

        Generates an instance of a Prewitt edge detection filter kernel
        of the specified size

        Args:
            normalize (bool): If True the kernel will be normalized. Defaults to False.

        Returns:
            Kernel: Prewitt Kernel
        '''
        kernel = np.array(  [
                            [-1, 0, 1],
                            [-1, 0, 1],
                            [-1, 0, 1]
                            ], dtype=float)
        sep_kernel_x = np.array([-1, 0, 1], dtype=float)
        sep_kernel_y = np.array([[1],[1],[1]], dtype=float)

        if(normalize):
            kernel = np.float32(kernel) / 6
            sep_kernel_x = np.float32(sep_kernel_x) / 2
            sep_kernel_y = np.float32(sep_kernel_y) / 3

        return cls(kernel, sep_kernel_x, sep_kernel_y)


# Max and Min are Dilate and Erode
# https://stackoverflow.com/questions/39772796/min-max-avg-filters-in-opencv2-4-13/39778555

class FunctionFilter:
    '''Superclass for filters that return the function that executes theirselves'''
    def __init__(self, apply):
        '''Generates the FunctionFilter object

        Args:
            apply (function): function that is executed to apply the filter
        '''
        self.apply = apply    #function to be executed to apply filter

class NonLinear(FunctionFilter):
    '''Class for non-linear filters. Each filter returns the function that applies it'''

    @classmethod
    def median(cls, size):
        '''Median filter

        Args:
            size (int): Kernel size

        Returns:
            NonLinear: NonLinear filter object with the correct apply function set as argument
        '''
        def apply(img):
            return cv.medianBlur(img.img, size)
        return cls(apply)

class Morph(FunctionFilter):
    '''Class for morphological filters. Each filter returns the function that applies it'''
    
    # There is room for improvement: add other parameters (border_type, anchor, ...)

    @classmethod
    def erode(cls, struct_el, iterations = 1):
        '''Erosion filter

        Args:
            struct_el (list | array): structuring element
            iterations (int): number of iterations of the filter

        Returns:
            NonLinear: NonLinear filter object with the correct apply function set as argument
        '''
        def apply(img):
            return cv.erode(img.img, struct_el, iterations = iterations)
        return cls(apply)

    @classmethod
    def dilate(cls, struct_el, iterations = 1):
        '''Dilation filter

        Args:
            struct_el (list | array): structuring element
            iterations (int): number of iterations of the filter

        Returns:
            NonLinear: NonLinear filter object with the correct apply function set as argument
        '''
        def apply(img):
            return cv.dilate(img.img, struct_el, iterations = iterations)
        return cls(apply)

    @classmethod
    def opening(cls, struct_el, iterations = 1):
        '''Opening filter

        Executes the erosion first and then executes the dilation on the eroded image

        Args:
            struct_el (list | array): structuring element
            iterations (int): number of iterations of the filter

        Returns:
            NonLinear: NonLinear filter object with the correct apply function set as argument
        '''
        def apply(img):
            return cv.morphologyEx(img.img, cv.MORPH_OPEN, struct_el, iterations = iterations)
        return cls(apply)

    @classmethod
    def closing(cls, struct_el, iterations = 1):
        '''Closing filter

        Executes the dilation first and then executes the erosion on the dilated image 

        Args:
            struct_el (list | array): structuring element
            iterations (int): number of iterations of the filter

        Returns:
            NonLinear: NonLinear filter object with the correct apply function set as argument
        '''
        def apply(img):
            return cv.morphologyEx(img.img, cv.MORPH_CLOSE, struct_el, iterations = iterations)
        return cls(apply)

class EdgeDetection(FunctionFilter):
    '''Class for edge detection filters. Each filter returns the function that applies it'''

    @classmethod
    def canny(cls, min, max):
        '''Canny edge detection algorithm

        Args:
            min (float): min value for the hysteresis thresholding criteria
            max (float): max value for the hysteresis thresholding criteria

        Returns:
            EdgeDetection: EdgeDetection filter object with the correct apply function set as argument
        '''
    # def canny(cls, min, max, kernel_size = 3, L2gradient: bool = False):
        # default values are the same default values of cv.Canny()
        def apply(img):
            return cv.Canny(img.img, min, max)
        return cls(apply)