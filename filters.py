import numpy as np
import cv2 as cv

class Kernel:
    # Class for filters that can be computed via convolution
    # Every classmethod returns a kernel

    def __init__(self, kernel, sep_kernel_x = None, sep_kernel_y = None):
        self.kernel = kernel
        self.is_auto_separated = False  #to keep track of whether the user provided separated filter
        if(sep_kernel_x is None and sep_kernel_y is None and self.is_separable()):
            self.sep_kernel_x, self.sep_kernel_y = self.decompose()
            self.is_auto_separated = True
        else:
            self.sep_kernel_x = sep_kernel_x
            self.sep_kernel_y = sep_kernel_y
            

    def is_separable(self):
        if(np.linalg.matrix_rank(self.kernel) > 1):
            return False
        return True

    def decompose(self):
        # TODO: check sign of kernels. Sometimes sign has the correct sign, sometimes not
        # (sign of eigenvector is irrelevant, but influences filter directionality (if not abs()))
        U, S, V = np.linalg.svd(self.kernel)
        sep_kernel_x = -V[0,:] * np.sqrt(S[0])
        sep_kernel_y = -U[:,0][:,None] * np.sqrt(S[0])  #[:,None] does the transposition
        return sep_kernel_x, sep_kernel_y

    ############################################
    ########### OPERATOR OVERLOADS #############
    ############################################

    def __mul__(self, coeff: float):
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
        if (size % 2 == 0):
            print("Warning: Kernel size is even (" + str(size) + ")")

        kernel = np.ones( (size,size), np.float32 ) / size ** 2
        sep_kernel_x = np.ones( size, np.float32 ) / size
        sep_kernel_y = sep_kernel_x[:,None]
        return cls(kernel, sep_kernel_x, sep_kernel_y)

    @classmethod
    def HPF(cls, size: int):
        if (size % 2 == 0):
            print("Warning: Kernel size is even (" + str(size) + ")")

        one_mtx = np.zeros( (size, size), np.float32 )
        one_mtx[int((size/2))][int((size/2))] = 1

        return cls(one_mtx - cls.LPF(size).kernel)

    @classmethod
    def gaussian(cls, size: int, sigma: float):
        if (size % 2 == 0):
            print("Warning: Kernel size is even (" + str(size) + ")")

        # getGaussianKernel returns 1D vector of already separated kernel
        sep_kernel_y = cv.getGaussianKernel(size, sigma)
        sep_kernel_x = sep_kernel_y.transpose((1,0))

        return cls(np.matmul(sep_kernel_y, sep_kernel_x), sep_kernel_x, sep_kernel_y)
    
    @classmethod
    def crispening(cls, size: int):
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
    def __init__(self, apply):
        self.apply = apply    #function to be executed to apply filter
    pass

class NonLinear(FunctionFilter):

    @classmethod
    def median(cls, size):
        def apply(img):
            return cv.medianBlur(img.img, size)
        return cls(apply)

class Morph(FunctionFilter):
    # There is room for improvement: add other parameters (border_type, anchor, ...)

    @classmethod
    def erode(cls, struct_el, iterations = 1):
        def apply(img):
            return cv.erode(img.img, struct_el, iterations = iterations)
        return cls(apply)

    @classmethod
    def dilate(cls, struct_el, iterations = 1):
        def apply(img):
            return cv.dilate(img.img, struct_el, iterations = iterations)
        return cls(apply)

    @classmethod
    def opening(cls, struct_el, iterations = 1):
        def apply(img):
            return cv.morphologyEx(img.img, cv.MORPH_OPEN, struct_el, iterations = iterations)
        return cls(apply)

    @classmethod
    def closing(cls, struct_el, iterations = 1):
        def apply(img):
            return cv.morphologyEx(img.img, cv.MORPH_CLOSE, struct_el, iterations = iterations)
        return cls(apply)

class EdgeDetection(FunctionFilter):

    @classmethod
    def canny(cls, min, max):
    # def canny(cls, min, max, kernel_size = 3, L2gradient: bool = False):
        # default values are the same default values of cv.Canny()
        def apply(img):
            return cv.Canny(img.img, min, max)
        return cls(apply)