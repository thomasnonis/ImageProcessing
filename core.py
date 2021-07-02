import cv2 as cv
import numpy as np
import copy

from filters import Kernel, FunctionFilter

# Contrast curve
class Curve:


    def __init__(self, shadows_coeff, mid_coeff, highlights_coeff, shadows_boundary, highlights_boundary):
        self.shadows_coeff = shadows_coeff              #alpha
        self.mid_coeff = mid_coeff                      #beta
        self.highlights_coeff = highlights_coeff        #gamma
        self.shadows_boundary = shadows_boundary        #a
        self.highlights_boundary = highlights_boundary  #b

    def get_value_at_shadows_boundary(self):
        return self.shadows_coeff * self.shadows_boundary

    def get_value_at_highlights_boundary(self):
        return (self.highlights_coeff * self.highlights_boundary) + self.get_value_at_shadows_boundary()


# The Image core is compatible with multi-dimensional images.
# The methods are implemented only for grayscale for testing

class Image:

    def __init__(self, img):
        self.img = img
        self.height = img.shape[0]

        if(len(img.shape) > 1):
            self.width = img.shape[1]
        else:
            self.width = 1

        # written badly
        if(len(img.shape) > 2):
            self.channels = img.shape[2]
        else:
            self.channels = 1
        
        # self.bitdepth = cv.depth(img)      #per-channel
        self.bitdepth = 8    #TODO: fix
        self.path = None
        self.filename = None
        self.extension = None

    @classmethod
    def from_path(cls, path, read_mode = cv.IMREAD_UNCHANGED):
        ret = cls(cv.imread(path, read_mode))
        ret.path = path
        ret.extension = path.split('.')[-1] 
        ret.filename = path.split('/')[-1].split('.')[0]
        return ret

    def get_copy(self):
        return copy.deepcopy(self)


    ############################################
    ################ UTILITIES #################
    ############################################


    def show(self, window_title = None):
        # Can be used in the middle of the processing chain, as it returns self.
        # Useful for showing partial processing
        if (self.img is None):
            raise ValueError("Trying to show 'None' image")
        # set window title
        if(window_title is None):
            if(self.path != None):
                window_title = self.path
            else:
                window_title = "Image"

        # show image
        cv.imshow(window_title, self.img)

        # return self, so that it can be done while doing an assignment
        # e.g.: img = lena.get_copy().show()
        return self

    def write(self, filename = None):
        # Can be used in the middle of the processing chain, as it returns self.
        # Useful for saving partial processing
        if (filename is None and self.path is not None and self.filename is not None and self.extension is not None):
            # If filename is not provided, use original one and save with "_EDIT" suffix
            filename = self.path.split(self.filename)[0] + self.filename + "_EDIT." + self.extension
        
        if(cv.imwrite(filename, self.img) == False):
            print("Failed to write to file (" + filename + ")")
        # return self, so that it can be done while doing an assignment
        # e.g.: img = lena.get_copy().write("C://")
        return self

    def size(self):
        return (self.height, self.width, self.channels)

    ############################################
    ############ OPERATOR OVERLOADS ############
    ############################################

    def __add__(self, other):
        return Image(np.add(self.img, other.img))

    def __sub__(self, other):
        return Image(np.subtract(self.img, other.img))

    def __iadd__(self, other):
        self.img += other.img
        return self

    def __isub__(self, other):
        self.img -= other.img
        return self

    ############################################
    ############# COLOR TRANSFORMS #############
    ############################################

    def make_grayscale(self):
        self.img = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)

        # TODO: avoid replicating conditions, make setter for img
        self.height = self.img.shape[0]

        if(len(self.img.shape) > 1):
            self.width = self.img.shape[1]
        else:
            self.width = 1

        # written badly
        if(len(self.img.shape) > 2):
            self.channels = self.img.shape[2]
        else:
            self.channels = 1

        return self

    ############################################
    ############# POINT OPERATIONS #############
    ############################################

    def get_max_value(self):
        max = 0

        for px in np.nditer(self.img, op_flags=["readonly"]):
            if px > max:
                max = px

        return max

    def get_min_value(self):
        min = 2 ** self.bitdepth

        for px in np.nditer(self.img, op_flags=["readonly"]):
            if px < min:
                min = px

        return min

    def negative(self):
        max_value = (2 ** self.bitdepth) - 1

        for px in np.nditer(self.img, op_flags=["readwrite"]):
            px[...] = max_value - px

        return self

    def threshold(self, threshold):
        max_value = (2 ** self.bitdepth) - 1
        print(max_value)

        if threshold < 0:
            threshold = 0
        elif threshold > max_value:
            threshold = max_value

        for px in np.nditer(self.img, op_flags=["readwrite"]):
            if px <  threshold:
                px[...] = 0
            else:
                px[...] = max_value

        #TODO: should return binary image
        return self
    
    def contrast_curve(self, curve: Curve):
        for px in np.nditer(self.img, op_flags=["readwrite"]):
            if px <= curve.shadows_boundary:
                px[...] = curve.shadows_coeff * px
            elif px < curve.highlights_boundary:
                px[...] = curve.mid_coeff * (px - curve.shadows_boundary) + curve.get_value_at_shadows_boundary()
            else:
                px[...] = curve.highlights_coeff * (px - curve.highlights_boundary) + curve.get_value_at_highlights_boundary()


        return self

    def get_bitplanes(self):
        pass

    ############################################
    ################ FILTERING #################
    ############################################

    def filter(self, kernel, ddepth = -1, force_inseparable: bool = False):
        # NonLinear filter objects return the function to execute
        if(isinstance(kernel, FunctionFilter)):
            self.img = kernel.apply(self)
        
        else:
            if(kernel.is_separable() and not force_inseparable):
                # Convolute axes separately
                # print("Executing separated 2D Convolution")
                self.img = cv.filter2D(self.img, ddepth, kernel.sep_kernel_y)
                self.img = cv.filter2D(self.img, ddepth, kernel.sep_kernel_x)
            else:
                # Normal convolution
                # print("Executing 2D Convolution")
                self.img = cv.filter2D(self.img, ddepth, kernel.kernel)
        
        return self

class CameraSource(Image):
    # Could implement as separate class, not subclass -> Image is an attribute
    # VideoCapture will not allow to deep copy -> separate class
    def __init__(self, cam):
        self.camera = cv.VideoCapture(cam)
        _, self.current_frame = self.camera.read()

        super().__init__(self.current_frame)

    def __del__(self):
        self.release_camera()

    def get_next_frame(self):
        # Could also return an Image object
        if(self.camera is not None):
            _, frame = self.camera.read()
            self.img = frame
        return self

    def release_camera(self):
        if(self.camera is not None):
            self.camera.release()

    def make_image(self):
        # May be useful, especially because it is not possible to make a deepcopy of this class, because of VideoCapture
        return Image(self.img)

    def get_copy(self):
        # Returns a copy of the frame, casted as an Image object
        return self.make_image().get_copy()