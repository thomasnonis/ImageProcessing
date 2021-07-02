import cv2 as cv
import numpy as np
import copy

from filters import Kernel, FunctionFilter

class Curve:
    '''
    Class used to represent a 3-segment curve for contrast

    ...

    Attributes
    ----------
    shadow_coeff : uint
        angle coefficient of the shadow segment of the curve
    mid_coeff : uint
        angle coefficient of the mid-tones segment of the curve
    highlights_coeff : uint
        angle coefficient of the highlights segment of the curve
    shadows_boundary : uint
        point where the shadow segment ends and the mid-tones segment starts
    highlights_boundary : uint
        point where the mid-tones segment ends and the highlights segment starts

    Methods
    -------
    get_value_at_shadows_boundary():
        calculates and returns the point, along the y axis, at which the shadow boundary occurs

    get_value_at_highlights_boundary():
        calculates and returns the point, along the y axis, at which the highlights boundary occurs
    '''    

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


class Image:
    '''
    Class used to represent an image and the operations that can be performed with it

    The Image core is compatible with multi-dimensional images.
    Some methods are only implemented for grayscale, for testing

    ...

    Attributes
    ----------
    img : list | np.array
        actual image represented as matrix
    height : uint
        height of the image
    width : uint
        width of the image
    channels : uint
        number of channels of the image
    bitdepth : uint
        bitdepth per-channel of the image
    path : str
        path of the original image
    filename : str
        name of the original image without extension
    extension : str
        file extension of the original image

    Methods
    -------
    from_path(path, read_mode = cv.IMREAD_UNCHANGED):
        create Image instance from an image file
    '''

    def __init__(self, img):
        '''
        Constructs the Image object from a matrix of pixels

        Parameters
        ----------
            img : list | np.array
                the image matrix
        '''
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
        '''Creates an Image instance from an image file

        This method also allows to include filename, path and extension to the object

        Args:
            path (str): path of the image to open
            read_mode (readmode, optional): specifies the read mode. Defaults to cv.IMREAD_UNCHANGED.

        Returns:
            Image: newly created Image object
        '''
        ret = cls(cv.imread(path, read_mode))
        ret.path = path
        ret.extension = path.split('.')[-1] 
        ret.filename = path.split('/')[-1].split('.')[0]
        return ret

    def get_copy(self):
        '''Returns a (deep)copy of the Image object. It does not return a reference to the calling object.

        Returns:
            Image: new copied Image object
        '''
        return copy.deepcopy(self)


    ############################################
    ################ UTILITIES #################
    ############################################


    def show(self, window_title = None):
        '''Displays the image on the screen. 

        It can be used in the middle of the processing chain or in an assignment, as it returns self.
        Useful for showing partial processing, like:

            img = img.get_copy().show()

        or

            img.negative().show().filter([...])

        Args:
            window_title (str, optional): The name of the window. Defaults to None.

        Raises:
            ValueError: Is raised if the method is called, but the img attribute is None

        Returns:
            Image: self
        '''
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

        return self

    def write(self, filename = None):
        '''Saves the image to a file.

        It can be used in the middle of the processing chain or in an assignment, as it returns self.
        Useful for saving partial processing, like:

            img = lena.get_copy().write("C://")

        Args:
            filename (str, optional): filename | path of where to save the image.
                If it is not provided and the image was imported from a file,
                the image is saved using the original path with the "_EDIT" suffix.
                Defaults to None.

        Returns:
            Image: self
        '''
        if (filename is None and self.path is not None and self.filename is not None and self.extension is not None):
            # If filename is not provided, use original one and save with "_EDIT" suffix
            filename = self.path.split(self.filename)[0] + self.filename + "_EDIT." + self.extension
        
        if(cv.imwrite(filename, self.img) == False):
            print("Failed to write to file (" + filename + ")")

        return self

    def size(self):
        '''Returns the size of the image in the form of a tuple (height, width, channels)

        Returns:
            (int, int, int): Size of the image (height, width, channels)
        '''
        return (self.height, self.width, self.channels)

    ############################################
    ############ OPERATOR OVERLOADS ############
    ############################################

    def __add__(self, other):
        '''Plus (+) operator overload.

        Args:
            other (Image): Image to add

        Returns:
            Image: A new Image object with the image as the sum of the matrices of the images
        '''
        return Image(np.add(self.img, other.img))

    def __sub__(self, other):
        '''Minus (-) operator overload.

        Args:
            other (Image): Image to subtract

        Returns:
            Image: A new Image object with the image as the difference of the matrices of the images
        '''
        return Image(np.subtract(self.img, other.img))

    def __iadd__(self, other):
        '''Plus assignment (+=) operator overload.

        Args:
            other (Image): Image to add

        Returns:
            Image: self
        '''
        self.img += other.img
        return self

    def __isub__(self, other):
        '''Minus assignment (-=) operator overload.

        Args:
            other (Image): Image to subtract

        Returns:
            Image: self
        '''
        self.img -= other.img
        return self

    ############################################
    ############# COLOR TRANSFORMS #############
    ############################################

    def make_grayscale(self):
        '''Converts the image to grayscale (assuming it was BGR)

        Returns:
            Image: self
        '''
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
        '''Returns the max value of the image

        Returns:
            int | float: the max value of the image
        '''
        max = 0

        for px in np.nditer(self.img, op_flags=["readonly"]):
            if px > max:
                max = px

        return max

    def get_min_value(self):
        '''Returns the min value of the image

        Returns:
            int | float: the min value of the image
        '''
        min = 2 ** self.bitdepth

        for px in np.nditer(self.img, op_flags=["readonly"]):
            if px < min:
                min = px

        return min

    def negative(self):
        '''Computes the negative of the image, considering its bitdepth.

        Returns:
            Image: self
        '''
        max_value = (2 ** self.bitdepth) - 1

        for px in np.nditer(self.img, op_flags=["readwrite"]):
            px[...] = max_value - px

        return self

    def threshold(self, threshold):
        '''Executes a thresholding of the image.

        If the threshold is negative, 0 is considered.
        If the threshold is greater than the maximum value allowed by the bitdepth,
            the maximum value allowed (2 ** bitdepth - 1) is considered

        All values below the threshold are set to 0.
        All values greater or equal to the threshold are set to 2 ** bitdepth - 1

        Args:
            threshold (int | float): Threshold value

        Returns:
            Image: self
        '''
        max_value = (2 ** self.bitdepth) - 1

        if threshold < 0:
            threshold = 0
        elif threshold > max_value:
            threshold = max_value

        for px in np.nditer(self.img, op_flags=["readwrite"]):
            if px <  threshold:
                px[...] = 0
            else:
                px[...] = max_value

        #TODO: should return binary image (?)
        return self
    
    def contrast_curve(self, curve: Curve):
        '''Applies the contrast curve to the image.

        Args:
            curve (Curve): The contrast curve

        Returns:
            Image: self
        '''
        for px in np.nditer(self.img, op_flags=["readwrite"]):
            if px <= curve.shadows_boundary:
                px[...] = curve.shadows_coeff * px
            elif px < curve.highlights_boundary:
                px[...] = curve.mid_coeff * (px - curve.shadows_boundary) + curve.get_value_at_shadows_boundary()
            else:
                px[...] = curve.highlights_coeff * (px - curve.highlights_boundary) + curve.get_value_at_highlights_boundary()


        return self

    def get_bitplanes(self):
        #TODO
        pass

    ############################################
    ################ FILTERING #################
    ############################################

    def filter(self, kernel, ddepth = -1, force_inseparable: bool = False):
        '''Computes the filtering of the image, accordingly to the type of kernel it is passed.

        If the kernel is convolvable, it applies the convolution.
        If it is also separable it also applies the filter separately for each direction,
        unless it is forced not to by setting:
            
            force_inseparable = True

        If instead the kernel is not convolvable, or there is an available function from openCV,
        it executes the filter function

        Args:
            kernel (Kernel | NonLinear | Morph | EdgeDetection): filter object
            ddepth (int, optional): The depth of returned image.
                A negative value (such as −1) indicates that
                the depth is the same as the source.
                Defaults to -1.
            force_inseparable (bool, optional): If set to true and the filter is separable,
                it forces the execution of the convolution of the full kernel,
                without separating the axes.
                Defaults to False.

        Returns:
            Image: self
        '''
        # NonLinear filter objects return the function to execute
        if(isinstance(kernel, FunctionFilter)):
            self.img = kernel.apply(self)
        
        else:
            if(kernel.is_separable() and not force_inseparable):
                # Convolute axes separately
                self.img = cv.filter2D(self.img, ddepth, kernel.sep_kernel_y)
                self.img = cv.filter2D(self.img, ddepth, kernel.sep_kernel_x)
            else:
                # Normal convolution
                self.img = cv.filter2D(self.img, ddepth, kernel.kernel)
        
        return self

class CameraSource(Image):
    '''
    Image subclass for acquiring the image from a camera source

    Dev note: It is implemented as a subclass of image,
        but it could implemented separately, with an Image object as an attribute.
        It cannot be implemented inside Image, thus without using another class,
        because the VideoCapture object does not allow deepcopying.
        (should write own deepcopy method to do so)

    ...

    Attributes
    ----------
    camera:
        VideoCapture source
    '''
    def __init__(self, cam = 0):
        '''Constructs the CameraSource object and reads a frame.

        Args:
            cam (int, optional): Selects the camera source.
                If not provided the first source from the system camera list is chosen.
                Defaults to 0.
        '''
        self.camera = cv.VideoCapture(cam)
        _, current_frame = self.camera.read()

        super().__init__(current_frame)

    def __del__(self):
        '''Release the camera upon object destruction
        '''
        self.release_camera()

    def get_next_frame(self):
        '''Gets the next frame from the camera source

        Dev note: It could be implemented to return an Image object

        Returns:
            CameraSource: self
        '''
        if(self.camera is not None):
            _, frame = self.camera.read()
            self.img = frame
        return self

    def release_camera(self):
        '''Release the camera'''
        if(self.camera is not None):
            self.camera.release()

    def make_image(self):
        '''returns a new Image object created from the CameraCapture frame.

        Returns:
            Image: New Image object from the CameraCapture frame
        '''
        # May be useful, especially because it is not possible to make a deepcopy of this class, because of VideoCapture
        return Image(self.img)

    def get_copy(self):
        '''Returns a copy of the frame, casted as an Image object

        Returns:
            Image: New copied Image object created from the CameraCapture frame
        '''
        return self.make_image().get_copy()