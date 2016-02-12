import numpy as np
import cv2
from scipy.interpolate import UnivariateSpline


def nothing(x):
    pass

cv2.namedWindow('Tint')

# create trackbars for color change
cv2.createTrackbar('R','Tint',0,255,nothing)
cv2.createTrackbar('G','Tint',0,255,nothing)
cv2.createTrackbar('B','Tint',0,255,nothing)

# create switch for ON/OFF functionality
switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch, 'Tint',0,1,nothing)

#Set up a video capture instnance
cap=cv2.VideoCapture(0)
#capture the shape of image to generate filters of same size
l,b,n=cap.read()[1].shape

class Hotshot:
	"""
	HotShot
	We are simply subtracting the frame from a scalar value of 150. This would cause all the RGB values 
	to be subtracted by 150 with a wrap around. This creates a hot texture in the image.
	"""
	def __init__(self):
		pass
	
	def render(self,frame):
		return 150-frame

class Negative:
	"""
	Negative
	We are subtracting the RGB values from a scalar value of 255. This would render a negative strip
	kind of an effect on the image
	"""
	def __init__(self):
		pass
	
	def render(self,frame):
		return 255-frame
		
class WaterColor:
	"""
	Water Color
	
	We follow a simple methodology to implement this. We downsample the image, apply iterative bilateral 
	filtering to it to make the colors uniform and then upsample it back to the original resolution.
	We then need to perform median blurring and adaptive thresholding to the obtained image to obtain the 
	edges.
	These operatios are performed in grayscale and then converted back to RGB.
	Eventually we add the blurred image and obtained edges by means of bitwise and and return a median blurred 
	image from that. 
	"""
	def __init__(self):
		pass
	def render(self,frame):
		numDownSamples = 2
		img_rgb = frame
		# number of downscaling steps
		numBilateralFilters = 7
		# number of bilateral filtering steps
		# -- STEP 1 --
		# downsample image using Gaussian pyramid
		img_color = img_rgb
		for _ in xrange(numDownSamples):
			img_color = cv2.pyrDown(img_color)
		# repeatedly apply small bilateral filter instead of applying
		# one large filter
		for _ in xrange(numBilateralFilters):
			img_color = cv2.bilateralFilter(img_color, 9, 9, 7)

		# upsample image to original size
		for _ in xrange(numDownSamples):
			img_color = cv2.pyrUp(img_color)
		# convert to grayscale and apply median blur
		img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
		img_blur = cv2.medianBlur(img_gray, 7)

		# detect and enhance edges
		img_edge = cv2.adaptiveThreshold(img_blur, 255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 9, 2)
		# -- STEP 5 --
		# convert back to color so that it can be bit-ANDed with color image
		img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
		final = cv2.bitwise_and(img_color, img_edge)
		return cv2.medianBlur(final,7)

class PencilSketch:
	'''
	PencilSketch
	We followed a similar approach to what we followed in the Water Color filter.
	But this time around after obtaining the major edges of the image, we 
	multiplied it with a pre built canvas sheet to give a pencil like image.	
	'''
	def __init__(self):
		pass
	
	def render(self,frame):
		canvas = cv2.imread("pen.jpg", cv2.CV_8UC1)
		numDownSamples = 2
		img_rgb = frame
		# number of downscaling steps
		numBilateralFilters = 3
		# number of bilateral filtering steps
		# -- STEP 1 --
		# downsample image using Gaussian pyramid
		img_color = img_rgb
		for _ in xrange(numDownSamples):
			img_color = cv2.pyrDown(img_color)
		# repeatedly apply small bilateral filter instead of applying
		# one large filter
		for _ in xrange(numBilateralFilters):
			img_color = cv2.bilateralFilter(img_color, 9, 9, 3)

		# upsample image to original size
		for _ in xrange(numDownSamples):
			img_color = cv2.pyrUp(img_color)
		# convert to grayscale and apply median blur
		img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
		img_blur = cv2.medianBlur(img_gray, 3)

		# detect and enhance edges
		img_edge = cv2.adaptiveThreshold(img_blur, 255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 9, 2)
		return  cv2.multiply(cv2.medianBlur(img_edge,7), canvas, scale=1./256)

class Mozaic:
	'''
	Mozaic
	We convert the frame to a grayscale and apply a binarisation thresholding operation.
	Gaussian blur operation is performd to smoothen out the image. We have an option to 
	invert the image at this point of time and reapply the gaussian filtering operation 
	to hava a varied texture. We eventually divide the inverted thresholded and blurred 
	image to obtain the mozaic effect. We cam impose it to a canvas so as to obtain the 
	sketched mozaic.
	'''
	def __init__(self):
		pass
		
	def render(self,frame):
		canvas = cv2.imread("pen.jpg", cv2.CV_8UC1)
		#convert frame to gray scale.
		img_gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		#perform binary threshold. With different values of threshold, we get different mozaic patterns 
		ret,img_thr=cv2.threshold(img_gray,70,255,cv2.THRESH_BINARY)
		#apply gaussian blur
		img_blur = cv2.GaussianBlur(img_thr, (3, 3), 0)
		#invert image
		img_invert= 255-img_blur
		img_blur=cv2.GaussianBlur(img_invert, ksize=(15, 15),sigmaX=0, sigmaY=0)
		#generate final mozaic effect
		final =255-cv2.divide(255-img_thr, 255-img_blur, scale=256)
		#render image over a canvas
		return cv2.multiply(final, canvas, scale=1./256)
	
class Marker:
	'''
	Marker
	We applied marker effect by means of first converting the frame to a gray scale. The we apply 
	Gaussian adative threshold operation over the image to locally binarize it. Then we simply 
	smooth out the image by means of median blur operation.
	'''
	def __init__(self):
		pass
	def render(self,frame):
		return cv2.medianBlur(cv2.adaptiveThreshold(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,27,3),5)
		

class Sketch:
	'''
	A very simple tactic. Run a canny edge detection algorithm and invert the image.
	'''
	def __init__(self):
		pass
		
	def render(self,frame):
		return 255-cv2.Canny((cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)),50,65)

class Tint:
	'''
	Tint
	Select the rgb value of a color tone and multiply it with he frame to create 
	a tinted effect into your image.
	'''
	def __init__(self):
		pass
	#create mask and render frame
	def render(self,frame,r,g,bl,s):
		if s == 0:
			temp=np.full((l,b,n),[255,255,255],dtype=np.int)
		else:
			temp=np.full((l,b,n),[bl,g,r],dtype=np.int)
		
		return np.multiply(frame,temp)


class WarmingFilter:
    """
	Warming filter
    A class that applies a warming filter to an image.
    The class uses curve filters to manipulate the perceived color
    temparature of an image. The warming filter will shift the image's
    color spectrum towards red, away from blue.
    """

    def __init__(self):
        # create look-up tables for increasing and decreasing a channel
        self.incr_ch_lut = self._create_LUT_8UC1([0, 64, 128, 192, 256],
                                                 [0, 70, 140, 210, 256])
        self.decr_ch_lut = self._create_LUT_8UC1([0, 64, 128, 192, 256],
                                                 [0, 30,  80, 120, 192])

    def render(self, img_rgb):
        # warming filter: increase red, decrease blue
        c_r, c_g, c_b = cv2.split(img_rgb)
        c_r = cv2.LUT(c_r, self.incr_ch_lut).astype(np.uint8)
        c_b = cv2.LUT(c_b, self.decr_ch_lut).astype(np.uint8)
        img_rgb = cv2.merge((c_r, c_g, c_b))

        # increase color saturation
        c_h, c_s, c_v = cv2.split(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV))
        c_s = cv2.LUT(c_s, self.incr_ch_lut).astype(np.uint8)

        return cv2.cvtColor(cv2.merge((c_h, c_s, c_v)), cv2.COLOR_HSV2RGB)

    def _create_LUT_8UC1(self, x, y):
        spl = UnivariateSpline(x, y)
        return spl(xrange(256))

class CoolingFilter:
    """
	Cooling filter
    A class that applies a cooling filter to an image.
    The class uses curve filters to manipulate the perceived color
    temparature of an image. The warming filter will shift the image's
    color spectrum towards blue, away from red.
    """

    def __init__(self):
		# create look-up tables for increasing and decreasing a channel
        self.incr_ch_lut = self._create_LUT_8UC1([0, 64, 128, 192, 256],
                                                 [0, 70, 140, 210, 256])
        self.decr_ch_lut = self._create_LUT_8UC1([0, 64, 128, 192, 256],
                                                 [0, 30,  80, 120, 192])

    def render(self, img_rgb):
        # cooling filter: increase blue, decrease red
        c_r, c_g, c_b = cv2.split(img_rgb)
        c_r = cv2.LUT(c_r, self.decr_ch_lut).astype(np.uint8)
        c_b = cv2.LUT(c_b, self.incr_ch_lut).astype(np.uint8)
        img_rgb = cv2.merge((c_r, c_g, c_b))

        # decrease color saturation
        c_h, c_s, c_v = cv2.split(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV))
        c_s = cv2.LUT(c_s, self.decr_ch_lut).astype(np.uint8)
        return cv2.cvtColor(cv2.merge((c_h, c_s, c_v)), cv2.COLOR_HSV2RGB)

    def _create_LUT_8UC1(self, x, y):
        spl = UnivariateSpline(x, y)
        return spl(xrange(256))

		
if __name__=="__main__":
	
	h1=Marker()
	h2=Mozaic()
	h3=WaterColor()
	h4=WarmingFilter()
	h5=CoolingFilter()
	h6=Tint()
	h7=Sketch()
	h8=Negative()
	h9=Hotshot()
	h10=PencilSketch()
	while(True):
		ret,frame=cap.read()
		r = cv2.getTrackbarPos('R','Tint')
		g = cv2.getTrackbarPos('G','Tint')
		bl = cv2.getTrackbarPos('B','Tint')
		s = cv2.getTrackbarPos(switch,'Tint')

		cv2.imshow("Marker",h1.render(frame))
		cv2.imshow("Mozaic",h2.render(frame))
		cv2.imshow("WaterColor",h3.render(frame))
		cv2.imshow('Warm',h4.render(frame))
		cv2.imshow('Cool',h5.render(frame))
		cv2.imshow("Tint",h6.render(frame,r,g,bl,s))
		cv2.imshow("Skecth",h7.render(frame))
		cv2.imshow("Negative",h8.render(frame))
		cv2.imshow("Hotshot",h9.render(frame))
		cv2.imshow("PencilSketch",h10.render(frame))
		if (cv2.waitKey(1) & 0xFF) == 27:
			break
		
	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()