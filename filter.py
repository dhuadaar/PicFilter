import cv2
import numpy as np

def nothing(x):
    pass
	
cv2.namedWindow('image')

# create trackbars for color change
cv2.createTrackbar('R','image',0,255,nothing)
cv2.createTrackbar('G','image',0,255,nothing)
cv2.createTrackbar('B','image',0,255,nothing)

# create switch for ON/OFF functionality
switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch, 'image',0,1,nothing)
	
cap=cv2.VideoCapture(0)
while(True):
	ret, frame = cap.read()
	if(ret):
		l,b,n=frame.shape
		img = np.zeros([l,b,n], np.uint8)
	
	
	# get current positions of four trackbars
	r = cv2.getTrackbarPos('R','image')
	g = cv2.getTrackbarPos('G','image')
	b = cv2.getTrackbarPos('B','image')
	s = cv2.getTrackbarPos(switch,'image')
	cv2.imshow('image',img)
	if s == 0:
		img[:] = 0
	else:
		img[:] = [b,g,r]
	
	cv2.imshow('frame',frame-img)
	if (cv2.waitKey(1) & 0xFF) == 27:
		break
		
cv2.imwrite("Output_file.jpg",frame)
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()