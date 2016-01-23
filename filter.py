import cv2
import numpy as np
def nothing(x):
    pass
	
cv2.namedWindow('Filter')

# create trackbars for color change
cv2.createTrackbar('R','Filter',0,255,nothing)
cv2.createTrackbar('G','Filter',0,255,nothing)
cv2.createTrackbar('B','Filter',0,255,nothing)

# create switch for ON/OFF functionality
switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch, 'Filter',0,1,nothing)

cap=cv2.VideoCapture(0)
l,b,n=cap.read()[1].shape
	
cv2.namedWindow("Image")
while(True):
	#pick BGR values
	r = cv2.getTrackbarPos('R','Filter')
	g = cv2.getTrackbarPos('G','Filter')
	bl = cv2.getTrackbarPos('B','Filter')
	s = cv2.getTrackbarPos(switch,'Filter')
	
	#create filter matrix
	if s == 0:
		temp1=np.full((l,b,n),[255,255,255],dtype=np.uint16)
	else:
		temp1=np.full((l,b,n),[bl,g,r],dtype=np.uint16)
	cv2.imshow('Filter',temp1)
	
	ret, frame = cap.read()
	#multiply original Image and filter
	f1=np.multiply(frame,temp1)
	#display image and filter
	cv2.imshow('Image',frame)
	cv2.imshow('Filter',f1)
	
	if (cv2.waitKey(1) & 0xFF) == 27:
		break
		
cv2.imwrite("Output_file.tiff",f1)
# When everything done, release the capture
cap.release()

cv2.destroyAllWindows()