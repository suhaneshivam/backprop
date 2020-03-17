from skimage.exposure import rescale_intensity
import numpy as np
import cv2
import argparse


def conv(image,kernal):
	iH,iW=image.shape[:2]
	kH,kW=kernal.shape[:2]
	
	pad=(kH-1) // 2
	image=cv2.copyMakeBorder(image,pad,pad,pad,pad,cv2.BORDER_REPLICATE)
	out=np.zeros((iH,iW),dtype="float")
	
	for y in np.arange(pad,iH+pad):
		for x in np.arange(pad,iW+pad):
			
			window=image[y-pad:y+pad+1,x-pad:x+pad+1]
			
			k=(window*kernal).sum()
			
			out[(y-pad),(x-pad)]=k
	out=rescale_intensity(out, in_range=(0,255))
	out=(out*255).astype("uint8")
	return out
ap=argparse.ArgumentParser()
ap.add_argument("-i","--input",required=True,help="path to input")
args=vars(ap.parse_args())

smallblur=np.ones((7,7),dtype="float")*(1/(7*7))

largeblur=np.ones((7,7),dtype="float")*(1/(21*21))

sharpen=np.array(([0,-1,0],
  		  [-1,5,-1],
  		  [0,-1,0]),dtype="int")

laplacian=np.array(([0,1,0],
		    [1,-4,1],
		    [0,1,0]),dtype="int")

sobelX=np.array((
		[-1,0,1],
		[0,0,0],
		[-2,0,2]), dtype="int")		
							   
sobelY=np.array((
		[-1,0,-2],
		[0,0,0],
		[1,0,2]), dtype="int")

emboss=np.array((
		[-2,-1,0],
		[-1,0,1],
		[0,1,2]),dtype="int")
								
kernalbank=(("smallblur",smallblur),
            ("largeblur",largeblur),                                      
            ("sharpen",sharpen),                                        
            ("laplacian",laplacian),                                     
            ("sobelX",sobelX),                                            
            ("sobelY",sobelY),
	    ("emboss",emboss))

image=cv2.imread(args["input"])
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

for (kernalname,k) in kernalbank:
	print("[INFO] applying {} kernal".format(kernalname))
	convolveout=conv(gray,k)
	opencvout=cv2.filter2D(gray,-1,k)
	
	cv2.imshow("original",gray)
	cv2.imshow("{}opencv".format(kernalname),opencvout)
	cv2.imshow("{}convolution".format(kernalname),convolveout)
	cv2.waitkey(0)
        cv2.destroyAllWindows()
