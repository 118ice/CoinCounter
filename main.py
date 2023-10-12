import numpy as np
import cv2
import os
import sys
import argparse
import math


parser = argparse.ArgumentParser(description='Convert RGB to GRAY')
parser.add_argument('-name', '-n', type=str, default='coins1.png')
args = parser.parse_args()

def normalise(array):
    #print(np.min(array))
    #print(np.max(array))
    return 255 *  (array - np.min(array)) / (np.max(array) - np.min(array))

def Sobel(img):
    height = img.shape[0] 
    width = img.shape[1] 
    kernelSize = 3
    outputDX = np.zeros_like(image)
    outputDY = np.zeros_like(image)
    kernelDX = [[-1,0,1],[-1,0,1],[-1,0,1]]
    kernelDY = [[-1,-1,-1],[0,0,0],[1,1,1]]
    for y in range(height-2):
        for x in range(width-2):
            if x >= kernelSize//2 and y >= kernelSize//2 and x < width-kernelSize//2 and y < height-kernelSize//2:
                newPixelDX = 0
                newPixelDY = 0
                for p in range (kernelSize):
                    for l in range(kernelSize):
                        newPixelDX += (kernelDX[p][l]*image[(x-kernelSize//2)+p][(y-kernelSize//2)+l])
                        newPixelDY += (kernelDY[p][l]*image[(x-kernelSize//2)+p][(y-kernelSize//2)+l])
                outputDX[x][y] = newPixelDX
                outputDY[x][y] = newPixelDY
    gradImage = np.sqrt(outputDX*outputDX+outputDY*outputDY)
    #toTan = outputDY/outputDX
    #print(np.min(toTan))
    psi = np.arctan2(outputDX,(outputDY)+(math.e**-10))
    psi = normalise(psi)
    outputDX = normalise(outputDX)
    outputDY = normalise(outputDY)
    return outputDX,outputDY,gradImage,psi



def GaussianBlur(input, size):

	# intialise the output using the input
	blurredOutput = np.zeros([input.shape[0], input.shape[1]], dtype=np.float32)
	# create the Gaussian kernel in 1D 
	kX = cv2.getGaussianKernel(size,1)
	kY = cv2.getGaussianKernel(size,1)
	# make it 2D multiply one by the transpose of the other
	kernel = kX * kY.T
	
	# CREATING A DIFFERENT IMAGE kernel WILL BE NEEDED
	# TO PERFORM OPERATIONS OTHER THAN GUASSIAN BLUR!!!
	
	# we need to create a padded version of the input
	# or there will be border effects
	kernelRadiusX = round(( kernel.shape[0] - 1 ) / 2)
	kernelRadiusY = round(( kernel.shape[1] - 1 ) / 2)
	
	paddedInput = cv2.copyMakeBorder(input, 
		kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY, 
		cv2.BORDER_REPLICATE)

	# now we can do the convoltion
	for i in range(0, input.shape[0]):	
		for j in range(0, input.shape[1]):
			patch = paddedInput[i:i+kernel.shape[0], j:j+kernel.shape[1]]
			sum = (np.multiply(patch, kernel)).sum()
			blurredOutput[i, j] = sum

	return blurredOutput

# ==== MAIN ==============================================
imageName = args.name

# ignore if no such file is present.
if not os.path.isfile(imageName):
    print('No such file')
    sys.exit(1)

# Read image from file
image = cv2.imread(imageName, 1)

# ignore if image is not array.
if not (type(image) is np.ndarray):
    print('Not image data')
    sys.exit(1)

# CONVERT COLOUR, BLUR AND SAVE
gray_image = cv2.cvtColor( image, cv2.COLOR_BGR2GRAY )
gray_image = gray_image.astype(np.float32)

outputDX,outputDY,gradImage,psi = Sobel(gray_image)
cv2.imwrite( "outputDX.jpg", outputDX )
cv2.imwrite( "outputDY.jpg", outputDY )
cv2.imwrite( "gradImage.jpg", gradImage )
cv2.imwrite( "psi.jpg", psi )


# apply Gaussian blur
coinBlurred = GaussianBlur(gray_image,23)
# save image
cv2.imwrite( "blur.jpg", coinBlurred )