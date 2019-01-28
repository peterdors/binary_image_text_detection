# Peter Dorsaneo
# A text detection program in Python using OpenCV API. 
# USAGE EXAMPLE: python text_detection.py --image images/image01.jpg
# 
# import the necessary packages
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import time
import pytesseract
import cv2
import sys

def create_argument_parser():
	# Construct the argument parser and parse the arguments.
	# Only have the image argument parser for now. 
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", type=str, help="path to input image")
	# ap.add_argument("-east", "--east", type=str,
	# 	help="path to input EAST text detector")
	# ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
	# 	help="minimum probability required to inspect a region")
	# ap.add_argument("-w", "--width", type=int, default=320,
	# 	help="resized image width (should be multiple of 32)")
	# ap.add_argument("-e", "--height", type=int, default=320,
	# 	help="resized image height (should be multiple of 32)")
	args = vars(ap.parse_args())
	return args

def read_image(image):
	return cv2.imread(image)

def grayscale_image(image):
	return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
def show_image(image):
	cv2.imshow("image", image)

def median_blur_image(image, k_size):
	return cv2.medianBlur(image, k_size)

def resize_image(image, width, height):
	return cv2.resize(image, (width, height))
	
def run_tesseract(image):
	# Define config parameters.
	# '-l eng'  for using the English language
	# '--oem 1' for using LSTM OCR Engine
	config = ('-l eng --oem 1 --psm 3')
	# Returns the string of text recognized in the image. 
	# Can use the below return call. 
	# return pytesseract.image_to_string(image, config=config)
	# Or likewise, below.
	return pytesseract.image_to_string(image)

if __name__ == '__main__':

	if (len(sys.argv) < 3):
		print("Wrong args")

	args = create_argument_parser()

	img = read_image(args["image"])

	img = grayscale_image(img)

	img = median_blur_image(img, 5)

	show_image(img)

	print(run_tesseract(img))
	print(img.shape)

	# Waits forever for user to press any key
	cv2.waitKey(0)
	cv2.destroyAllWindows()


	


