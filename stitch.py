from utils import Stitcher
import argparse
import imutils
import cv2
from PIL import Image
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True,
	help="path to the first image")
ap.add_argument("-s", "--second", required=True,
	help="path to the second image")
args = vars(ap.parse_args())

path_a = args["first"]
path_b = args["second"]
imageA = cv2.imread(args["first"])
imageB = cv2.imread(args["second"])
A_name = os.path.splitext(os.path.basename(path_a))[0]
B_name = os.path.splitext(os.path.basename(path_b))[0]
f_name = os.path.splitext(os.path.dirname(path_a))[0]

# stitch the images together to create a panorama
stitcher = Stitcher()

# whether see the good_matches or not
see = True

result,img = stitcher.stitch([imageA, imageB], showMatches=see)
# show the images
if see:
	pillow_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
	pillow_image.show()
# cv2.imshow("Result", result)
pillow_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
pillow_image.show()
pillow_image.save(f"{f_name}/{A_name}_{B_name}.png")


# USAGE
# python stitch.py -f image_test/a.png -s image_test/b.png 
# python stitch.py -f images/631.png -s images/2.png 
# python stitch.py -f images/878.png -s images/631_2.png
# python stitch.py -f images/1123.png -s images/878_631_2.png
# python stitch.py -f images/1330.png -s images/1123_878_631_2.png
# python stitch.py -f image/3.png -s image/4.png
# python stitch.py -f image_user/1.jpg -s image_user/2.jpg