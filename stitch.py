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

# stitch the images together to create a panorama
stitcher = Stitcher()
(result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)
# show the images
# cv2.imshow("Image A", imageA)
# pillow_image = Image.fromarray(cv2.cvtColor(imageA, cv2.COLOR_BGR2RGB))
# pillow_image.show()
# # cv2.imshow("Image B", imageB)
# pillow_image = Image.fromarray(cv2.cvtColor(imageB, cv2.COLOR_BGR2RGB))
# pillow_image.show()
# # cv2.imshow("Keypoint Matches", vis)
# pillow_image = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
# pillow_image.show()
# cv2.imshow("Result", result)
pillow_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
pillow_image.show()
pillow_image.save(f"images/{A_name}_{B_name}.png")
# cv2.waitKey(0)
# USAGE
# python stitch.py -f image_test/a.png -s image_test/b.png 