import numpy as np
import cv2

def crop_black_borders(image):
    if len(image.shape) == 3 and image.shape[2] == 3:  # 彩色图像
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 2:  # 灰度图像
        gray = image
    else:
        raise ValueError("Unsupported image format")
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        return image[y:y+h, x:x+w]
    return image

def crop_black_borders2(image):
    if len(image.shape) == 3 and image.shape[2] == 3:  # 彩色图像
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 2:  # 灰度图像
        gray = image
    else:
        raise ValueError("Unsupported image format")
    # 调整二值化阈值
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    # 使用形态学操作
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # 寻找轮廓
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        return image[y:y+h, x:x+w]
    return image

class Stitcher:
	def stitch(self, images, showMatches=False):
		imageB, imageA = images
		keypoints1, featuresA = self.extract_features(imageA)
		keypoints2, featuresB = self.extract_features(imageB)
		good_matches = self.match_features_bf(featuresA, featuresB)

		if not good_matches:
			raise ValueError("No good matches found.")
		# match features between the two images
		M = self.estimate_transform(keypoints1, keypoints2, good_matches)
		if M is None:
			return None, None
		# 图像拼接
		result = cv2.warpPerspective(imageA, M, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]+imageB.shape[0]))
		result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
		result = crop_black_borders2(result)
		if showMatches:
			img6 = cv2.drawMatchesKnn(imageB,keypoints2,imageA,keypoints1, [good_matches] ,None,flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS | cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
		else :
			img6 = None
		return result ,img6

	def extract_features(self, image):
		# convert the image to grayscale
		feature_extractor = cv2.SIFT_create()
		keypoints, descriptors = feature_extractor.detectAndCompute(image, None)
		return keypoints, descriptors

	def match_features_bf(self,descriptors1, descriptors2):
		bf = cv2.BFMatcher()
		matches = bf.knnMatch(descriptors1, descriptors2, k=2)
		good_matches = []
		for m, n in matches:
			if m.distance < 0.5 * n.distance:
				good_matches.append(m)
		return good_matches

	def estimate_transform(self, keypoints1, keypoints2, good_matches):
		src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
		dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
		M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
		return M