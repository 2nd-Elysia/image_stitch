import numpy as np
import imutils
import cv2

def crop_black_borders(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        return image[y:y+h, x:x+w]
    return image

class Stitcher:
	def stitch(self, images,
		showMatches=False):
		# unpack the images, then detect keypoints and extract
		# local invariant descriptors from them
		(imageB, imageA) = images
		(kpsA, featuresA) = self.detectAndDescribe(imageA)
		(kpsB, featuresB) = self.detectAndDescribe(imageB)
		# match features between the two images
		M = self.matchKeypoints(kpsA, kpsB,
			featuresA, featuresB)
		# if the match is None, then there aren't enough matched
		# keypoints to create a panorama
		if M is None:
			return None
		# otherwise, apply a perspective warp to stitch the images
		# together
		# (matches, H, status) = M
		# result = cv2.warpPerspective(imageA, M,
		# 	(imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
		# result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
		# 图像拼接
		result = cv2.warpPerspective(imageA, M, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
		# 检查透视变换结果的尺寸是否正确
		if result.shape[1] < imageB.shape[1]:
			result = cv2.resize(result, (imageB.shape[1], result.shape[0]))
		# 图像拼接
		result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
		# check to see if the keypoint matches should be visualized
		# if showMatches:
		# 	vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches,
		# 		status)
		# 	# return a tuple of the stitched image and the
		# 	# visualization
		# 	return (result, vis)
		# # return the stitched image
		return result

	def detectAndDescribe(self, image):
		# convert the image to grayscale
		feature_extractor = cv2.SIFT_create()
		keypoints, descriptors = feature_extractor.detectAndCompute(image, None)
		return keypoints, descriptors

	def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
		):
		# compute the raw matches and initialize the list of actual
		# matches
		# matcher = cv2.BFMatcher()
		# rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
		# matches = []
		bf = cv2.BFMatcher()
		matches = bf.knnMatch(featuresA, featuresB, k=2)
		good_matches = []
		for m, n in matches:
			if m.distance < 0.5 * n.distance:
				good_matches.append(m)
	
		src_pts = np.float32([kpsA[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
		dst_pts = np.float32([kpsB[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
		# 使用RANSAC算法估计仿射变换矩阵
		M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
		return M
		# loop over the raw matches
		# for m in good_matches:
		# 	# ensure the distance is within a certain ratio of each
		# 	# other (i.e. Lowe's ratio test)
		# 	if len(m) == 2 and m[0].distance < m[1].distance * ratio:
		# 		matches.append((m[0].trainIdx, m[0].queryIdx))

		# # computing a homography requires at least 4 matches
		# if len(good_matches) > 2:
		# 	# construct the two sets of points
		# 	ptsA = np.float32([kpsA[i] for (_, i) in matches])
		# 	ptsB = np.float32([kpsB[i] for (i, _) in matches])

		# 	# compute the homography between the two sets of points
		# 	(H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
		# 		reprojThresh)

		# 	# return the matches along with the homograpy matrix
		# 	# and status of each matched point
		# 	return (matches, H, status)

		# # otherwise, no homograpy could be computed
		# return None

	def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
		# initialize the output visualization image
		(hA, wA) = imageA.shape[:2]
		(hB, wB) = imageB.shape[:2]
		vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
		vis[0:hA, 0:wA] = imageA
		vis[0:hB, wA:] = imageB
		# loop over the matches
		for ((trainIdx, queryIdx), s) in zip(matches, status):
			# only process the match if the keypoint was successfully
			# matched
			if s == 1:
				# draw the match
				ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
				ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
				cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
		# return the visualization
		return vis