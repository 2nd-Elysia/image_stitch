import cv2
from utils import *
import numpy as np
def validate_translation_sift(img1, img2):
    # 读取两张图像
    image1 = img1
    image2 = img2

    # 使用SIFT特征检测器
    sift = cv2.SIFT_create()

    # 寻找关键点和描述符
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    # 使用FLANN匹配器
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # 应用比率测试，保留最佳匹配
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # 获取关键点的坐标
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 计算仿射变换矩阵
    M, _ = cv2.estimateAffinePartial2D(points1, points2)

    # 判断是否为平移变换
    if M is not None:
        return True
    else:
        return False

    # # 可视化匹配结果
    # img_matches = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # plt.imshow(img_matches)
    # img = Image.fromarray(img_matches)
    # img.show()

    # plt.show()

def test_crop_black_borders():
    # 假设在 'tests' 目录中有一张测试图像
    image = cv2.imread(r'images\1123_878_631_2.png')
    cropped_image = crop_black_borders(image)
    # 根据预期添加断言
    assert cropped_image.shape == (3036, 5106, 3)
    
def test_stitch():
    image1 = cv2.imread(r'images\222.png')
    image2 = cv2.imread(r'images\2.png')
    image3 = cv2.imread(r'image_user\u1.jpg')
    stitcher = Stitcher()
    see = True
    result,img = stitcher.stitch([image1, image2], showMatches=see)
    assert result.shape == (3039, 4231, 3)
    assert validate_translation_sift(result, image2) == True
    assert validate_translation_sift(result, image3) == False
