import cv2
import numpy as np

def canny(image):
    gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blur, 50, 150) #Canny(image, low_threshold, high_threshold) , the ratio should be 1:2 or 1:3
    return canny

def region_of_interest(image):
    height = image.shape[0]
    triangle = np.array([[(200,height),(1100, height),(550, 250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, triangle, 255)
    return mask

image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)
canny_img = canny(lane_image)

cv2.imshow("result",region_of_interest(canny_img))
cv2.waitKey(0)

