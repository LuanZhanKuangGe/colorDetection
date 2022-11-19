import numpy as np
import cv2

def getShape(contour):
    # cv2.approxPloyDP() function to approximate the shape
    approx = cv2.approxPolyDP(
        contour, 0.01 * cv2.arcLength(contour, True), True)

    # finding center point of shape
    M = cv2.moments(contour)
    if M['m00'] != 0.0:
        x = int(M['m10'] / M['m00'])
        y = int(M['m01'] / M['m00'])

    # putting shape name at center of each shape
    if len(approx) == 3:
        return 'Triangle', x, y

    elif len(approx) == 4:
        return 'Quadrilateral', x, y

    elif len(approx) == 5:
        return 'Pentagon', x, y

    elif len(approx) == 6:
        return 'Hexagon', x, y

    else:
        return 'circle', x, y

font = cv2.FONT_HERSHEY_SIMPLEX

lower_green = np.array([35, 43, 46])  # 绿色低阈值
upper_green = np.array([77, 255, 255])  # 绿色高阈值
lower_red = np.array([0, 127, 128])  # 红色低阈值
upper_red = np.array([10, 255, 255])  # 红色高阈值
lower_blue = np.array([100, 43, 46])  # 蓝色低阈值
upper_blue = np.array([124, 255, 255])  # 蓝色高阈值
lower_yellow = np.array([11, 43, 46])  # 黄色低阈值
upper_yellow = np.array([34, 255, 255])  # 黄色高阈值

frame = cv2.imread('rect.png')

hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask_green = cv2.inRange(hsv_img, lower_green, upper_green)  # 根据颜色范围删选
mask_red = cv2.inRange(hsv_img, lower_red, upper_red)
mask_blue = cv2.inRange(hsv_img, lower_blue, upper_blue)
mask_yellow = cv2.inRange(hsv_img, lower_yellow, upper_yellow)

mask_green = cv2.medianBlur(mask_green, 7)  # 中值滤波
mask_red = cv2.medianBlur(mask_red, 7)
mask_blue = cv2.medianBlur(mask_blue, 7)
mask_yellow = cv2.medianBlur(mask_yellow, 7)

mask = cv2.bitwise_or(mask_green, mask_red)
mask = cv2.bitwise_or(mask, mask_blue)
mask = cv2.bitwise_or(mask, mask_yellow)

cv2.imwrite('mask.png', mask)

contours, hierarchy = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours2, hierarchy2 = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours3, hierarchy3 = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours4, hierarchy4 = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

for cnt in contours:
    cv2.drawContours(frame, [cnt], 0, (255, 255, 255), 5)
    shape, x, y = getShape(cnt)
    cv2.putText(frame, "Green " + shape, (x, y - 5), font, 0.7, (255, 0, 255), 2)

for cnt2 in contours2:
    cv2.drawContours(frame, [cnt2], 0, (255, 255, 255), 5)
    shape, x2, y2 = getShape(cnt2)
    cv2.putText(frame, "Red " + shape, (x2, y2 - 5), font, 0.7, (255, 0, 255), 2)

for cnt3 in contours3:
    cv2.drawContours(frame, [cnt3], 0, (255, 255, 255), 5)
    shape, x3, y3 = getShape(cnt3)
    cv2.putText(frame, "Blue " + shape, (x3, y3 - 5), font, 0.7, (255, 0, 255), 2)

for cnt4 in contours4:
    cv2.drawContours(frame, [cnt4], 0, (255, 255, 255), 5)
    shape, x4, y4 = getShape(cnt4)
    cv2.putText(frame, "Yellow " + shape, (x4, y4 - 5), font, 0.7, (255, 0, 255), 2)

cv2.imshow("dection", frame)

cv2.waitKey(0)
cv2.destroyAllWindows()
