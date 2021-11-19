import cv2
import numpy as np

L = 0
H = 255


def func(gray2, img2):
    global L, H

    # 将灰度图转行成二值图
    ret, binary = cv2.threshold(gray, L, H, cv2.THRESH_BINARY)

    # 二值图黑白调换
    binary = cv2.bitwise_not(binary, binary)
    # cv2.imshow("show", binary)

    # 寻找图中轮廓
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    # 用红线画出全部轮廓，可以去掉
    cv2.drawContours(img2, contours, -1, (0, 0, 255), 2)

    if not len(contours):
        return img2

    # 计算全部轮廓的凸包，寻找面积最大的一个
    target = 0
    max = 0
    for i in range(len(contours)):
        hull = cv2.convexHull(contours[i])
        area = cv2.contourArea(hull)
        if area > max:
            max = area
            target = i

    # 用绿线画出最大的凸包
    hull = cv2.convexHull(contours[target])
    for i in range(len(hull)):
        cv2.line(
            img2, tuple(hull[i][0]), tuple(hull[(i + 1) % len(hull)][0]), (0, 255, 0), 2
        )

    # 用蓝线画出凸包的外接矩形，也就是目标
    x, y, w, h = cv2.boundingRect(hull)
    cv2.rectangle(img2, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return img2


def LL(x):
    global L
    L = x


def HH(x):
    global H
    H = x


cv2.namedWindow("show")
cv2.createTrackbar("low", "show", 0, 255, LL)
cv2.createTrackbar("high", "show", 255, 255, HH)

camera = True # 切换图片和视频

if camera:
    vid = cv2.VideoCapture(0)

while 1:

    if camera:
        ret, img = vid.read()
    else:
        img = cv2.imread("test.jpg")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = func(gray, img)

    cv2.imshow("show", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

if camera:
    vid.release()

cv2.destroyAllWindows()
