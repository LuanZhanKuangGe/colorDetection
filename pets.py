import cv2

cascade = cv2.CascadeClassifier('haarcascade_frontal_dog_face.xml')

min = 8.0
max = 10

img = cv2.imread('dog.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = cascade.detectMultiScale(img_gray, min, max)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
    roi_gray = img_gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]

cv2.imshow('dog', img)

img = cv2.imread('cat.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = cascade.detectMultiScale(img_gray, min, max)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
    roi_gray = img_gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]

cv2.imshow('cat', img)


cv2.waitKey(0)
cv2.destroyAllWindows()