import cv2

#-----Reading the image-----------------------------------------------------
img = cv2.imread('The_River_10.jpg', 1)
cv2.imshow("img",img)
b = 20
imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

imghsv[:,:,2] = [[max(pixel - b, 0) if pixel < 190 else min(pixel + b, 255) for pixel in row] for row in imghsv[:,:,2]]
vis = cv2.imshow('contrast', cv2.cvtColor(imghsv, cv2.COLOR_HSV2BGR))
cv2.imwrite("output.jpg", cv2.cvtColor(imghsv, cv2.COLOR_HSV2BGR))
cv2.waitKey(0)
