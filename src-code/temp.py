import easyocr
import cv2
from matplotlib import pyplot as plt
import numpy as np

IMAGE_PATH = '/Users/hungnguyen/TaiLieu/NhanDang/model/Screenshot 2024-05-18 at 1.09.58 AM.png'
reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory
result = reader.readtext('/Users/hungnguyen/TaiLieu/NhanDang/model/Screenshot 2024-05-18 at 1.09.58 AM.png')

top_left = tuple(result[0][0][0])
bottom_right = tuple(result[0][0][2])
text = result[0][1]
font = cv2.FONT_HERSHEY_SIMPLEX

# img = cv2.imread(IMAGE_PATH)
# img = cv2.rectangle(img,top_left,bottom_right,(0,255,0),3)
# img = cv2.putText(img,text,top_left, font, 0.5,(255,255,255),2,cv2.LINE_AA)

img = cv2.imread(IMAGE_PATH)
spacer = 100
for detection in result: 
    top_left = tuple(detection[0][0])
    bottom_right = tuple(detection[0][2])
    text = detection[1]
    img = cv2.rectangle(img,top_left,bottom_right,(0,255,0),3)
    img = cv2.putText(img,text,(20,spacer), font, 0.5,(0,255,0),2,cv2.LINE_AA)
    spacer+=15
    
plt.imshow(img)
plt.show()
