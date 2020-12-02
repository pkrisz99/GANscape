from dataread import DataRead
import copy
import cv2

i = (0,100)

validData = DataRead(r'C:\\places\\','train', 128, 10, shuffle = False)
cv2.imshow('target',validData.target_images[i[0],i[1]].astype('uint8'))



own_cropped = copy.deepcopy(validData.target_images[i[0],i[1]])
p = validData.csv[i[0],i[1]]
own_cropped[p[4]:p[6]+1, p[5]:p[7]+1] = [0,0,0]
cv2.imshow('cropped with csv params',own_cropped.astype('uint8'))
cv2.imshow('validData cropped',validData.cropped_images[i[0],i[1]].astype('uint8'))
print("csv params:", p)
cv2.waitKey(0)
cv2.destroyAllWindows()



