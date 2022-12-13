import cv2
import numpy as np
from matplotlib import pyplot as plt

def tich_chap(img,mask):
    img_new = np.zeros([row,col])
    for i in range(1, row-1):
        for j in range(1,col-1):
            temp = (img[i-1,j-1]*mask[0,0]\
                + img[i-1,j]*mask[0,1]\
                + img[i-1,j+1]*mask[0,2]\
                + img[i,j-1]*mask[1,0]\
                + img[i,j]*mask[1,1]\
                + img[i,j+1]*mask[1,2]\
                + img[i+1,j-1]*mask[2,0]\
                + img[i+1,j]*mask[2,1]\
                + img[i+1,j+1]*mask[2,2])
            img_new[i,j]= temp
            
    img_new = img_new.astype(np.uint8)
    return img_new

locTB3x3 = np.array (([1/9,1/9,1/9],
                      [1/9,1/9,1/9],
                      [1/9,1/9,1/9]),dtype="float")
def loc_trung_vi(img):
    for i in range (1,row-1):
        for j in range(1,col-1):
            temp = [img[i-1,j-1],
                    img[i-1,j],
                    img[i-1,j+1],
                    img[i,j-1],
                    img[i,j],
                    img[i,j+1],
                    img[i+1,j-1],
                    img[i+1,j],
                    img[i+1,j+1]]
            temp.sort()
            img_new[i,j]=temp[4]
    return img_new

if __name__ == "__main__":
    img = plt.imread('final1.bmp')
    # img=img/255
    row,col = img.shape
    global img_new
    img_new = np.zeros([row,col])
    
    fig = plt.figure(figsize=(16,9))
    (img_goc,img_tb,img_tv) = fig.subplots(1,3)
    #Hien thi anh goc
    img_goc.imshow(img, cmap="gray")
    img_goc.set_title("Anh Goc")
#loc trung vi
    img_TrungVi = loc_trung_vi(img)
    img_tv.imshow(img_TrungVi,cmap = 'gray')
    cv2.imwrite("AnhLocTrungVi.png",img_TrungVi)
    img_tv.set_title("Anh loc trung vi")
    
    #LocTB3x3
    imgTB_3x3 = tich_chap(img,locTB3x3)
    img_tb.imshow(imgTB_3x3,cmap='gray')
    cv2.imwrite("AnhLocTrungBinh.png",imgTB_3x3)
    img_tb.set_title("Anh loc trung binh")
plt.show()