import cv2 
import matplotlib.pyplot as plt
import numpy as np
def convolution(input,kernelShape):
  weight,height = input.shape
  sobelRow = np.zeros((weight-kernelShape+1,height-kernelShape+1))
  sobelCol = np.zeros((weight-kernelShape+1,height-kernelShape+1))
  sobelFilter = np.zeros((weight-kernelShape+1,height-kernelShape+1))
  print(sobelRow.shape)
  print(sobelCol.shape)
  # Bo loc Sobel theo chieu ngang
  kernelRow = np.array([[-1,-2,-1],
                         [0,0,0],
                         [1,2,1]])
  kernelRow = np.multiply(1/2,kernelRow) 
  # Bo loc Sobel theo chieu doc
  kernelCol = np.array([[1, 0, -1], 
                         [2, 0, -2], 
                         [1, 0, -1]])
  kernelCol = np.multiply(1/2,kernelCol)
  for row in range(0, height-kernelShape+1):
    for col in range(0, weight-kernelShape+1 ):
      sobelRow[row,col]= np.sum(input[row:row+kernelShape,col:col+kernelShape]*kernelRow)
      sobelCol[row,col]= np.sum(input[row:row+kernelShape,col:col+kernelShape]*kernelCol)
      sobelFilter[row,col] = np.sqrt(sobelRow[row,col]**2+sobelCol[row,col]**2)
 
  return sobelRow,sobelCol,sobelFilter
if __name__ == "__main__":
  img = cv2.imread("final2.jpg")
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img = cv2.resize(img,(700,700))
  print(img.shape)
  sobel_Row, sobel_Col, sobel_Filter = convolution(img,3)
  cv2.imwrite("sobelRow.jpg",sobel_Row)
  cv2.imwrite("sobelCol.jpg",sobel_Col)
  cv2.imwrite("SobelFilter.jpg",sobel_Filter)
