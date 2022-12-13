import cv2 
import matplotlib.pyplot as plt
import numpy as np
def GaussianBlur(image):
    image = cv2.GaussianBlur(image, (5, 5), 0)
    return image
   
def convolution(input,kernelShape):
  input = GaussianBlur(input)
  weight,height = input.shape
  sobelRow = np.zeros((weight-kernelShape+1,height-kernelShape+1))
  sobelCol = np.zeros((weight-kernelShape+1,height-kernelShape+1))
  SobelFilter_image = np.zeros((weight-kernelShape+1,height-kernelShape+1))
  print(sobelRow.shape)
  print(sobelCol.shape)
 
  # Bo loc Sobel theo chieu ngang
  kernelRow = np.array([[-1,-2,-1],
                     [0,0,0],
                     [1,2,1]])
  kernelRow = np.multiply(1/2,kernelRow) 
  # Bo loc Sobel theo chieu doc
  kernelCol = np.array([[-1, 0, 1], 
                 [-2, 0, 2], 
                 [-1, 0, 1]])
  kernelCol = np.multiply(1/2,kernelCol)
  for row in range(0, height-kernelShape+1):
    for col in range(0, weight-kernelShape+1 ):
      sobelRow[row,col]= np.sum(input[row:row+kernelShape,col:col+kernelShape]*kernelRow)
      sobelCol[row,col]= np.sum(input[row:row+kernelShape,col:col+kernelShape]*kernelCol)
      SobelFilter_image[row,col] = np.sqrt(sobelRow[row,col]**2+sobelCol[row,col]**2)
  SobelFilter_image=SobelFilter_image.astype('uint8')
  angles = np.rad2deg(np.arctan2( sobelRow,sobelCol))
  angles[angles < 0] += 180
 
  return sobelRow, sobelCol,SobelFilter_image, angles
 
 
def nonMaximumSuppression(image, angle):
    size = image.shape
    suppressed = np.zeros(size)
    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1):
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                value_to_compare = max(image[i, j - 1], image[i, j + 1])
            elif (22.5 <= angle[i, j] < 67.5):
                value_to_compare = max(image[i - 1, j - 1], image[i + 1, j + 1])
            elif (67.5 <= angle[i, j] < 112.5):
                value_to_compare = max(image[i - 1, j], image[i + 1, j])
            else:
                value_to_compare = max(image[i + 1, j - 1], image[i - 1, j + 1])
            
            if image[i, j] >= value_to_compare:
                suppressed[i, j] = image[i, j]
    suppressed = np.multiply(suppressed, 255.0 / suppressed.max())
    return suppressed
 
    
def double_threshold_hysteresis(image, low, high):
    weak = 50
    strong = 255
    size = image.shape
    result = np.zeros(size)
    weak_x, weak_y = np.where((image > low) & (image <= high))
    strong_x, strong_y = np.where(image >= high)
    result[strong_x, strong_y] = strong
    result[weak_x, weak_y] = weak
    dx = np.array((-1, -1, 0, 1, 1, 1, 0, -1))
    dy = np.array((0, 1, 1, 1, 0, -1, -1, -1))
    size = image.shape
    
    while len(strong_x):
        x = strong_x[0]
        y = strong_y[0]
        strong_x = np.delete(strong_x, 0)
        strong_y = np.delete(strong_y, 0)
        for direction in range(len(dx)):
            new_x = x + dx[direction]
            new_y = y + dy[direction]
            if((new_x >= 0 & new_x < size[0] & new_y >= 0 & new_y < size[1]) and (result[new_x, new_y]  == weak)):
                result[new_x, new_y] = strong
                np.append(strong_x, new_x)
                np.append(strong_y, new_y)
    result[result != strong] = 0
    return result
 
 
def Canny(img, low, high):
    Sobel_row, Sobel_col, sobelFilter, angle = convolution(img,3)
    img = nonMaximumSuppression(sobelFilter, angle)
    gradient = np.copy(img)
    img = double_threshold_hysteresis(img, low, high)
    return img, gradient
if __name__ == "__main__":
    img = cv2.imread("final2.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(800,800))
    print(img.shape)
    plt.imshow(img)
    Canny_image, gradient = Canny(img, 0, 150)
    plt.imshow(Canny_image)
    cv2.imwrite("canny.jpg",Canny_image)
