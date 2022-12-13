import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy
import cv2

img = plt.imread("final3.jpg")
width = img.shape[0]
height = img.shape[1]
img = img.reshape(width*height,3)
print(img.shape)
kmeans = KMeans(n_clusters=6).fit(img)
labels = kmeans.predict(img)
clusters = kmeans.cluster_centers_
print(labels)
print(clusters)
img2 = numpy.zeros((width,height,3), dtype=numpy.uint8)
index = 0
for i in range(height):
	for j in range(width):
		label_of_pixel = labels[index]
		img2[i][j] = clusters[label_of_pixel]
		index += 1
plt.imshow(img2)
cv2.imwrite("6Kmeans.png", img2)
plt.show()