

import os
import matplotlib.pyplot as plt
import pylab
import librosa
import librosa.display
import numpy as np
from scipy import misc
from PIL import Image


img_array = np.load('data.npy')
output = []


k=11
for imgNum in range((k-1)*50, k*50):
	m = np.zeros((img_array.shape[1],img_array.shape[2]))
	for i in range(0, img_array.shape[1]):
		for j in range(0, img_array.shape[2]):
			m[i][j] = img_array[imgNum][i][j][0];


	
	fig = plt.figure(frameon=False, figsize=(2.8,2.83))
	plt.axis('off')
	plt.margins(0)
	librosa.display.specshow(m)
	fig.savefig('temp.png', bbox_inches='tight', pad_inches=0)
	pylab.close(fig)

	file = Image.open('temp.png')
	RGB = np.array(file.convert('RGB'))
	output.append(RGB)
	file.close()
	##print imgNum

output = np.array(output)
np.save("out"+str(k)+".npy", output)