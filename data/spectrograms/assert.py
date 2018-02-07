import os
import matplotlib.pyplot as plt
import pylab
import librosa
import librosa.display
import numpy as np
from scipy import misc
from PIL import Image




for k in range(1, 12):
	img_array = np.load('out'+str(k)+'.npy')
	print img_array.shape[0]
	for i in range(img_array.shape[0]):
		#print str(img_array[i].shape[0])+" "+str(img_array[i].shape[1])
		if(img_array[i].shape[0]!=224 or img_array[i].shape[1]!=224):
			print "caooooo"