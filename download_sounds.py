import urllib.request
import requests

print('Beginning file downlaod..')

for i in range(1, 40):
   url = 'http://accent.gmu.edu/soundtracks/portuguese' + str(i) + '.mp3'
   save = './pt/portuguese'+str(i)+'.mp3'
   urllib.request.urlretrieve(url, save)
