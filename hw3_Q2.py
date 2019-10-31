'''
NAME: MOHANA KRISHNA VUTUKURU
HOMEWORK 3
Driver script for homework 3
'''

from skimage import io

img = io.imread('stadium.bmp')
img = img/255

print(type(img))
