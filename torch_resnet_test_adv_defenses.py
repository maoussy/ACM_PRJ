import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from advertorch.utils import predict_from_logits
from advertorch_examples.utils import get_mnist_test_loader
from advertorch_examples.utils import _imshow
from advertorch.attacks import CarliniWagnerL2Attack, LinfPGDAttack, GradientSignAttack, LinfBasicIterativeAttack

from advertorch.defenses import ConvSmoothing2D, AverageSmoothing2D, GaussianSmoothing2D, MedianSmoothing2D, BitSqueezing, JPEGFilter, BinaryFilter

from tqdm import tqdm
from time import sleep

import time

from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from PIL import Image
from random import randrange

import torchvision.models as models
from torchvision import datasets, transforms

from scipy import stats 
import keras_ocr
import cv2

torch.manual_seed(0)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


# ### Load the Model

epsilons = [0, .05, .1, .15, .2, .25, .3]
filename = "models/tsr_resnet152_25_9_10PM.pt"
filename2 = "models/resnet_ocr_sd_2.pt"
use_cuda=True
device = torch.device('cuda')

model = models.resnet152(pretrained=True).to(device)
model.fc = nn.Linear(in_features=2048, out_features=18).to(device)

class MyModel(nn.Module):
	def __init__(self):
		super(MyModel, self).__init__()
		self.cnn = models.resnet152(pretrained=True).to(device)
		self.cnn.fc = nn.Linear(in_features=2048, out_features=29).to(device)
		
		self.fc1 = nn.Linear(29 + 21 + 4, 36).to(device)
		self.fc2 = nn.Linear(36, 18).to(device)
		
	def forward(self, image):
		data = ocr(image)
		data2 = sd(image)
		x1 = self.cnn(image).to(device)
		x2 = data.to(device)
		x3 = data2.to(device)
		x = torch.cat((x1, x2, x3), dim=1).to(device)
		x = F.relu(self.fc1(x)).to(device)
		x = self.fc2(x).to(device)
		return x

# Load the pretrained model
model.load_state_dict(torch.load(filename, map_location='cpu'))
# Set the model in evaluation mode. In this case this is for the Dropout layers
model.eval()

model2 = MyModel()
# Load the pretrained model
model2.load_state_dict(torch.load(filename2, map_location='cpu'))
# Set the model in evaluation mode. In this case this is for the Dropout layers
model2.eval()

sign_types = ['curve left', 'curve right', 'do not enter', 
			  'lane end', 'merge', 'pedestrian crossing', 
			  'roundabout', 'school zone', 'signal ahead', 
			  'speed limit 15', 'speed limit 30', 'speed limit 35', 
			  'speed limit 45', 'speed limit 55', 'speed limit 65',
			  'stop', 'stop ahead', 'yield']


# ### Data Loader

test_transforms = transforms.Compose([#transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
									  #transforms.CenterCrop(size=256),
									  #transforms.ToPILImage(),
									  transforms.Resize((224,224)),
									  transforms.ToTensor(),
									  transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])


									  #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

test_data = datasets.ImageFolder('my_data_aug/test/', test_transforms)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1,shuffle=True)

def ocr(x):
	images = []
	n_images = x.shape[0] 
	for i in range(n_images):
		im = x[0].cpu().permute(1,2,0)
		im = im*0.5+0.5
		im = im.detach().numpy()
		im = im*255
		images.append(im)
	prediction_groups = pipeline.recognize(images)
	word_list = ['signal', 'stop', 'ahead', 'speed', 'limit', '15', '30', '3o', '35', '45', '55', '65', 'lane', 'ends', 'do', 'not', 'enter', 'yield', 'merge', 'pedestrian', 'school']
	data = torch.zeros(n_images,len(word_list))
	for i in range(n_images):
		n_words = len(prediction_groups[i])
		for j in range(n_words):
			word = prediction_groups[i][j][0]
			for k in range(len(word_list)):
				if(word == word_list[k]):
					data[i,k] = 1
	return data

def sd(x):
	n_images = x.shape[0] 
	data = torch.zeros(n_images,4)
	for i in range(n_images):
		im = x[0].cpu().permute(1,2,0)
		im = im*0.5+0.5
		im = im.detach().numpy()
		im2 = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
		im2 = cv2.cvtColor(im2, cv2.COLOR_RGB2BGR)
		im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
		_,threshold = cv2.threshold(im2, 140, 255, cv2.THRESH_BINARY) 
		contours,_ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
		areas = []
		sides = []
		for cnt in contours: 
			area = cv2.contourArea(cnt)
			if area > 1000:	
				approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True) 
				areas.append(area)
				sides.append(len(approx))
		if(len(areas)<=1):
			n_side = 0
		elif(len(areas)==2):
			ar_ind = np.argsort(areas)[-1]
			n_side = sides[ar_ind]
		else:
			ar_ind = np.argsort(areas)[-2]
			n_side = sides[ar_ind] 
			if(n_side>15):
				n_side = 15
		side_bin = [int(j) for j in list('{0:04b}'.format(n_side))] 
		side_bin = side_bin[-4:]
		data[i,:] = torch.Tensor(side_bin)
	return data

def rancrop(data):
	trans = transforms.Resize((112,112))
	trans2 = transforms.RandomCrop((208,208))
	trans3 = transforms.Resize((224,224)) 
	#data = trans(data)
	#data = trans3(data)
	sample = 10
	sample_list = []

	for i in range(sample):
		img = trans2(data)
		img = trans3(img)
		sample_list.append(img)
		
	return sample_list

#imarr = np.array(im)
#print(imarr.shape)
#plt.imshow(imarr)
#plt.show()
#c_true_label = true_label.copy()
#plt.imshow((c_cln_data[0].permute(1, 2, 0)*0.5 + 0.5))
#plt.title(sign_types[true_label[0]])
#plt.show()

def plot_classification_report(cr, title='Classification report ', with_avg_total=False, cmap=plt.cm.Blues):

	lines = cr.split('\n')

	classes = []
	plotMat = []
	for line in lines[2 : (len(lines) - 3)]:
		#print(line)
		t = line.split()
		# print(t)
		if(len(t)==0):
			break
		classes.append(t[0])
		v = [float(x) for x in t[1: len(t) - 1]]
		#print(v)
		plotMat.append(v)

	if with_avg_total:
		aveTotal = lines[len(lines) - 1].split()
		classes.append('avg/total')
		vAveTotal = [float(x) for x in t[1:len(aveTotal) - 1]]
		plotMat.append(vAveTotal)

	return plotMat, classes

# ### Create Attack

#adversary = CarliniWagnerL2Attack(model,num_classes=18,max_iterations=100)

adversary = LinfPGDAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.1)

#adversary = GradientSignAttack(model,eps=0.1)

count = 0

#conv_smoothing = ConvSmoothing2D(kernel=3)
#average_smoothing = AverageSmoothing2D(channels=3, kernel_size=3)
#gaussian_smoothing = GaussianSmoothing2D(sigma=0.5, channels=3, kernel_size=3)
median_smoothing = MedianSmoothing2D(kernel_size=3, stride=1)
jpeg_filter = JPEGFilter(quality=80)
bit_squeezing = BitSqueezing(bit_depth=6)
binary_filter = BinaryFilter()

#pipeline = keras_ocr.pipeline.Pipeline()

y_true = []
y_pred1 = []
y_pred2 = []
y_pred3 = []
y_pred4 = []
y_pred5 = []
y_pred6 = []
y_pred7 = []

for cln_data, true_label in test_loader:
	
	#print ('itr : ', count , end='\t')
	count += 1

	cln_data, true_label = cln_data.to(device), true_label.to(device)
	#plt.imshow((torch.squeeze(cln_data).cpu().permute(1, 2, 0)*0.5 + 0.5))
	#plt.title(sign_types[true_label])
	#plt.show()

	# create the attack.
	#adv_untargeted = cln_data
	adv_untargeted = adversary.perturb(cln_data, true_label)
	adv_untargeted = adv_untargeted.to(device)
	
	#defense1 = conv_smoothing(adv_untargeted).to(device)
	#defense2 = average_smoothing(adv_untargeted).to(device)
	#defense3 = gaussian_smoothing(adv_untargeted).to(device)
	defense4 = median_smoothing(adv_untargeted).to(device)
	defense5 = jpeg_filter(adv_untargeted).to(device)
	defense6 = bit_squeezing(adv_untargeted).to(device)
	defense7 = binary_filter(adv_untargeted).to(device)
	
	#pred1 = predict_from_logits(model(defense1)).to(device)
	#pred2 = predict_from_logits(model(defense2)).to(device)
	#pred3 = predict_from_logits(model(defense3)).to(device)
	pred4 = predict_from_logits(model(defense4)).to(device)
	pred5 = predict_from_logits(model(defense5)).to(device)
	pred6 = predict_from_logits(model(defense6)).to(device)
	pred7 = predict_from_logits(model(defense7)).to(device)
		
	y_true.append(true_label.cpu().numpy()[0])
	#y_pred1.append(pred1.cpu().numpy()[0])
	#y_pred2.append(pred2.cpu().numpy()[0])
	#y_pred3.append(pred3.cpu().numpy()[0])
	y_pred4.append(pred4.cpu().numpy()[0])
	y_pred5.append(pred5.cpu().numpy()[0])
	y_pred6.append(pred6.cpu().numpy()[0])
	y_pred7.append(pred7.cpu().numpy()[0])
	
	#if(count==5): break

#print(accuracy_score(y_true, y_pred1))
#print(accuracy_score(y_true, y_pred2))
#print(accuracy_score(y_true, y_pred3))
print(accuracy_score(y_true, y_pred4))
print(accuracy_score(y_true, y_pred5))
print(accuracy_score(y_true, y_pred6))
print(accuracy_score(y_true, y_pred7))
