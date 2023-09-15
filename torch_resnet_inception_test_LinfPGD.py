import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from advertorch.utils import predict_from_logits
from advertorch_examples.utils import get_mnist_test_loader
from advertorch_examples.utils import _imshow
from advertorch.attacks import CarliniWagnerL2Attack, LinfPGDAttack, GradientSignAttack, LinfBasicIterativeAttack

from advertorch.defenses import BitSqueezing, JPEGFilter, BinaryFilter

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
filename = "models/inception_v3.pt"
filename2 = "models/tsr_resnet152_25_9_10PM.pt"
use_cuda=True
device = torch.device('cuda')

model = models.inception_v3(pretrained=True,aux_logits=False).to(device)
model.fc = nn.Linear(in_features=2048, out_features=18).to(device)
# Load the pretrained model
model.load_state_dict(torch.load(filename, map_location='cpu'))
# Set the model in evaluation mode. In this case this is for the Dropout layers
model.eval()

model2 = models.resnet152(pretrained=True).to(device)
model2.fc = nn.Linear(in_features=2048, out_features=18).to(device)
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
									  transforms.Resize((299,299)),
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
	trans = transforms.RandomCrop((280,280))
	trans2 = transforms.Resize((299,299)) 
	sample = 8
	sample_list = []

	for i in range(sample):
		img = trans(data)
		img = trans2(img)
		sample_list.append(img)
		
	return sample_list
	
def rancrop2(data):
	trans = transforms.RandomCrop((208,208))
	trans2 = transforms.Resize((224,224)) 
	sample = 10
	sample_list = []

	for i in range(sample):
		img = trans(data)
		img = trans2(img)
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

#adversary = CarliniWagnerL2Attack(model,num_classes=18,max_iterations=80)

adversary = LinfPGDAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.2)

#adversary = GradientSignAttack(model,eps=0.1)

count = 0

jpeg_filter = JPEGFilter(quality=50)
bit_squeezing = BitSqueezing(bit_depth=4)
binary_filter = BinaryFilter()

pipeline = keras_ocr.pipeline.Pipeline()

y_true = []
y_pred1 = []
y_pred2 = []
y_pred3 = []
y_pred4 = []
y_pred5 = []
y_pred6 = []
y_pred7 = []

trans = transforms.Resize((224,224))

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
	adv_inception = adv_untargeted
	adv_resnet = trans(adv_untargeted)
	
	jf = jpeg_filter(adv_inception).to(device)
	bs = bit_squeezing(adv_inception).to(device)
	bf = binary_filter(adv_inception).to(device)
	
	# sample_list = rancrop(adv_inception)
	# pred_rc = torch.zeros(len(sample_list), dtype=torch.int32)
	# for i in range(len(sample_list)):
		# pred = model(sample_list[i])
		# pred_rc[i] = predict_from_logits(pred).to(device)		
	# pred_rc = pred_rc.to(device)
	# rc_mode = stats.mode(pred_rc.cpu().numpy())[0][0]
	# pred1 = rc_mode
	
	pred1 = predict_from_logits(model(adv_inception)).to(device)
	pred1 = pred1.cpu().numpy()[0]
	
	data = ocr(adv_inception)
	data = data.numpy()[0]
	#print(data)

	sample_list1 = rancrop(adv_inception)
	pred_rc1 = torch.zeros(len(sample_list1), dtype=torch.int32)
	sample_list2 = rancrop2(adv_resnet)
	pred_rc2 = torch.zeros(len(sample_list2), dtype=torch.int32)
	for i in range(len(sample_list1)):
		temp1 = model(sample_list1[i])
		pred_rc1[i] = predict_from_logits(temp1).to(device)
	for i in range(len(sample_list2)):
		temp2 = model2(sample_list2[i])
		pred_rc2[i] = predict_from_logits(temp2).to(device)			
	pred_rc1 = pred_rc1.to(device)
	pred_rc2 = pred_rc2.to(device)
	pred_rc = torch.cat((pred_rc1,pred_rc2))
	rc_mode = stats.mode(pred_rc.cpu().numpy())[0][0]
	pred7 = rc_mode	
	
	if(np.array_equal(data,[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0])):
		pred2 = 2
	elif(np.array_equal(data,[0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0])):
		pred2 = 3
	elif(np.array_equal(data,[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0])):
		pred2 = 4
	elif(np.array_equal(data,[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0])):
		pred2 = 5
	elif(np.array_equal(data,[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])):
		pred2 = 7
	elif(np.array_equal(data,[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])):
		pred2 = 8
	elif(np.array_equal(data,[0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]) or np.array_equal(data,[0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])):
		pred2 = 9
	elif(np.array_equal(data,[0,0,0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]) or np.array_equal(data,[0,0,0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]) or np.array_equal(data,[0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]) or np.array_equal(data,[0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0])):
		pred2 = 10
	elif(np.array_equal(data,[0,0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]) or np.array_equal(data,[0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0])):
		pred2 = 11
	elif(np.array_equal(data,[0,0,0,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]) or np.array_equal(data,[0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0])):
		pred2 = 12
	elif(np.array_equal(data,[0,0,0,1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]) or np.array_equal(data,[0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0])):
		pred2 = 13
	elif(np.array_equal(data,[0,0,0,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]) or np.array_equal(data,[0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0])):
		pred2 = 14
	elif(np.array_equal(data,[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])):
		pred2 = 15
	elif(np.array_equal(data,[0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])):
		pred2 = 16
	elif(np.array_equal(data,[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0])):
		pred2 = 17
	else:
		pred2 = pred7
	
	pred3 = stats.mode(pred_rc1.cpu().numpy())[0][0]

	pred4 = predict_from_logits(model(jf)).to(device)
	pred4 = pred4.cpu().numpy()[0]

	pred5 = predict_from_logits(model(bs)).to(device)
	pred5 = pred5.cpu().numpy()[0]

	pred6 = predict_from_logits(model(bf)).to(device)
	pred6 = pred6.cpu().numpy()[0]		
		
	y_true.append(true_label.cpu().numpy()[0])
	y_pred1.append(pred1)
	y_pred2.append(pred2)
	y_pred3.append(pred3)
	y_pred4.append(pred4)
	y_pred5.append(pred5)
	y_pred6.append(pred6)
	y_pred7.append(pred7)
	
	#if(count==5): break

con_mat1 = confusion_matrix(y_true, y_pred1, labels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
con_mat2 = confusion_matrix(y_true, y_pred2, labels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
con_mat3 = confusion_matrix(y_true, y_pred3, labels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
con_mat4 = confusion_matrix(y_true, y_pred4, labels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
con_mat5 = confusion_matrix(y_true, y_pred5, labels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
con_mat6 = confusion_matrix(y_true, y_pred6, labels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
con_mat7 = confusion_matrix(y_true, y_pred7, labels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
print(con_mat1)
print(con_mat2)
print(con_mat3)
print(con_mat4)
print(con_mat5)
print(con_mat6)
print(con_mat7)
cr1 = classification_report(y_true, y_pred1)
cr2 = classification_report(y_true, y_pred2)
cr3 = classification_report(y_true, y_pred3)
cr4 = classification_report(y_true, y_pred4)
cr5 = classification_report(y_true, y_pred5)
cr6 = classification_report(y_true, y_pred6)
cr7 = classification_report(y_true, y_pred7)
print(cr1)
print(cr2)
print(cr3)
print(cr4)
print(cr5)
print(cr6)
print(cr7)
print(accuracy_score(y_true, y_pred1))
print(accuracy_score(y_true, y_pred2))
print(accuracy_score(y_true, y_pred3))
print(accuracy_score(y_true, y_pred4))
print(accuracy_score(y_true, y_pred5))
print(accuracy_score(y_true, y_pred6))
print(accuracy_score(y_true, y_pred7))
