import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from advertorch.utils import predict_from_logits
from advertorch_examples.utils import get_mnist_test_loader
from advertorch_examples.utils import _imshow
from advertorch.attacks import CarliniWagnerL2Attack, LinfPGDAttack, GradientSignAttack, LinfBasicIterativeAttack

from advertorch.defenses import MedianSmoothing2D, BitSqueezing, JPEGFilter

from tqdm import tqdm
from time import sleep

import time

from sklearn.preprocessing import normalize
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from random import randrange

import torchvision.models as models
from torchvision import datasets, transforms

from scipy import stats 
import keras_ocr

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


# ### Load the Model

epsilons = [0, .05, .1, .15, .2, .25, .3]
filename = "models/tsr_resnet152_25_9_10PM.pt"
filename2 = "models/resnet_ocr_2.pt"
use_cuda=True
device = torch.device('cuda')

model = models.resnet152(pretrained=True).to(device)
model.fc = nn.Linear(in_features=2048, out_features=18).to(device)

class MyModel(nn.Module):
	def __init__(self):
		super(MyModel, self).__init__()
		self.cnn = models.resnet152(pretrained=True).to(device)
		self.cnn.fc = nn.Linear(in_features=2048, out_features=35).to(device)
		
		self.fc1 = nn.Linear(35 + 19, 36).to(device)
		self.fc2 = nn.Linear(36, 18).to(device)
		
	def forward(self, image):
		data = ocr(image)
		x1 = self.cnn(image).to(device)
		x2 = data.to(device)
		
		x = torch.cat((x1, x2), dim=1).to(device)
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
	print(prediction_groups)
	word_list = ['signal', 'stop', 'ahead', 'speed', 'limit', '15', '30', '35', '45', '55', '65', 'right', 'left', 'lane', 'ends', 'do', 'not', 'enter', 'yield']
	data = torch.zeros(n_images,len(word_list))
	for i in range(n_images):
		n_words = len(prediction_groups[i])
		for j in range(n_words):
			word = prediction_groups[i][j][0]
			for k in range(len(word_list)):
				if(word == word_list[k]):
					data[i,k] = 1
	return data

def rancrop(data):
	trans = transforms.Resize((112,112))
	trans2 = transforms.RandomCrop((192,192))
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


# ### Create Attack

#adversary = CarliniWagnerL2Attack(model,num_classes=18,max_iterations=100)

adversary = LinfPGDAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.1)

#adversary = GradientSignAttack(model,eps=0.1)

count = 0
count1 = 0
count2 = 0
count3 = 0
count4 = 0
count5 = 0
count6 = 0

#pipeline = keras_ocr.pipeline.Pipeline()

bits_squeezing = BitSqueezing(bit_depth=5)
median_filter = MedianSmoothing2D(kernel_size=3)
jpeg_filter = JPEGFilter(10)

defense = nn.Sequential(
    jpeg_filter,
    bits_squeezing,
    median_filter,
)

pipeline = keras_ocr.pipeline.Pipeline()

for cln_data, true_label in test_loader:
	
	#print ('itr : ', count , end='\t')
	count += 1

	cln_data, true_label = cln_data.to(device), true_label.to(device)
	plt.imshow((torch.squeeze(cln_data).cpu().permute(1, 2, 0)*0.5 + 0.5))
	plt.title(sign_types[true_label])
	plt.show()

	# create the attack.
	adv_untargeted = adversary.perturb(cln_data, true_label)
	adv_defended = defense(adv_untargeted).to(device)
	cln_defended = defense(cln_data).to(device)
	#adv_untargeted = cln_data
	
	#cln_data = convert_to_input(cln_data)
	#adv_untargeted = convert_to_input(adv_untargeted)
	
	pred_resnet = predict_from_logits(model(cln_data)).to(device)
	if(pred_resnet[0] != true_label[0]):
		count1 += 1
	pred_untargeted_adv = predict_from_logits(model(adv_untargeted)).to(device)
	if(pred_untargeted_adv[0] != true_label[0]):
		count2 += 1
	pred_resnet_def = predict_from_logits(model(cln_defended)).to(device)
	if(pred_resnet_def[0] != true_label[0]):
		count3 += 1
	pred_untargeted_adv_def = predict_from_logits(model(adv_defended)).to(device)
	if(pred_untargeted_adv_def[0] != true_label[0]):
		count4 += 1		
	#plt.imshow((torch.squeeze(adv_untargeted).cpu().permute(1, 2, 0)*0.5 + 0.5))
	#plt.title(sign_types[pred_untargeted_adv])
	#plt.show()	
	
	im_ocr_1 = ocr(cln_data)
	im_ocr_2 = ocr(adv_untargeted)
	#print(im_ocr_1)
	#print(im_ocr_2)
	
	sample_list = rancrop(adv_untargeted)
	pred_untargeted_adv_rc = torch.zeros(len(sample_list), dtype=torch.int32)
	#pred_untargeted_adv_rc2 = torch.zeros(len(sample_list), dtype=torch.int32)
	for i in range(len(sample_list)):
		pred1 = model(sample_list[i])
		#pred2 = model2(sample_list[i])
		pred_untargeted_adv_rc[i] = predict_from_logits(pred1).to(device)
		#pred_untargeted_adv_rc2[i] = predict_from_logits(pred2).to(device)
		
		#print(pred1)
		#print(pred2)
		#print(predict_from_logits(pred1), predict_from_logits(pred2), true_label)
		
		#plt.imshow((torch.squeeze(sample_list[i]).cpu().permute(1, 2, 0)*0.5 + 0.5))
		#plt.title(sign_types[pred_untargeted_adv_rc[i].cpu().numpy()])
		#plt.show()
	
	pred_untargeted_adv_rc = pred_untargeted_adv_rc.to(device)
	#pred_untargeted_adv_rc2 = pred_untargeted_adv_rc2.to(device)
	rc_mode = stats.mode(pred_untargeted_adv_rc.cpu().numpy())[0][0]
	#rc_mode2 = stats.mode(pred_untargeted_adv_rc2.cpu().numpy())[0][0]
	if(rc_mode != true_label[0]):
		count5 += 1
	#if(rc_mode2 != true_label[0]):
	#	count6 += 1
	#print ('True : ' , sign_types[true_label])
	#print ('Resnet : ' , sign_types[pred_resnet])
	#print ('Resnet Adv No RC : ' , sign_types[pred_untargeted_adv])
	#print ('Resnet Adv RC: ' , np.array(sign_types)[pred_untargeted_adv_rc.cpu().numpy()])
	
	#print (rc_mode, rc_mode2, true_label[0].tolist())
	
	if(count==5): break

print(count, count1, count2, count3, count4, count5)
