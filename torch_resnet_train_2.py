import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import os
import random
import numpy as np
import cv2
import time
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os
import keras_ocr
import torchvision.models as models

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

train_transforms = transforms.Compose([transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
									  transforms.RandomRotation(degrees=5),
									  transforms.ColorJitter(),
									  #transforms.RandomHorizontalFlip(),
									  transforms.CenterCrop(size=224),	# Image net standards
									  #transforms.Resize((112,112)),
									  #transforms.Resize((224,224)),
									  transforms.ToTensor(),
									  transforms.Normalize([0.5, 0.5, 0.5],
														   [0.5, 0.5, 0.5])
									  ])

test_transforms = transforms.Compose([
						   #transforms.CenterCrop((224, 224)),
							transforms.Resize((224,224)),
							transforms.ToTensor(),
							transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
					   ])

					   
train_data = datasets.ImageFolder('my_data_aug/train/', train_transforms)
test_data = datasets.ImageFolder('my_data_aug/test/', test_transforms)
#test_data = datasets.ImageFolder('data/dogs-vs-cats/test', test_transforms)

n_train_examples = int(len(train_data)*0.8)
n_valid_examples = n_test_examples = len(train_data) - n_train_examples

train_data, valid_data = torch.utils.data.random_split(train_data, [n_train_examples, n_valid_examples])
#train_data, test_data = torch.utils.data.random_split(train_data, [n_train_examples-n_valid_examples, n_test_examples])

print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(valid_data)}')
print(f'Number of testing examples: {len(test_data)}')

BATCH_SIZE = 32
train_iterator = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
valid_iterator = torch.utils.data.DataLoader(valid_data, batch_size=BATCH_SIZE)
test_iterator = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE)

device = torch.device('cuda')

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

model = MyModel()
for param in model.parameters():
	param.requires_grad = True

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

def calculate_accuracy(fx, y):
	preds = fx.max(1, keepdim=True)[1]
	correct = preds.eq(y.view_as(preds)).sum()
	acc = correct.float()/preds.shape[0]
	return acc
	
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

def train(model, device, iterator, optimizer, criterion):  
	epoch_loss = 0
	epoch_acc = 0	
	model.train()
	for (x, y) in iterator:
		x = x.to(device)
		y = y.to(device)
		optimizer.zero_grad()				  
		fx = model(x).to(device)
		loss = criterion(fx, y)
		acc = calculate_accuracy(fx, y)
		loss.backward()
		optimizer.step()
		epoch_loss += loss.item()
		epoch_acc += acc.item()
		
	return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, device, iterator, criterion):
	
	epoch_loss = 0
	epoch_acc = 0	
	model.eval()
	
	with torch.no_grad():
		for (x, y) in iterator:
			x = x.to(device)
			y = y.to(device)
			fx = model(x).to(device)
			loss = criterion(fx, y)
			acc = calculate_accuracy(fx, y)
			epoch_loss += loss.item()
			epoch_acc += acc.item()
		
	return epoch_loss / len(iterator), epoch_acc / len(iterator)

t_train_acc = []
t_val_acc = []
t_train_loss = []
t_val_loss = []


EPOCHS = 20
SAVE_DIR = 'models'
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'resnet_ocr_sd_2.pt')
best_valid_loss = float('inf')
pipeline = keras_ocr.pipeline.Pipeline()

if not os.path.isdir(f'{SAVE_DIR}'):
	os.makedirs(f'{SAVE_DIR}')

for epoch in range(EPOCHS):
	
	start = time.time()
	
	train_loss, train_acc = train(model, device, train_iterator, optimizer, criterion)
	valid_loss, valid_acc = evaluate(model, device, valid_iterator, criterion)
	
	
	t_train_loss.append(train_loss)
	t_train_acc.append(train_acc)
	
	t_val_acc.append(valid_acc)
	t_val_loss.append(valid_loss)
   
	
	if valid_loss < best_valid_loss:
		best_valid_loss = valid_loss
		torch.save(model.state_dict(), MODEL_SAVE_PATH)
	
	T = time.time() - start
	
	print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:05.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:05.2f}% | T: {T:0.2f}s|')


model.load_state_dict(torch.load(MODEL_SAVE_PATH))
test_loss, test_acc = evaluate(model, device, test_iterator, criterion)
print(f'| Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:05.2f}% |')

import matplotlib.pyplot as plt
acc = t_train_acc
val_acc = t_val_acc
loss = t_train_loss
val_loss = t_val_loss
 
epochs = range(len(acc))
 
plt.plot(epochs, acc, 'b', label='Training acc',marker = "*")
plt.plot(epochs, val_acc, 'r', label='Validation acc',marker = "+")
#plt.title('Training and validation accuracy')
plt.legend()
plt.xlabel('Number of epochs')
plt.ylabel('Accuracy')
plt.savefig('trainning_resnet152_acc.png', dpi=300)
plt.show()

plt.figure()
 
plt.plot(epochs, loss, 'b', label='Training loss',marker = "*")
plt.plot(epochs, val_loss, 'r', label='Validation loss',marker = "+")
#plt.title('Training and validation loss')
plt.legend()
plt.xlabel('Number of epochs')
plt.ylabel('Loss')
plt.savefig('trainning_resnet152_loss.png', dpi=300)
plt.show()
