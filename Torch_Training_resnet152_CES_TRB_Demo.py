
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
import torchvision.datasets as datasets
import os
import random
import numpy as np
import time
import copy
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

import matplotlib.pyplot as plt
plt.ion()

print ('OK')


# In[2]:


import glob
import os
imgs = 'my_data_aug/train/*/*'
imgs = glob.glob(imgs)
#print (imgs[3])

for img in imgs:
    try:
        #print (img)
        Image.open(img)
    except:
        os.remove(img)
        print ('removed:', img)
        continue
print ('Done!')    


# In[3]:


SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

print ('OK')


# In[28]:


train_transforms = transforms.Compose([
                           #transforms.RandomHorizontalFlip(),
                           transforms.RandomRotation(10),
                           transforms.RandomCrop((224, 224), pad_if_needed=True),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
                       ])

train_transforms = transforms.Compose([transforms.RandomResizedCrop(size=256, scale=(0.5, 1.0)),
                                        transforms.RandomRotation(degrees=10),
                                        #transforms.RandomHorizontalFlip(),
                                        transforms.ColorJitter(),
                                        transforms.CenterCrop(size=224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                      ])

test_transforms = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                       ])


print ('OK')


# In[33]:


train_data = datasets.ImageFolder('my_data_aug/train/', train_transforms)
test_data = datasets.ImageFolder('my_data_aug/test/', test_transforms)
#test_data = datasets.ImageFolder('data/dogs-vs-cats/test', test_transforms)

import os

#print(len(os.listdir('my_data/train')))

#train_data = dataloaders['train']
#test_data = dataloaders['test']

n_train_examples = int(len(train_data)*0.8)
n_valid_examples = n_test_examples = len(train_data) - n_train_examples

train_data, valid_data = torch.utils.data.random_split(train_data, [n_train_examples, n_valid_examples])
#train_data, test_data = torch.utils.data.random_split(train_data, [n_train_examples-n_valid_examples, n_test_examples])


# In[34]:


print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(valid_data)}')
print(f'Number of testing examples: {len(test_data)}')


# In[35]:


BATCH_SIZE = 4
train_iterator = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
valid_iterator = torch.utils.data.DataLoader(valid_data, batch_size=BATCH_SIZE)
test_iterator = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE)


# In[36]:


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(train_iterator))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])


# ### Configure Model Hyper-parameters

# In[38]:


device = torch.device('cuda')


# In[39]:


import torchvision.models as models
model = models.resnet152(pretrained=True).to(device)


# In[40]:


for param in model.parameters():
    param.requires_grad = True


# In[41]:


print(model.fc)


# In[42]:


model.fc = nn.Linear(in_features=2048, out_features=18).to(device)


# In[43]:


optimizer = optim.Adam(model.parameters())


# In[44]:


criterion = nn.CrossEntropyLoss()


# In[45]:


# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


# ### Train Model

# In[46]:


SAVE_DIR = 'models'
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'tsr_resnet152_25_9_10PM.pt')

if not os.path.isdir(f'{SAVE_DIR}'):
    os.makedirs(f'{SAVE_DIR}')
    


# In[47]:


def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = correct.float()/preds.shape[0]
    return acc


# In[48]:


def train(model, device, iterator, optimizer, criterion):  
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for (x, y) in iterator:
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
              
        fx = model(x)

        loss = criterion(fx, y)
        
        acc = calculate_accuracy(fx, y)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# In[49]:


def evaluate(model, device, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
        for (x, y) in iterator:

            x = x.to(device)
            y = y.to(device)

            fx = model(x)

            loss = criterion(fx, y)

            acc = calculate_accuracy(fx, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# In[50]:


t_train_acc = []
t_val_acc = []
t_train_loss = []
t_val_loss = []


# In[51]:


EPOCHS = 50

SAVE_DIR = 'models'
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'tsr_resnet152_25_9_10PM.pt')

best_valid_loss = float('inf')

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


# In[52]:


model.load_state_dict(torch.load(MODEL_SAVE_PATH))
test_loss, test_acc = evaluate(model, device, test_iterator, criterion)
print(f'| Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:05.2f}% |')


# In[53]:


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

