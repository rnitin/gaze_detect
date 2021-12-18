### IMPORT ###
import torch
import numpy as np
import math

import torch.utils.data
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F

import glob
import torchvision
import scipy.io

import cv2
from PIL import Image
from torchvision.utils import make_grid
from torchvision.io import read_image
import torchvision.transforms.functional as Fv
import matplotlib.pyplot as plt

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from utils_common import convert_polar_vector_single, convert_polar_vector
### END IMPORT 


### CONFIG ###
data_path = "../data/MPIIGaze/Data/Normalized/" # training data path
save_path = "../assets/models/gazenet_gh_alex.pt" # save path for trained model
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
criterion = nn.MSELoss() # mean square loss
epochs = 50 # no of training epochs

### END CONFIG ###


### DATA LOADER ###
class MPIIGaze(Dataset): # custom data loader for MPIIGAZE normalized data
    def __init__(self, data_path, transform=None):
        self.image_list = []
        self.label_list = []
        self.pose_list = []
        for mat_file in glob.glob(data_path + "*/*.mat"):
            mat = scipy.io.loadmat(mat_file)
            gaze_list_arr = mat["data"][0].item()[0].item()[0] # gaze
            pose_list_arr = mat["data"][0].item()[0].item()[2]  # image
            img_list_arr = mat["data"][0].item()[0].item()[1] # pose
            for i in range(len(img_list_arr)):
                img_arr = img_list_arr[i]
                gaze_arr = gaze_list_arr[i]
                pose_arr = pose_list_arr[i]
                self.image_list.append(img_arr)
                self.label_list.append(gaze_arr)
                self.pose_list.append(pose_arr)
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.transform_resize = transforms.Compose([transforms.Resize((36*8, 60*8))]) # AlexNet requires min dim 224x224

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        img = self.transform(self.image_list[index])
        img = self.transform_resize(img)
        x = torch.tensor(self.label_list[index][0])
        y = torch.tensor(self.label_list[index][1])
        z = torch.tensor(self.label_list[index][2])
        theta = torch.asin(-1*y)
        phi = torch.atan2(-1*x, -1*z) 
        g  = torch.tensor([theta, phi])
        h = torch.tensor(self.pose_list[index])
        return img, g, h
dataset = MPIIGaze(data_path)

train_size = int(0.1 * len(dataset))
valid_size = int(0.1 * len(dataset))
test_size = len(dataset) - (train_size + valid_size)

train_set, valid_set, test_set = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size]) # create train, validation and test sets

train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=256, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=True)
### END DATA LOADER ###


### MODEL DEFINITION ###

# Download predefined AlexNet and modify for grayscale and to reduce params
AlexNet = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=False)
AlexNet.features[0] = nn.Conv2d(1, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
AlexNet.classifier[4] = nn.Linear(4096, 1000)
AlexNet.classifier[6] = nn.Linear(1000, 128)

class GazeAlexNet(nn.Module): # AlexNet based model
    def __init__(self):
        super(GazeAlexNet, self).__init__()
        self.Alex = AlexNet
        self.reg = nn.Linear(131,2) # regression layer
        self.float()
    def forward(self, x, y):
        x = self.Alex(x)
        x = self.reg(torch.cat((x,y.float()), dim = 1))  # concatenate head pose before last layer
        return x

gazenet = GazeAlexNet() # create model
optimizer = optim.Adam(gazenet.parameters()) # use Adam optimizer
### END MODEL DEF ###


### TRAIN ###
for epoch in range(epochs):
    train_loss = 0
    val_loss = 0
    accuracy = 0
    
    # Training the model
    gazenet.to(device) # move model to device
    gazenet.train()
    counter = 0
    for inputs, labels_g, labels_h in train_loader:
        inputs, labels_g, labels_h = inputs.to(device), labels_g.to(device), labels_h.to(device) # move to device
        optimizer.zero_grad() # clear optimizer
        outputs = gazenet.forward(inputs, labels_h) # compute outputs
        loss = criterion(outputs.double(), labels_g.double()) # compute loss
        loss.backward() # backpropagation
        optimizer.step() 
        train_loss += loss.item()*inputs.size(0)
        counter += 1
        #print(counter, "/", len(train_loader))
        
    # Validating the model
    gazenet.eval()
    counter = 0
    with torch.no_grad():
        for inputs, labels_g, labels_h in valid_loader:
            inputs, labels_g, labels_h = inputs.to(device), labels_g.to(device), labels_h.to(device) 
            outputs = gazenet.forward(inputs, labels_h) 
            valloss = criterion(outputs.double(), labels_g.double())
            val_loss += valloss.item()*inputs.size(0)
            counter += 1
            #print(counter, "/", len(valid_loader))
    
    # Get the average loss for the entire epoch
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = val_loss/len(valid_loader.dataset)
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))

torch.save(gazenet, save_path) # save trained model
### END TRAIN ###

### EVALUATE TRAINED MODEL ###
test_loss = 0
accuracy = 0
with torch.no_grad():
    gazenet.to(device)
    gazenet.eval()
    counter = 0
    error = 0
    for inputs, labels_g, labels_h in test_loader:
        inputs, labels_g, labels_h = inputs.to(device), labels_g.to(device), labels_h.to(device)
        outputs = gazenet.forward(inputs, labels_h)
        loss = criterion(outputs.double(), labels_g.double())
        test_loss += loss.item()*inputs.size(0)
        ip_x, ip_y, ip_z = convert_polar_vector(labels_g) # find cartesian gaze vector from ground truth
        op_x, op_y, op_z = convert_polar_vector(outputs) # find cartesian gaze vector from output
        costheta = ip_x*op_x + ip_y*op_y + ip_z*op_z # cosine similarity
        costheta = torch.clamp(costheta, min=-1.0, max=1.0) # prevent singularity when costheta exceeds limits due to Python
        error_rad = torch.acos(costheta) # error between output and truth
        error_deg = error_rad * 180 / math.pi
        error += torch.mean(error_deg)
        counter += 1
    test_error = error / counter # mean angle error on test set
    test_loss = test_loss/len(test_loader.dataset)
    print("Test loss: {:.4f} \tTest error: {:.4f}".format(test_loss, test_error))
### END EVAL  ###

### VISUALIZE TEST DATA ###
examples = enumerate(test_loader)
batch_id, (images, labels_g, labels_h) = next(examples) # read a batch from the test set
images = images[0:4].to(device)
labels_h = labels_h[0:4].to(device)
outputs = gazenet(images, labels_h)
output_gazevec = torch.stack(convert_polar_vector(outputs), dim = 1) # get 3D gaze Cartesian vector
plt.figure(figsize=(13,3))
plt.suptitle("Head pose independent, model: LeNet")

for i in range(0,4): # plot projection of 3D gaze on the XY plane
    plt.subplot(1,4,i+1)
    g_vec = output_gazevec[i].squeeze().detach().numpy()
    ip_g_vec = convert_polar_vector_single(labels_g[i])
    img_gray = images[i].squeeze()
    img_len_x = img_gray.shape[1]
    img_len_y = img_gray.shape[0]
    center_x = img_len_x / 2
    center_y = img_len_y / 2
    plt.quiver([center_x],[center_y],g_vec[0],-1*g_vec[1], scale=0.2, scale_units='inches', color="green")
    plt.quiver([center_x],[center_y],ip_g_vec[0],-1*ip_g_vec[1], scale=0.2, scale_units='inches', color="blue")
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img_gray, cmap="gray")
### END EVAL  ###
