from config import *
import os
import cv2
import sys
#mport airsim
import torch
import numpy as np
from math import pi, acos,sqrt, cos, sin, atan, tan
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
#from torchmetrics import Precision, Recall, F1Score, Accuracy, ConfusionMatrix

def angle2D(vector):
    x, y = vector[0],vector[1]
    if y>=0:
        return acos(x/sqrt(x**2+y**2))
    else:
        return 2*pi - acos(x/sqrt(x**2+y**2))
def mulScalar(v1, v2):
    sum = 0
    length = len(v1) if type(v1) is list else v1.shape[0]
    for i in range(length):
        sum+=v1[i] * v2[i]
    return sum
def angle_2vectors(v1, v2):
    s = mulScalar(v1, v2)
    v1, v2 = np.array(v1), np.array(v2)
    u1 = np.linalg.norm(v1)
    u2 = np.linalg.norm(v2)
    return acos(s/(u1*u2)) # rad

def isCorrect(vectorPred, vecLabel, error_val):
    return error_val > (vecLabel - vectorPred).pow(2).sum().sqrt().item()

def accuracy(vectorPreds, vecLabels, error_val):
    error = abs(vectorPreds - vecLabels)
    return lengthTensor(error[error<error_val])/lengthTensor(error)

# def Loss_(Preds, Label):
#     #t = nn.tensor.ones(size=(16, 5, 45))
#     weight = torch.ones_like(Label)   
#     # num_classes = 45
#     kq = weight @ Label
#     #weight = torch.where(Label == 0, kq * 0.7, kq * 0.3)
#     weight = (weight @ Label)*0.7 + (weight @ torch.logical_not(Label))*0.3
#     criterion = nn.CrossEntropyLoss(weight = weight, reduction="none")
#     return criterion(Label,Preds) 
def Loss_(Preds, Label):
    # Chuyển tensor Label thành chỉ số của lớp đúng
    # label_index = torch.argmax(Label, dim=2)
    # # Loại bỏ chiều thứ hai của tensor Label
    # label_index = label_index.squeeze(1)
    
    # Tạo weight
    # weight = torch.ones_like(Label)
    weight = Label * 0.3 + (1 - Label) * 0.7 
    # min_value = torch.min(Preds)
    # max_value = torch.max(Preds)
    # normalized_data = (Preds - min_value) / (max_value - min_value)
    # Sử dụng hàm mất mát CrossEntropyLoss
    criterion = nn.BCELoss(reduction="none")
    #criterion = nn.CrossEntropyLoss(reduction="none")
    loss = criterion(Preds, Label)
    # print(torch.max(Preds))
    # print(torch.min(Preds))
    # print(weight.size())
    
    return torch.mean(torch.sum(loss * weight, dim=(1, 2)))

# def pre(Preds, Label):
#     precision = Precision()
#     return precision(Preds, Label)

# def rc(Preds, Label):
#     recall = Recall()
#     return recall(Preds, Label)

# def F1(Preds, Label):
#     f1 = F1Score()
#     return f1(Preds, Label)

# def confusion_matrix(Preds, Label):
#     cm = ConfusionMatrix()
#     return cm(Preds, Label)

# def Draw_graph(epoch, x):
#     train_value = []
#     validation_value =[]
#     for epoch in range(num_epochs):
#     # Train model và tính loss
#     train_loss = train_model(...)
    
#     # Validate model và tính validation loss
#     validation_loss = validate_model(...)
    
#     # Cập nhật giá trị loss vào mảng
#     train_loss_values.append(train_loss)
#     validation_loss_values.append(validation_loss)

# # Vẽ đồ thị
# plt.plot(range(1, num_epochs + 1), train_loss_values, label='Train Loss')
# plt.plot(range(1, num_epochs + 1), validation_loss_values, label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training and Validation Loss over Epochs')
def processListPath(list_path):
    lis = []
    for i in list_path:
        i=i.replace("\\", "/")
        lis.append(i)
    return lis
def saveLabel(path, label):
    f = open(path, "w")
    text = ""
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            text+=str(label[i, j])
            text+=" " if (j<label.shape[1]-1) else ""
        text+="\n" if (i<label.shape[0]-1) else ""
    f.write(text)
    f.close()
    print(f"Saved: {path}")

def loadLabel(path, ROW, COL):
    if os.path.exists(path):
        f = open(path, "r", encoding="utf-8")
        data = f.read().split("\n")
        data_list = []
        for i in data:
            data_temp = []
            for j in i.split(" "):
                data_temp.append(int(j))
            data_list.append(data_temp)
        
        data = np.array(data_list)
        if ((ROW, COL)!= data.shape):
            data = np.zeros((ROW, COL), dtype=np.uint8)
        f.close()
    else:
        data = np.zeros((ROW, COL), dtype=np.uint8)
    return data

def UAVStart():
    client = airsim.MultirotorClient()
    client.confirmConnection()

    client.enableApiControl(True)
    print("arming the drone...")
    client.armDisarm(True)

    landed = client.getMultirotorState().landed_state
    if landed == airsim.LandedState.Landed:
        print("taking off...")
        client.takeoffAsync().join()

    landed = client.getMultirotorState().landed_state
    if landed == airsim.LandedState.Landed:
        print("takeoff failed - check Unreal message log for details")
        exit()
    return client

def getRGBImg(client):
        rawImage = client.simGetImage("0", airsim.ImageType.Scene)
        if (rawImage == None):
            print("Camera is not returning image, please check airsim for error messages")
            sys.exit(0)
        else:
            png = cv2.imdecode(airsim.string_to_uint8_array(rawImage), cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(png, cv2.COLOR_RGBA2RGB)
            return img

def getPILImg(client):
        rawImage = client.simGetImage("0", airsim.ImageType.Scene)
        if (rawImage == None):
            print("Camera is not returning image, please check airsim for error messages")
            sys.exit(0)
        else:
            png = cv2.imdecode(airsim.string_to_uint8_array(rawImage), cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(png, cv2.COLOR_BGR2RGB)
            return img

def load_model(net, path="./weight.pth"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net.to(device)
    load_weight = torch.load(path, map_location=device)
    
    print(f"[DEVICE]: {device}")
    # load_weight = torch.load(path, map_location=("cuda:0"))
    net.load_state_dict(load_weight)
    return net

def lengthTensor(tensor: torch.Tensor):
    l = 1
    for i in tensor.size():
        l*=i
    return l

def pil2cv2(img):
    open_cv_image = np.array(img) 
    # Convert RGB to BGR 
    return open_cv_image[:, :, ::-1].copy() 

def getNameFile(path):
    file = os.path.basename(path)
    name, tail = file.split(".")
    return name, tail
def getVFOV(HFOV, img_size):
    ''' 
        HFOV: Degree
        img_size: (w, h)
    '''
    w, h = img_size
    return 2*atan(h/w * tan(HFOV*pi/360))*180/pi
def getVFOVRad(HFOV, img_size):
    ''' 
        HFOV: rad
        img_size: (w, h)
    '''
    w, h = img_size
    return 2*atan(h/w * tan(HFOV/2))
def rollPoint2D(point, angle):
    '''angle: rad'''
    x,y = point
    return x*cos(angle)-y*sin(angle), x*sin(angle) + y*cos(angle)