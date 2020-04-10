import torch     
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, TensorDataset
import copy
from sklearn.metrics import accuracy_score,f1_score,roc_curve,precision_recall_curve,average_precision_score,auc
from sklearn.metrics import precision_score, recall_score, f1_score,confusion_matrix,matthews_corrcoef,roc_auc_score
import matplotlib.pyplot as plt
import torch.utils.data as Data
import numpy
import pandas as pd
from tqdm import tqdm
from PIL import Image
import cv2

BATCH_SIZE = 32

class MyDataset(Dataset):

    def __init__(self,sign_for_train):
        if sign_for_train == 1:
            datasets,labellists,length = self.pre_process("train.npy",1)
        else:
            datasets,labellists,length = self.pre_process("train.npy",0)
        self.x_data = datasets
        self.y_data = labellists
        self.len = length
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

    def label_to_onehot(self,label,class_num):
        one_hot = numpy.zeros(class_num)
        one_hot[label] = 1
        return one_hot

    def pre_process(self,npy_source,sign_for_train):
        if sign_for_train == 1:
            current_npy = numpy.load(npy_source)
            current_labelmes = pd.read_csv("train.csv")
            length = current_npy.shape[0] *4 // 5
            dataset = []
            dataset_channel = []
            label_list = []
            k = 0
            for i in range(current_npy.shape[0]):
                if (i % 3000) < 2400:
                    #图像
                    picture = []
                    picture_line = []
                    for j in range(current_npy.shape[1]):
                        if k < 28:
                            threshold = 50
                            current_data = current_npy[i][j]
                            if j != 0 and j != (current_npy.shape[1] - 1) and abs(int(current_npy[i][j]) - int(current_npy[i][j-1])) > threshold and abs(int(current_npy[i][j]) - int(current_npy[i][j+1])) > threshold:
                                current_data = current_npy[i][j-1]
                            picture_line.append(current_data/255)
                            k = k + 1
                            if k == 28:
                                k = 0
                                picture.append(picture_line)
                                picture_line = []
                    #图像大小变换并去噪
                    pic_array = numpy.array(picture)
                    tempim = Image.fromarray(pic_array)
                    tempim = tempim.resize((64,64)) #大小变换
                    img = cv2.cvtColor(numpy.asarray(tempim),cv2.COLOR_RGB2BGR)
                    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    img = cv2.medianBlur(img, 3) #中值滤波去噪
                    pic_array = numpy.array(img)
                    picture = pic_array
                    #封装
                    dataset_channel.append(picture)
                    dataset.append(dataset_channel)
                    dataset_channel = []
                    #标签
                    temp_label = current_labelmes["label"][i]
                    label_list.append(int(temp_label))
            data_tensor = torch.Tensor(dataset)
            label_tensor = torch.Tensor(label_list)
        else:
            current_npy = numpy.load(npy_source)
            current_labelmes = pd.read_csv("train.csv")
            length = current_npy.shape[0] // 5
            dataset = []
            dataset_channel = []
            label_list = []
            k = 0
            for i in range(current_npy.shape[0]):
                if (i % 3000) >= 2400:
                    #图像
                    picture = []
                    picture_line = []
                    for j in range(current_npy.shape[1]):
                        if k < 28:
                            threshold = 50
                            current_data = current_npy[i][j]
                            if j != 0 and j != (current_npy.shape[1] - 1) and abs(int(current_npy[i][j]) - int(current_npy[i][j-1])) > threshold and abs(int(current_npy[i][j]) - int(current_npy[i][j+1])) > threshold:
                                current_data = current_npy[i][j-1]
                            picture_line.append(current_data/255)
                            k = k + 1
                            if k == 28:
                                k = 0
                                picture.append(picture_line)
                                picture_line = []
                    #图像大小变换并去噪
                    pic_array = numpy.array(picture)
                    tempim = Image.fromarray(pic_array)
                    tempim = tempim.resize((64,64)) #大小变换
                    img = cv2.cvtColor(numpy.asarray(tempim),cv2.COLOR_RGB2BGR)
                    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    img = cv2.medianBlur(img, 3) #中值滤波去噪
                    pic_array = numpy.array(img)
                    picture = pic_array
                    #封装
                    dataset_channel.append(picture)
                    dataset.append(dataset_channel)
                    dataset_channel = []
                    #标签
                    temp_label = current_labelmes["label"][i]
                    label_list.append(int(temp_label))
            data_tensor = torch.Tensor(dataset)
            label_tensor = torch.Tensor(label_list)
        return data_tensor,label_tensor,length

#加载文件中的数据集
my_train_dataset = MyDataset(1)
my_test_dataset = MyDataset(0)

#加载小批次数据，即将数据集中的data分成每组batch_size的小块，shuffle指定是否随机读取
my_train_loader = Data.DataLoader(dataset=my_train_dataset,batch_size=BATCH_SIZE,shuffle=True)
my_test_loader = Data.DataLoader(dataset=my_test_dataset,batch_size=BATCH_SIZE,shuffle=True)

print(my_train_dataset.x_data[0][0].numpy())

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        #卷积层
        self.conv7 = nn.Conv2d(1, 32, 3, padding = 1)
        self.conv8 = nn.Conv2d(32, 32, 3, padding = 1)
        self.conv1 = nn.Conv2d(32, 64, 3, padding = 1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding = 1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding = 1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding = 1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding = 1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding = 1)
        #池化层
        self.maxpool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AvgPool2d(2, 2)
        self.globalavgpool = nn.AvgPool2d(16, 16)
        #Batchnorm层
        self.bn0 = nn.BatchNorm2d(32)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        #Dropout层
        self.dropout50 = nn.Dropout(0.5)
        self.dropout10 = nn.Dropout(0.1)
        #全连接层
        self.fc = nn.Linear(256, 10)
 
    def forward(self, x):
        x = self.bn0(F.relu(self.conv7(x)))
        x = self.bn0(F.relu(self.conv8(x)))
        x = self.maxpool(x)
        x = self.dropout10(x)
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn1(F.relu(self.conv2(x)))
        x = self.maxpool(x)
        x = self.dropout10(x)
        x = self.bn2(F.relu(self.conv3(x)))
        x = self.bn2(F.relu(self.conv4(x)))
        x = self.avgpool(x)
        x = self.dropout10(x)
        x = self.bn3(F.relu(self.conv5(x)))
        x = self.bn3(F.relu(self.conv6(x)))
        x = self.globalavgpool(x)
        x = self.dropout50(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x