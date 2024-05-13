from _process_dataset import DatasetLSTM, ImageTransform, make_data_path_list_lstm
from torch.utils.data import DataLoader
from tqdm import tqdm
from networks.new_model import CNNLSTM_Trans
import torch
import torch.nn as nn
from config import *
from utils import accuracy, load_model
import os
import matplotlib.pyplot as plt

if os.path.exists(PRE_MODEL):
    net = load_model(CNNLSTM_Trans, PRE_MODEL)
else:
    net = CNNLSTM_Trans(32, 8, 45)

# Create Dataloader
resize = IMG_SIZE_TRAIN
batch_size = BATCH_SIZE
data_root = DATA_ROOT
list_train = make_data_path_list_lstm(root = DATA_ROOT, phase="train", sequence_length= SEQ_LEN)
list_val = make_data_path_list_lstm(root = DATA_ROOT,phase="val", sequence_length= SEQ_LEN)
train_dataset = DatasetLSTM(list_train, transform=ImageTransform(resize), phase="train")
print("co loi khong",len(train_dataset))
val_dataset = DatasetLSTM(list_val, transform=ImageTransform(resize), phase="val")
print("co loi khong nhi",len(val_dataset))
train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
print("loi khong nhi",len(train_dataloader))
val_dataloader = DataLoader(val_dataset, batch_size, shuffle=True)
print("123",len(val_dataloader))
dataloader_dict = {"train": train_dataloader, "val": val_dataloader}

# loss function
criterior = nn.L1Loss() 

# Define classes to learn
for name, param in net.named_parameters():
    param.requires_grad = True  # Always update

# Optimizer
# optimizer = torch.optim.SGD(params=net.parameters(), lr=0.01, momentum=0.9)
optimizer = torch.optim.Adam(params=net.parameters(), lr=0.01)


def train(net, dataloader_dict, criterior, optimizer, epochs, save_path, save_each_epoch):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[DEVICE]: {device}")
    net.to(device)
    #print("123")
    accuracy_list = []
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}:")
        # Train
        epoch_loss = 0
        acc = 0
        for img, label in tqdm(dataloader_dict["train"]):
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                # img2 = ImageTransform.toCv2Img(img[0])
                # cv2.imshow("a", img2)
                # cv2.waitKey(0)
                out = net(img)
                loss = criterior(out, label)
                loss.backward()
                optimizer.step()
                epoch_loss+=loss.item()*img.size(0)
                acc+= accuracy(out, label, 0.3)*batch_size


        loss_avg = epoch_loss/len(dataloader_dict["train"].dataset)
        acc_avg = acc/len(dataloader_dict["train"].dataset)
        accuracy_list.append(acc_avg)
        print(f"Train: -Loss: {loss_avg}, -acc: {acc_avg}")

        # # Val
        epoch_loss1 = 0
        acc1 = 0
        for img, label in dataloader_dict["val"]:
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                out = net(img)
                loss = criterior(out, label)
                epoch_loss1+=loss.item()*img.size(0)
                acc1+= accuracy(out, label, 0.3)*batch_size

        loss_avg = epoch_loss1/len(dataloader_dict["val"].dataset)
        acc_avg1 = acc1/len(dataloader_dict["val"].dataset)
        print(f"Valid: -Loss: {loss_avg}, -acc: {acc_avg1}")

        if save_each_epoch:
            torch.save(net.state_dict(), save_path)
    if not save_each_epoch:
        torch.save(net.state_dict(), save_path)
    
    # Tính accuracy trung bình
    average_accuracy = sum(accuracy_list) / len(accuracy_list)
    # Vẽ đồ thị
    epoch_s = list(range(1, epochs + 1))
    plt.plot(epoch_s, accuracy_list, marker='o')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    #plt.savefig("/content/drive/MyDrive/ĐATN/PD/sevae")
    # Tạo đường dẫn đầy đủ của tập tin đồ thị
    folder_path = "/content/drive/MyDrive/ĐATN/PD/sevae"  # Đường dẫn của thư mục mong muốn
    file_name = 'accuracy_plot.png'  # Tên tập tin ảnh
    file_path = os.path.join(folder_path, file_name)  # Kết hợp đường dẫn của thư mục và tên tập tin

    # Lưu đồ thị vào tập tin ảnh
    plt.savefig(file_path)
    plt.show()
    # In ra accuracy trung bình
    print("Average Accuracy:", average_accuracy)
    
print("Parameters:", sum(p.numel() for p in net.parameters()))
train(net, dataloader_dict, criterior, optimizer, EPOCHS, SAVE_PATH, SAVE_EACH_EPOCH)
