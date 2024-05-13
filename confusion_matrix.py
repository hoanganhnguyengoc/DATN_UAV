import airsim
from networks.cnn_lstm import CNNLSTMModel 
from config import *
from utils import *
from _process_dataset import ImageTransform
import torch
from PIL import Image
import time 
import glob
import os 
import matplotlib.pyplot as plt
import seaborn as sns
from utils import loadLabel
#from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchmetrics import Precision, Recall, F1Score, Accuracy, ConfusionMatrix

#from torchmetrics.plot import PrecisionRecallCurve, ConfusionMatrixPlot, ROCAUCPlot
ROW, COL = 5, 9
THRES = 0.8
w, h = (1, 1)
ROOT = r"/content/drive/MyDrive/ĐATN/Data"
def replaceArea(img, index, value, img_ori):
    img = img.copy()
    # w, h = img.shape[1]//COL, img.shape[0]//ROW
    x, y = index[0]*w, index[1]*h
    if value == 0:
        img[y:y+h, x:x+w] = img_ori[y:y+h, x:x+w]
    if value == 1:
        red = np.zeros((h,w,3), dtype=np.uint8)
        red[:,:] = (0, 0, 255)
        img[y:y+h, x:x+w] = cv2.addWeighted(img_ori[y:y+h, x:x+w], 0.6, red, 0.4, 0)
    return img
def updateImg(img_ori, label):
    img = img_ori.copy()
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            img = replaceArea(img, (j, i), label[i, j], img_ori)
    return img

datatest = glob.glob(os.path.join(ROOT, "images", "**"))
labels = glob.glob(os.path.join(ROOT, "labels", "**"))
net = load_model(CNNLSTMModel(2,5,45), r"E:\Tai lieu\Đồ án tốt nghiệp\ĐATN\PD\models\weightv6_2.pt")
transform = ImageTransform(IMG_SIZE)
cnt = 0
#Khởi tạo các chỉ số đánh giá hiệu suất mô hình
precision = Precision()
recall = Recl()
f1 = F1Score(al)
accuracy1 = Accuracy()
confusion_matrix = ConfusionMatrix()
#roc_auc = ROCAUC()
out_true = []
out_pred = []
while True:
    imgcv2 = cv2.imread(datatest[cnt])
    img = Image.fromarray(imgcv2)
    img = torch.unsqueeze(transform(img, "train"), dim=0)
    w, h = imgcv2.shape[1]//COL, imgcv2.shape[0]//ROW
    # predict
    out = net(img)[0]
    out[out<THRES] = 0
    out[out>=THRES] = 1
    out = torch.reshape(out, (ROW, COL))
    out = torch.tensor(out, dtype=torch.uint8)
    # print("Output:", out)

    out_pred += out.flatten().tolist()
    namef,_ = getNameFile(datatest[cnt])
    out_true += loadLabel(os.path.join(ROOT, "labels", namef+".txt"), ROW, COL).flatten().tolist()
    imgshow = updateImg(imgcv2, out)
    cv2.imshow("im", imgshow)
    key = cv2.waitKey(1)
    cnt+=1
    if key == ord("q") or cnt >= len(datatest):
        break
    # elif key == ord("d"):
#Tính toán các chỉ số       
#cm = confusion_matrix(out_true, out_pred)
precision = precision(out_pred, out_true)
recall = recall(out_pred, out_true)
f1 = f1(out_pred, out_true)
accuracy_v2 = accuracy1(out_pred, out_true)
confusion_matrix = confusion_matrix(out_pred, out_true)
roc_auc = roc_auc(out_pred, out_true)
# In kết quả & Vẽ đồ thị
##precision 
print(f"Precision: {precision}")
plt.plot([0, 1], [precision, precision], label="Precision")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.show()

##Recall 
print(f"Recall: {recall}")
plt.plot([0, 1], [recall, recall], label="Recall")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.show()

##F1
print(f"F1 Score: {f1}")
plt.plot([0, 1], [f1, f1], label="F1 Score")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.show()
##Acc 
print(f"Accuracy: {accuracy}")
plt.bar(["Accuracy"], [accuracy])
plt.xlabel("Metric")
plt.ylabel("Accuracy")
plt.show()
##Confu_Matr
print(f"Confusion Matrix: \n{confusion_matrix}")
sns.heatmap(confusion_matrix, annot=True, fmt="d")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
##ROC
# print(f"ROC AUC: {roc_auc}")
# fpr, tpr, thresholds = roc_auc(out_pred, out_true)
# plt.plot(fpr, tpr, label="ROC Curve")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.legend()
# plt.show()
# disp = ConfusionMatrixDisplay(confusion_matrix=cm)
# disp.plot()
