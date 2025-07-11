
import numpy as np
import torch.nn as nn
import torch
from models.model import lstmFcAutoEncoder
from data.dataset import LipEncodeDataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils.serializebyframe import serializenpy
from matplotlib.lines import Line2D
import os
def predict(filename, net):
    datas = serializenpy(filename)
    loss_list = []
    for data in datas:
        data = torch.tensor(data).cuda().type(torch.float32)
        data = data.reshape(data.shape[0], -1).unsqueeze(0)
        out = net(data)
        out = out.squeeze()
        loss = F.mse_loss(out, data.squeeze())
        loss_list.append(loss.item())
        
    return loss_list



net = lstmFcAutoEncoder(input_dim = 80, hidden_dim = 80, output_dim = 40)
net.load_state_dict(torch.load(r'E:\lyh\code\lip_motion_detection\wandb\run-20240510_150008-izf0rrok\files\model_100.pth'))
net.cuda()
net.eval()
plt.figure(figsize=(10,6)) 
healthy = [r'E:\lyh\code\lip_motion_detection\data\4\test\healthy\C60-lips-4.npy', r'E:\lyh\code\lip_motion_detection\data\4\test\healthy\C58-lips-4.npy']
PD = [r'E:\lyh\code\lip_motion_detection\data\4\test\P\PDIN-183-lips-4.npy', r'E:\lyh\code\lip_motion_detection\data\4\test\P\PDIN-184-lips-4.npy']

for filename in os.listdir(r'E:\lyh\code\lip_motion_detection\data\4\test\healthy'):
    filename = os.path.join(r'E:\lyh\code\lip_motion_detection\data\4\test\healthy', filename)
    with torch.no_grad():
        losses = predict(filename, net)
        # if max(losses) < 0.41:
        # plt.plot(losses[:min(7,len(losses))], 'b-') # 绘制损失值
        plt.plot(losses, 'b-')
        
for filename in os.listdir(r'E:\lyh\code\lip_motion_detection\data\4\test\P'):
    filename = os.path.join(r'E:\lyh\code\lip_motion_detection\data\4\test\P', filename)
    with torch.no_grad():
        losses = predict(filename, net)
        # plt.plot(losses[:min(7,len(losses))], 'r--') # 绘制损失值
        plt.plot(losses, 'r--')

legend_elements = [Line2D([0], [0], color='b', lw=2, label='Healthy'),
                   Line2D([0], [0], color='r', lw=2, label='Abnormal')]

plt.xlabel('Time', fontsize = 18) # 设置x轴标签
plt.ylabel('Loss', fontsize = 18) # 设置y轴标签
plt.title('Reconstruction Error Curve', fontsize = 20) # 设置标题
plt.legend(handles=legend_elements, loc='upper right', fontsize=18) # 设置图例
# plt.legend(['testP', 'train']) # 设置图例
plt.savefig('loss3.png') # 保存图片
        
    


