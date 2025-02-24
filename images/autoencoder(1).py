import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm
import os
import scipy.io as sio
import numpy as np
from torch.utils.data import Dataset, DataLoader

class CorrelationDataset(Dataset):
    def __init__(self, input_folder):
        self.file_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.mat')]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # 读取.mat文件
        file_path = self.file_paths[idx]
        data = sio.loadmat(file_path)
        correlation_matrix = data.get('ROICorrelation', None)

        if correlation_matrix is None:
            raise ValueError(f"No valid 'FC' found in {file_path}")

        # Fisher Z转换
        fisher_z_matrix = np.arctanh(correlation_matrix)

        # 取上三角矩阵并向量化
        upper_tri_indices = np.triu_indices_from(fisher_z_matrix, k=1)
        vectorized = fisher_z_matrix[upper_tri_indices]

        # 转换为PyTorch的张量
        vectorized_tensor = torch.tensor(vectorized, dtype=torch.float32)
        return vectorized_tensor

def get_data_loader(input_folder, batch_size=16, shuffle=True):
    dataset = CorrelationDataset(input_folder)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.linear2 = nn.Linear(hidden_size, latent_size)

    def forward(self, x):  # x: bs, input_size
        x = F.relu(self.batch_norm1(self.linear1(x)))  # -> bs, hidden_size
        x = self.linear2(x)  # -> bs, latent_size
        return x

class Decoder(nn.Module):
    def __init__(self, latent_size, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_size, hidden_size)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):  # x: bs, latent_size
        x = F.relu(self.batch_norm1(self.linear1(x)))  # -> bs, hidden_size
        x = torch.sigmoid(self.linear2(x))  # -> bs, output_size
        return x

class AE(nn.Module):
    def __init__(self, input_size, output_size, latent_size, hidden_size):
        super(AE, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, latent_size)
        self.decoder = Decoder(latent_size, hidden_size, output_size)

    def forward(self, x):  # x: bs, input_size
        feat = self.encoder(x)  # feat: bs, latent_size
        re_x = self.decoder(feat)  # re_x: bs, output_size
        return re_x

#损失函数
#交叉熵，衡量各个像素原始数据与重构数据的误差
loss_BCE = torch.nn.BCELoss(reduction = 'sum')
#均方误差可作为交叉熵替代使用.衡量各个像素原始数据与重构数据的误差
loss_MSE = torch.nn.MSELoss(reduction = 'sum')

# '超参数及构造模型'
latent_size =32 #压缩后的特征维度
hidden_size = 128 #encoder和decoder中间层的维度
input_size= output_size = 1*6670 #原始图片和生成图片的维度

#训练参数
epochs = 20 
learning_rate = 1e-3 
batch_size = 16
device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_folder = 'preprocessData\FC\\train'  
test_folder = 'preprocessData\FC\\test' 
train_loader = get_data_loader(train_folder, batch_size=batch_size)
test_loader = get_data_loader(test_folder, batch_size=batch_size)

#确定模型，导入已训练模型（如有）
model = AE(input_size,output_size,latent_size,hidden_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

loss_history = {'train':[],'eval':[]}
for epoch in range(epochs):   
    #训练
    model.train()
    #每个epoch重置损失，设置进度条
    train_loss = 0
    train_nsample =0
    # t = tqdm(train_loader,desc = f'[train]epoch:{epoch}')
    for imgs in train_loader: 
        bs = imgs.shape[0]
        #获取数据
        imgs = imgs.to(device).view(bs,input_size) 
        #模型运算     
        re_imgs = model(imgs)
        #计算损失
        loss = loss_MSE(re_imgs, imgs) # 重构与原始数据的差距(也可使用loss_MSE)
        #反向传播、参数优化，重置
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        #计算平均损失，设置进度条
        train_loss += loss.item()
        train_nsample += bs
        # t.set_postfix({'loss':train_loss/train_nsample})
    #每个epoch记录总损失
    train_loss = train_loss/train_nsample
    loss_history['train'].append(train_loss)

    #测试
    model.eval()
    #每个epoch重置损失，设置进度条
    test_loss = 0
    test_nsample =0
    # e = tqdm(test_loader,desc = f'[eval]epoch:{epoch}')
    for imgs in test_loader:
        bs = imgs.shape[0]
        #获取数据
        imgs = imgs.to(device).view(bs,input_size)
        #模型运算   
        re_imgs = model(imgs)
        #计算损失
        loss = loss_MSE(re_imgs, imgs) 
        #计算平均损失，设置进度条
        test_loss += loss.item()
        test_nsample += bs
        # e.set_postfix({'loss':test_loss/test_nsample})
    #每个epoch记录总损失 
    test_loss = test_loss/test_nsample   
    loss_history['eval'].append(test_loss)

    print('epoch: {}\ttrain loss: {:.1f}\ttest loss: {:.1f}'.format(epoch, train_loss, test_loss))

#显示每个epoch的loss变化
# plt.plot(range(epoch+1),loss_history['train'])
# plt.plot(range(epoch+1),loss_history['eval'])
# plt.show()

#存储模型
modelname = 'model\AE\AE_test.pth'
torch.save(model.state_dict(),modelname)

# load model
model.load_state_dict(torch.load(modelname))
print('[INFO] Load Model complete')

pa_path = 'preprocessData\FC\\test'
pa_dataloader = get_data_loader(pa_path, batch_size=1)
model.eval()
pa_loss = []
pa_nsample =0
thresh = 2000
for imgs in pa_dataloader:
    bs = imgs.shape[0]
    #获取数据
    imgs = imgs.to(device).view(bs,input_size)
    #模型运算   
    re_imgs = model(imgs)
    #计算损失
    loss = loss_MSE(re_imgs, imgs) 
    #计算平均损失，设置进度条
    pa_loss.append(loss.item())
  
print(max(pa_loss))

