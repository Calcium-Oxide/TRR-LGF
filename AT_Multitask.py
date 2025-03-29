import os
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import accuracy_score
from Att_TR import Att_Tr
from PIL import Image
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.metrics import mean_absolute_error
from torch.optim.lr_scheduler import CosineAnnealingLR
from Att_TR import Att_Tr_NoD
from Att_TR import Att_MLP
from SHVIT_Multitask import SHViT
from repvit_Multitask import repvit_m0_9
from resnet20_Multitask import ResNet20

# 设置随机种子和设备
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 配置参数
image_size = 32
gender_class = 2
race_class = 5  
max_epoch = 300
learning_rate = 0.0001
batch_size = 64
rank=[5,5,15,15,5,5]
a = 20
m_channel = 512
num_layers = 2

# 定义图像变换
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomCrop(image_size, padding=4),     
    transforms.RandomHorizontalFlip(),        
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),                      
    transforms.ToTensor(),
])

transform2 = transforms.Compose([ 
    transforms.Resize((image_size, image_size)),                   
    transforms.ToTensor(),
])

# 自定义数据集
class CustomDataset(Dataset):
    def __init__(self, image_dir, excel_path, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.data = pd.read_excel(excel_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]  
        image_path = os.path.join(self.image_dir, img_name)
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # 获取年龄、性别和种族标签
        age = self.data.iloc[idx, 1]        
        gender = self.data.iloc[idx, 2]    
        race = self.data.iloc[idx, 3]       
        
        return image, age, gender, race

# 创建数据集
train_dataset = CustomDataset(
    image_dir='/home/yaohu-uestc/Tensor_Regression/Tensor_Regression/Data/Dataset/UTKFace/Train/Data',
    excel_path='/home/yaohu-uestc/Tensor_Regression/Tensor_Regression/Data/Dataset/UTKFace/Train/Excel/train_info.xlsx',
    transform=transform
)

test_dataset = CustomDataset(
    image_dir='/home/yaohu-uestc/Tensor_Regression/Tensor_Regression/Data/Dataset/UTKFace/Test/Data',
    excel_path='/home/yaohu-uestc/Tensor_Regression/Tensor_Regression/Data/Dataset/UTKFace/Test/Excel/test_info.xlsx',
    transform=transform2
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# 模型定义
in_shape = torch.randn(batch_size, m_channel, int(0.25 * image_size), int(0.25 * image_size))
out_shape = torch.randn(batch_size,1,gender_class,race_class)
#model = Att_Tr(in_shape.shape, out_shape.shape, rank , num_layers, m_channel)
#model = Att_MLP(in_shape.shape, out_shape.shape, rank, num_layers, m_channel)
#model = SHViT()
#model = repvit_m0_9()
model = ResNet20(layers=[3,3,3], age_classes=1, gender_classes=2, race_classes=5)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {total_params}")
model.to(device)

# 损失函数和优化器
#loss_age = nn.MSELoss()  
loss_age = nn.L1Loss()
loss_gender = nn.CrossEntropyLoss()
loss_race = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
# Initialize cosine annealing scheduler
scheduler = CosineAnnealingLR(optimizer, T_max=300)

# 训练模型
for epoch in range(max_epoch):
    model.train()
    epoch_loss = 0
    all_preds_gender = []
    all_preds_race = []
    all_targets_gender = []
    all_targets_race = []
    all_preds_age = []
    all_targets_age = []

    for inputs, age_targets, gender_targets, race_targets in train_loader:
        inputs = inputs.to(device)
        age_targets = age_targets.to(device).float()
        gender_targets = gender_targets.to(device).long()
        race_targets = race_targets.to(device).long()
        
        all_targets_gender.extend(gender_targets.cpu().detach().numpy())
        all_targets_race.extend(race_targets.cpu().detach().numpy())
        all_targets_age.extend(age_targets.cpu().detach().numpy())

        optimizer.zero_grad()
        #W, bias, regulation, y_hat = model(inputs)
        y_hat = model(inputs)

        # 计算损失
        loss_a = loss_age(y_hat[:, :, 0, 0].squeeze(), age_targets)  # MAE
        loss_g = loss_gender(y_hat[:, 0, :, 0], gender_targets)
        loss_r = loss_race(y_hat[:, 0, 0, :], race_targets)
        print_loss = loss_a + loss_g + loss_r
        #loss = print_loss + a * regulation
        loss = print_loss
        loss.backward()
        optimizer.step()
        
        epoch_loss += print_loss.item()
        
        preds_age = y_hat[:, :, 0, 0]
        preds_gender = torch.argmax(y_hat[:, 0, :, 0], dim=1)
        preds_race = torch.argmax(y_hat[:, 0, 0, :], dim=1)

        all_preds_gender.extend(preds_gender.cpu().detach().numpy())
        all_preds_race.extend(preds_race.cpu().detach().numpy())
        all_preds_age.extend(preds_age.cpu().detach().numpy())

    acc_gender = accuracy_score(all_targets_gender, all_preds_gender)
    acc_race = accuracy_score(all_targets_race, all_preds_race)
    mae_age = mean_absolute_error(all_targets_age, all_preds_age)  # 计算 MAE

    print(f"Epoch {epoch + 1}: Loss: {epoch_loss / len(train_loader)}, Gender ACC: {acc_gender}, Race ACC: {acc_race}, Age MAE: {mae_age}")

    if (epoch + 1) % 5 == 0:
        model.eval()
        test_preds_gender = []
        test_preds_race = []
        test_targets_gender = []
        test_targets_race = []
        test_preds_age = []
        test_targets_age = []

        with torch.no_grad():
            for inputs, age_targets, gender_targets, race_targets in test_loader:
                inputs = inputs.to(device)
                age_targets = age_targets.to(device).float()
                gender_targets = gender_targets.to(device).long()
                race_targets = race_targets.to(device).long()

                #W, bias, regulation, y_hat = model(inputs)
                y_hat = model(inputs)

                preds_age = y_hat[:, :, 0, 0]
                preds_gender = torch.argmax(y_hat[:, 0, :, 0], dim=1)
                preds_race = torch.argmax(y_hat[:, 0, 0, :], dim=1)

                test_targets_gender.extend(gender_targets.cpu().detach().numpy())
                test_targets_race.extend(race_targets.cpu().detach().numpy())
                test_targets_age.extend(age_targets.cpu().detach().numpy())

                test_preds_age.extend(preds_age.cpu().detach().numpy())
                test_preds_gender.extend(preds_gender.cpu().detach().numpy())
                test_preds_race.extend(preds_race.cpu().detach().numpy())

        # 计算测试集的准确率和 MAE
        acc_gender_test = accuracy_score(test_targets_gender, test_preds_gender)
        acc_race_test = accuracy_score(test_targets_race, test_preds_race)
        mae_age_test = mean_absolute_error(test_targets_age, test_preds_age)  # 计算 MAE

        # 保存结果到文件
        with open('/home/yaohu-uestc/Tensor_Regression/Tensor_Regression/Output/result/test_results.txt', 'a') as f:
            f.write(f"Epoch {epoch + 1}: Test Gender ACC: {acc_gender_test}, Test Race ACC: {acc_race_test}, Test Age MAE: {mae_age_test}\n")

        print(f"Test Results - Epoch {epoch + 1}: Gender ACC: {acc_gender_test}, Race ACC: {acc_race_test}, Age MAE: {mae_age_test}")

    scheduler.step()