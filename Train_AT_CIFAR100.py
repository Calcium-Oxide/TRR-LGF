import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score
from Att_TR import Att_Tr

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

image_size = 32
num_classes = 100
max_epoch = 1000
learning_rate = 0.0001
batch_size = 64
rank = [5,25,35,5]
a = 20
m_channel = 512
num_layers = 2

transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),     
    transforms.RandomHorizontalFlip(),        
    transforms.ColorJitter(brightness=0.2,     
                            contrast=0.2,       
                            saturation=0.2,      
                            hue=0.1),                      
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         #std=[0.229, 0.224, 0.225])
])

transform2 = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 修改数据集路径
train_dataset = datasets.ImageFolder(root='/home/yaohu-uestc/Tensor_Regression/Tensor_Regression/Data/Dataset/CIFAR100/cifar100-32/train', transform=transform)
test_dataset = datasets.ImageFolder(root='/home/yaohu-uestc/Tensor_Regression/Tensor_Regression/Data/Dataset/CIFAR100/cifar100-32/test', transform=transform2)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

in_shape = torch.randn(batch_size, m_channel, int(0.25*image_size), int(0.25*image_size))
out_shape = torch.randn(batch_size, num_classes)
model = Att_Tr(in_shape.shape, out_shape.shape, rank, num_layers, m_channel)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {total_params}")
model.to(device)

Loss = nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

best_test_acc = 0.68

# Training loop
for epoch in range(max_epoch):
    model.train()
    epoch_loss = 0
    all_preds = []
    all_targets = []

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        W, bias, regulation, y_hat = model(inputs)
        print_loss = Loss(y_hat, targets)
        loss = Loss(y_hat, targets) + a * regulation
        loss.backward()
        optimizer.step()

        epoch_loss += print_loss.item()
        preds = torch.argmax(y_hat, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    acc = accuracy_score(all_targets, all_preds)
    print(f"Epoch {epoch + 1}: Loss: {epoch_loss / len(train_loader):.4f}, ACC: {acc:.4f}")

    # Evaluation phase
    if (epoch + 1) % 5 == 0:
        model.eval()
        test_preds = []
        test_targets = []

        with torch.no_grad():
            for test_inputs, test_targets_batch in test_loader:
                test_inputs = test_inputs.to(device)
                test_targets_batch = test_targets_batch.to(device)

                _, _, _, y_hat = model(test_inputs)
                test_preds_batch = torch.argmax(y_hat, dim=1)

                test_preds.extend(test_preds_batch.cpu().numpy())
                test_targets.extend(test_targets_batch.cpu().numpy())

        test_acc = accuracy_score(test_targets, test_preds)
        print(f"Test result {epoch + 1}: ACC: {test_acc:.4f}")

        # Save the model if the test accuracy improves
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            model_save_path = '/home/yaohu-uestc/Tensor_Regression/Tensor_Regression/Output/model/CIFAR100/CATR/best_model.pth'
            torch.save(model.state_dict(), model_save_path)
            print(f'Model saved at epoch {epoch + 1} to {model_save_path}')

        # Log the results
        with open('/home/yaohu-uestc/Tensor_Regression/Tensor_Regression/Output/result/CATR_C.txt', 'a') as f:
            f.write(f"Epoch: {epoch + 1}, Test ACC: {test_acc:.4f}\n")

    # Adjust learning rate
    if (epoch + 1) % 20 == 0:
        a = min(a * 1.5,20)
        learning_rate = max(learning_rate * 0.75, 1e-7)
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate