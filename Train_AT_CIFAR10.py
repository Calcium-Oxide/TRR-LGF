import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score
from Att_TR import Att_Tr


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    image_size = 32
    num_classes = 10  
    initial_learning_rate = 0.0001
    batch_size = 64
    rank = [5, 25, 30, 5]
    max_epoch = 500
    m_channel = 512
    num_layers = 2
    a_values = [20]

    transform = transforms.Compose([
        transforms.RandomCrop(int(image_size), padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform2)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    best_results = {}

    for a in a_values:
        print(f"Training with a = {a}")

        # 创建模型实例
        in_shape = torch.randn(batch_size, m_channel, int(0.25 * image_size), int(0.25 * image_size))
        out_shape = torch.randn(batch_size, num_classes)
        model = Att_Tr(in_shape.shape, out_shape.shape, rank, num_layers, m_channel).to(device)

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total params: {total_params}")

        Loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate, weight_decay=1e-4)


        for epoch in range(max_epoch):
            model.train()
            epoch_loss = 0
            all_preds = []
            all_targets = []

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                y_hat,regulation = model(inputs)
                print_loss = Loss(y_hat, targets)
                loss = print_loss  
                (loss+regulation).backward()
                optimizer.step()

                epoch_loss += print_loss.item()
                preds = torch.argmax(y_hat, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

            acc = accuracy_score(all_targets, all_preds)
            print(f"Epoch {epoch + 1}: Loss: {epoch_loss / len(train_loader)}, ACC: {acc}")

            # 每50个epoch调整学习率
            if (epoch + 1) % 50 == 0:
                for param_group in optimizer.param_groups:
                    new_lr = max(param_group['lr'] / 2, 1e-7)
                    param_group['lr'] = new_lr
                    print(f"Learning rate updated to: {new_lr}")

            # 每5个epoch记录测试结果
            if (epoch + 1) % 5 == 0:
                model.eval()
                test_preds = []
                test_targets = []

                with torch.no_grad():
                    for test_inputs, test_targets_batch in test_loader:
                        test_inputs = test_inputs.to(device)
                        test_targets_batch = test_targets_batch.to(device)

                        y_hat,_ = model(test_inputs)
                        test_preds_batch = torch.argmax(y_hat, dim=1)

                        test_preds.extend(test_preds_batch.cpu().numpy())
                        test_targets.extend(test_targets_batch.cpu().numpy())

                test_acc = accuracy_score(test_targets, test_preds)
                print(f"Test result {epoch + 1}: ACC: {test_acc}")
                with open('/home/yaohu-uestc/Tensor_Regression/Tensor_Regression/Output/result/CATR.txt', 'a') as f:
                     f.write(f"a: {a}, Test result {epoch + 1}: ACC: {test_acc}\n")

if __name__ == '__main__':
    main()