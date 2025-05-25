# train,test process cite from https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
# Deeplabv3 model cite from https://github.com/qubvel-org/segmentation_models.pytorch.git
import pandas as pd
import torch
from torch.utils.data import DataLoader
from dataset import WSdataset
from train import train
from evaluate import test, get_metrics2
from utils import load_image_paths, prepare_data_dict
import matplotlib.pyplot as plt
from torch import nn

# Choose which model to import
model_type = "resnet34"  # Change to "mobilenetv2" for mobilenet_v2

if model_type == "resnet34":
    from resnet34 import get_model
else:
    from mobilenetv2 import get_model

# Load image paths
image_dir = '/root/autodl-tmp/K-03/'
img_path_list, mask_path_list = load_image_paths(image_dir)
pd_data = prepare_data_dict(img_path_list, mask_path_list)

# Create datasets and dataloaders
train_dataset = WSdataset(pd_data[:3130])
test_dataset = WSdataset(pd_data[3130:3913])

train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, drop_last=True)
test_dl = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, drop_last=True)

# Initialize model, loss function, and optimizer
model, device = get_model()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Train and test the model
epochs = 5
train_loss_list, test_loss_list = [], []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dl, model, loss_fn, optimizer, device)
    train_loss = test(train_dl, model, loss_fn, device)
    test_loss = test(test_dl, model, loss_fn, device)
    train_loss_list.append(train_loss)
    test_loss_list.append(test_loss)

# Evaluate the model
classify_result, iou_result = get_metrics2(model, test_dl, device, num_classes=21, ignore_indices=[0, 20], merge_indices={4: 4, 5: 4, 10: 10, 11: 10, 12: 10, 13: 10})
print(classify_result)
print(iou_result)

# Plot the loss
plt.plot(train_loss_list, label='train_loss')
plt.plot(test_loss_list, label='test_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()