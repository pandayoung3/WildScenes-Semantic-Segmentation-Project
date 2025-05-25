#!/usr/bin/env python
# coding: utf-8
# train,test process cite from https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
# Deeplabv3 model cite from https://github.com/qubvel-org/segmentation_models.pytorch.git
import os
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
# import pytorch_lightning as pl


import torchvision.transforms as transforms
from PIL import Image
import segmentation_models_pytorch as smp

from pprint import pprint
from torch.utils.data import DataLoader

from PIL import Image
import pandas as pd
from pathlib import Path
import cv2

from sklearn.metrics import confusion_matrix
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from sklearn.metrics import classification_report


eg_image_path_mask = eg_image_path_mask = '/root/autodl-tmp/K-03/indexLabel/1639434794-787983411.png'
Image.open(eg_image_path_mask)

# define the #classes, 15 need to be evaluated.
num_classes = 21

one_mask_array = np.array(Image.open(eg_image_path_mask))

np.unique(one_mask_array,return_counts=True)




eg_image_path = '/root/autodl-tmp/K-03/image/1639434794-787983411.png'


from pathlib import Path
import pandas as pd
image_dir = '/root/autodl-tmp/K-03/'
p = Path(image_dir)
all_path_list = [str(item) for item in list(p.rglob('*.png'))]
# print(all_path_list)
img_path_list = [item for item in all_path_list if item.split('/')[4] == 'image']
# print(img_path_list)
mask_path_list = [item for item in all_path_list if item.split('/')[4] == 'indexLabel']
img_path_list.sort()
mask_path_list.sort()

place_name_list = [item.split('/')[3] for item in img_path_list]

# img_path_list[:5],mask_path_list[:5], place_name_list[:5]



np.array(Image.open(mask_path_list[0]))


data_dict = {'ori_img':img_path_list,"mask_img":mask_path_list}
pd.set_option('display.max_colwidth', 500)
pd_data = pd.DataFrame(data_dict)
pd_data


class WSdataset(torch.utils.data.Dataset):
    def __init__(self, df_data):

        self.ori_img_path_list = list(df_data['ori_img'])
        self.mask_img_path_list = list(df_data['mask_img'])

    def __len__(self):
        return len(self.ori_img_path_list)

    def __getitem__(self, idx):

        image_path = self.ori_img_path_list[idx]
        mask_path = self.mask_img_path_list[idx]
        # print(mask_path)

        image = np.array(Image.open(image_path).resize((256, 256)).convert("RGB"))
        mask = np.array(Image.open(mask_path).resize((256, 256))).astype('int32')

        # convert to other format HWC -> CHW
        image = np.moveaxis(image, -1, 0)
        # mask = np.expand_dims(mask, 0)
        image = torch.tensor(image).float()
        mask = torch.tensor(mask).long()

        return image,mask




train_dataset = WSdataset(pd_data[ :3130])
test_dataset = WSdataset(pd_data[3130:3913 ])



n_cpu = os.cpu_count()
# print(n_cpu)
train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, drop_last=True)
test_dl = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, drop_last=True)



for img,mask in train_dl:
  print(img.shape)
  print(mask.shape)
  print(np.unique(mask,return_counts=True))
  break




# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")


# Define the training funtions
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# Define the testing funtion
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for (img_rgb, y) in dataloader:
            img_rgb = img_rgb.to(device)
            y = y.to(device)
            pred = model(img_rgb)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print(f"Test Error: \n , Avg loss: {test_loss:>8f} \n")
    return test_loss



# Define the evaluation MioU methods
def calculate_miou1(pred, target, num_classes, ignore_indices=[0,20]):
    pred = pred.view(-1)
    target = target.view(-1)
    miou = []
    for cls in range(num_classes):
        if cls in ignore_indices:
            continue
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().item()
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        if union == 0:
            miou.append(1)  # If there is no ground truth, consider it a perfect match
        else:
            miou.append(intersection / union)
    return np.mean(miou) if miou else float('nan')  # Return NaN if all classes are ignored
# use evaluation metrix.
def get_metrics2(model, test_dl, num_classes=21, ignore_indices=[0,20], merge_indices={4: 4, 5: 4,10:10,11:10,10:10,12:10,10:10,13:10}):
    model.eval()
    label_list = []
    pred_list = []
    with torch.no_grad():
        for images, labels in test_dl:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Get predictions
            _, preds = torch.max(outputs, 1)
            preds_l = preds.cpu().numpy()
            labels_l = labels.cpu().numpy()

            # Map indices for merging classes
            for old_idx, new_idx in merge_indices.items():
                preds_l[preds_l == old_idx] = new_idx
                labels_l[labels_l == old_idx] = new_idx

            label_list.append(labels_l.flatten())
            pred_list.append(preds_l.flatten())

    # Concatenate all predictions and labels
    label_array = np.concatenate(label_list, axis=0)
    pred_array = np.concatenate(pred_list, axis=0)

    # Filter out ignore indices from classification report
    valid_indices = np.isin(label_array, ignore_indices, invert=True)
    label_array_filtered = label_array[valid_indices]
    pred_array_filtered = pred_array[valid_indices]

    classify_result = classification_report(label_array_filtered, pred_array_filtered, digits=3)
    print(classify_result)

    # Compute IoU ignoring specified indices
    iou_result = calculate_miou1(torch.tensor(pred_array), torch.tensor(label_array), num_classes, ignore_indices)
    print(f"Mean IoU (ignoring indices {ignore_indices}): {iou_result}")

    return classify_result, iou_result



# # DeepLabv3 Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 21
model = smp.DeepLabV3(
    encoder_name="mobilenet_v2",
    encoder_weights="imagenet",
    in_channels=3,
    classes=num_classes,
)
model = model.to(device)


print(mask.shape)
print(model(img.to(device)).shape)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
# Train loop
epochs = 5
train_loss_list = []
test_loss_list = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dl, model, loss_fn, optimizer)
    train_loss = test(train_dl, model, loss_fn)
    test_loss = test(test_dl, model, loss_fn)
    train_loss_list.append(train_loss)
    test_loss_list.append(test_loss)
print("Training finished")


plt.plot(train_loss_list, label='train_loss')
plt.plot(test_loss_list, label = 'test_loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
# plt.ylim([0.5, 1])    
plt.legend(loc='upper right')


device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
classify_result2, iou_result2 = get_metrics2(model, test_dl, num_classes=21, ignore_indices=[0,20], merge_indices={ 4: 4, 5: 4,10:10,11:10,10:10,12:10,10:10,13:10})

print(classify_result2)
print(iou_result2)

# def plot_comparison(original_img, original_mask, predicted_mask, save_path=None):
#     fig, axs = plt.subplots(1, 3, figsize=(15, 5))
#     axs[0].imshow(original_img)
#     axs[0].set_title('Original Image')
#     axs[1].imshow(original_mask, cmap='gray')
#     axs[1].set_title('Original Mask')
#     axs[2].imshow(predicted_mask, cmap='gray')
#     axs[2].set_title('Predicted Mask')
#     plt.show()
    
#     if save_path:
#         fig.savefig(save_path)
#         plt.close(fig)  # save images
# ## the exact image 1639434794-787983411.png
# image_path = '/root/autodl-tmp/K-03/image/1639434794-787983411.png'
# mask_path = '/root/autodl-tmp/K-03/indexLabel/1639434794-787983411.png'

# image = np.array(Image.open(image_path).resize((256, 256)).convert("RGB"))
# mask = np.array(Image.open(mask_path).resize((256, 256))).astype('int32')

# image = np.moveaxis(image, -1, 0)
# image = torch.tensor(image).float().unsqueeze(0).to(device)

# model.eval()
# with torch.no_grad():
#     pred = model(image)
#     pred = torch.argmax(pred, dim=1).squeeze().cpu().numpy()

# image_for_plot = np.moveaxis(image.squeeze().cpu().numpy(), 0, -1).astype('uint8')

# # Save comparison figures.
# output_path = '/content/autodl-tmp/output_comparison.png'
# plot_comparison(image_for_plot, mask, pred, save_path=output_path)