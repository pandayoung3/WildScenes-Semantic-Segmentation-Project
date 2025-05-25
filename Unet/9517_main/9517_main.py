import argparse
import torch.nn as nn
from model.Unet_v2 import MobileNetV2_UNet
from model.Unet_aspp_v2 import MobileNetV2_ASPP_UNet
import torch
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms
from data.SegDataloder import RandomResizeCropFlip,SegmentationDataset,Resize
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def parse_args():
    parser = argparse.ArgumentParser(description='UNet Training Arguments')
    parser.add_argument('--dataDir', default='./data', type=str, help='Directory for training data')
    parser.add_argument('--saveDir', default='./ckpt', type=str, help='Directory to save checkpoints')
    parser.add_argument('--trainList', default='./data/train.txt', type=str, help='Path to the training list file')
    parser.add_argument('--vaList', default='./data/val.txt', type=str, help='Path to the validation list file')
    parser.add_argument('--load', default='unet', type=str, help='Model to load')
    parser.add_argument('--resume', default=False, help='Flag for finetuning')
    parser.add_argument('--without_gpu', action='store_true', help='Flag to disable GPU usage')
    parser.add_argument('--nThreads', default=0, type=int, help='Number of threads for data loading')
    parser.add_argument('--train_batch', default=2, type=int, help='Batch size for training')
    parser.add_argument('--patch_size', default=16, type=int, help='Size of patches for training')
    parser.add_argument('--freeze_epoch', default=20, type=int, help='Number of epochs to freeze layers')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--lrDecay', default=25, type=int, help='Epoch interval for learning rate decay')
    parser.add_argument('--lrdecayType', default='step', choices=['keep', 'step'], type=str, help='Type of learning rate decay')
    parser.add_argument('--nEpochs', default=10, type=int, help='Number of epochs to train')
    parser.add_argument('--save_epoch', default=1, type=int, help='Interval of epochs to save checkpoint')
    parser.add_argument('--gpus', default='0', type=str, help='GPUs to use')
    parser.add_argument('--model',default='Unet_aspp', type=str, help='model')
    parser.add_argument('--phase', default='test', type=str, help='train or test')

    args = parser.parse_args()
    return args



def train_model(net, train_loader, val_loader,args):
    if args.resume:
        print('Resuming from checkpoint..')
        model_state_dict = torch.load('ckpt/best_model_unet.pth')
        net.load_state_dict(model_state_dict)
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.lrDecay, gamma=0.1)

    net.to(device)
    best_val_loss = float('inf')
    save_path=args.saveDir
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for epoch in range(args.nEpochs):
        net.train()
        running_loss = 0.0
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.nEpochs}")
        for i, (images, labels) in enumerate(train_loader_tqdm):
            images = images.to(device)
            labels = labels.to(device).long()

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 10 == 9:    # Print every 10 batches
                train_loader_tqdm.set_postfix(loss=loss.item())

        scheduler.step()

        # Validation phase
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).long()

                outputs = net(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f'Epoch [{epoch+1}/{args.nEpochs}], Validation Loss: {val_loss:.4f}')

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(save_path, 'best_model.pth')
            torch.save(net.state_dict(), best_model_path)
            print(f'Saved Best Model at {best_model_path}')

        # Save the current model after every epoch
        current_model_path = os.path.join(save_path, f'model_epoch_{epoch+1}.pth')
        torch.save(net.state_dict(), current_model_path)
        print(f'Saved Current Model at {current_model_path}')

    print('Training Complete')



def main(args):
    if args.model == 'Unet':
        net = MobileNetV2_UNet(classes=19,pretrain=True).to(device)
    else:
        net=MobileNetV2_ASPP_UNet(classes=19,pretrain=True).to(device)

    train_loader, test_loader = get_loaders(args.dataDir+'/image/',args.dataDir+'/indexLabel/',args.trainList,args.vaList,args.patch_size)
    if args.phase=='train':
        train_model(net,train_loader,test_loader,args)
    else:
        checkpoint = torch.load('ckpt/best_model_unet_aspp.pth')
        net.load_state_dict(checkpoint)
        print('avg_miou',test(net,test_loader,19))
        process_folder(net, 'data/test/images','data/test/outcome')





def calculate_miou(pred, target, num_classes):
    pred = pred.view(-1)
    target = target.view(-1)
    miou = []
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().item()
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        if union == 0:
            miou.append(1)  # If there is no ground truth, consider it a perfect match
        else:
            miou.append(intersection / union)
    return np.mean(miou)

def test(model, dataloader, num_classes):
    model.eval()
    miou_total = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            miou_total += calculate_miou(preds, labels, num_classes)

    return miou_total / len(dataloader)

def get_loaders(image_dir, label_dir, train_list, val_list, batch_size, num_workers=0):
    # Define normalization transform
    image_normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Define transformation for labels
    label_transform = transforms.Compose([
        #transforms.ToTensor(),
        lambda lbl: torch.tensor(np.array(lbl), dtype=torch.int64),
        lambda lbl: torch.squeeze(lbl, 0)
    ])

    # Data augmentation for training images
    image_train_transform = transforms.Compose([
        Resize((512, 512),False),
        image_normalize
    ])

    # Data augmentation for training labels (only resize and flip)
    label_train_transform = transforms.Compose([
        Resize((512, 512),True),
        label_transform
    ])

    # No augmentation for validation images, only normalization
    image_val_transform = transforms.Compose([
        Resize((512, 512),False),
        image_normalize
    ])

    # Resize for validation labels
    label_val_transform = transforms.Compose([
        Resize((512, 512),True),
        label_transform
    ])
    # Create datasets
    train_dataset = SegmentationDataset(image_dir, label_dir, train_list, image_transform=image_train_transform,
                                        label_transform=label_train_transform)
    val_dataset = SegmentationDataset(image_dir, label_dir, val_list, image_transform=image_val_transform,
                                      label_transform=label_val_transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


def segment_image(model,image_path, output_dir):
    image_normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_val_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        image_normalize
    ])
   
    image = Image.open(image_path).convert("RGB")
    input_image = image_val_transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(input_image)[0]
    softmax_output = F.softmax(output, dim=0)
    output_predictions = softmax_output.argmax(0).byte().cpu().numpy()

    colors = np.array([
        [128, 0, 0], [0, 128, 0], [0, 0, 128], [128, 128, 0],
        [128, 0, 128], [0, 128, 128], [64, 0, 0], [192, 0, 0],
        [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
        [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
        [0, 192, 0], [128, 192, 0]
    ], dtype=np.uint8)
    segmented_image = np.zeros((output_predictions.shape[0], output_predictions.shape[1], 3), dtype=np.uint8)
    for cls in range(18):
        segmented_image[output_predictions == cls] = colors[cls]
    segmented_image_pil = Image.fromarray(segmented_image)
    
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    original_path = os.path.join(output_dir, f"{name}_original{ext}")
    segmented_path = os.path.join(output_dir, f"{name}_segmented{ext}")

    
    image.save(original_path)
    segmented_image_pil.save(segmented_path)

    print(f"Saved {original_path} and {segmented_path}")



def process_folder(model,input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, file_name)
            segment_image(model,image_path, output_folder)


if __name__ == '__main__':
    args = parse_args()
    print(args)
    main(args)
