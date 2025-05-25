import torch
import numpy as np
from sklearn.metrics import classification_report

def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for (img_rgb, y) in dataloader:
            img_rgb, y = img_rgb.to(device), y.to(device)
            pred = model(img_rgb)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print(f"Test Error: \n , Avg loss: {test_loss:>8f} \n")
    return test_loss

def calculate_miou1(pred, target, num_classes, ignore_indices=[0, 20]):
    pred, target = pred.view(-1), target.view(-1)
    miou = []
    for cls in range(num_classes):
        if cls in ignore_indices:
            continue
        pred_inds, target_inds = pred == cls, target == cls
        intersection = (pred_inds[target_inds]).long().sum().item()
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        miou.append(1 if union == 0 else intersection / union)
    return np.mean(miou) if miou else float('nan')

def get_metrics2(model, test_dl, device, num_classes=21, ignore_indices=[0, 20], merge_indices=None):
    model.eval()
    label_list= []
    pred_list = []
    with torch.no_grad():
        for images, labels in test_dl:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            preds_l, labels_l = preds.cpu().numpy(), labels.cpu().numpy()
            if merge_indices:
                for old_idx, new_idx in merge_indices.items():
                    preds_l[preds_l == old_idx] = new_idx
                    labels_l[labels_l == old_idx] = new_idx
            label_list.append(labels_l.flatten())
            pred_list.append(preds_l.flatten())
    label_array, pred_array = np.concatenate(label_list, axis=0), np.concatenate(pred_list, axis=0)
    valid_indices = np.isin(label_array, ignore_indices, invert=True)
    label_array_filtered, pred_array_filtered = label_array[valid_indices], pred_array[valid_indices]
    classify_result = classification_report(label_array_filtered, pred_array_filtered, digits=3)
    iou_result = calculate_miou1(torch.tensor(pred_array), torch.tensor(label_array), num_classes, ignore_indices)
    print(classify_result)
    print(f"Mean IoU (ignoring indices {ignore_indices}): {iou_result}")
    return classify_result, iou_result