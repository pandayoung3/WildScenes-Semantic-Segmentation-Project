
# train,test process cite from https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
# Deeplabv3 model cite from https://github.com/qubvel-org/segmentation_models.pytorch.git
import torch
import segmentation_models_pytorch as smp

def get_model(num_classes=21):
    model = smp.DeepLabV3(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return model, device