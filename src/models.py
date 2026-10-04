"""CNN model factory used in the CarbonDefect benchmark."""

import torch.nn as nn
from torchvision.models import (
    resnet18, ResNet18_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
    densenet121, DenseNet121_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights,
    swin_t, Swin_T_Weights,
)

MODEL_NAMES = (
    "ResNet18",
    "EfficientNet_B0",
    "MobileNetV3_Small",
    "DenseNet121",
    "ConvNeXt_Tiny",
    "Swin_Tiny",
)


def replace_classifier(model, model_name, num_classes):
    if model_name == "ResNet18":
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "EfficientNet_B0":
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "MobileNetV3_Small":
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif model_name == "DenseNet121":
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == "ConvNeXt_Tiny":
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    elif model_name == "Swin_Tiny":
        model.head = nn.Linear(model.head.in_features, num_classes)
    else:
        raise ValueError(model_name)
    return model


def build_model(model_name, num_classes, pretrained=True):
    if model_name == "ResNet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    elif model_name == "EfficientNet_B0":
        model = efficientnet_b0(
            weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None
        )
    elif model_name == "MobileNetV3_Small":
        model = mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        )
    elif model_name == "DenseNet121":
        model = densenet121(
            weights=DenseNet121_Weights.DEFAULT if pretrained else None
        )
    elif model_name == "ConvNeXt_Tiny":
        model = convnext_tiny(
            weights=ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        )
    elif model_name == "Swin_Tiny":
        model = swin_t(weights=Swin_T_Weights.DEFAULT if pretrained else None)
    else:
        raise ValueError(model_name)

    return replace_classifier(model, model_name, num_classes)
