# Methodology

## Experimental design

The current benchmark compares six input representations against six pretrained
CNN architectures under the same research pipeline.

```text
RGB / Grayscale / Viridis / Inferno / Turbo / HEDCM
                           ×
ResNet18 / EfficientNet-B0 / MobileNetV3-Small /
DenseNet121 / ConvNeXt-Tiny / Swin-Tiny
```

The work contains two classification settings:

- 7 classes: defect type only;
- 8 classes: Perfect + seven defect classes.

The primary comparison uses manual GT crops to isolate classification quality
from localization quality. A secondary operational track evaluates automatic
localization followed by classification.

## Important methodological rule

Train/validation/test separation should be performed by original source canvas
or production run rather than by augmented patch or individual video frame.
This reduces information leakage between splits.

## Industrial evaluation

In addition to Macro-F1, the project plans to evaluate:
- defect recall at a fixed false alarm rate;
- total pipeline latency;
- FPS;
- parameter count and FLOPs;
- FP32 vs FP16 vs INT8 execution.

## Future model

CarbonDefectNet is planned as a lightweight dual-head model combining RGB
information with HEDCM anomaly information.
