# mobilenetv2: customizing convolutional nets with pre-trained models

## Objectives

Use a mobilenetv2 model, freezing its layers, and putting convolutional layers at the end of network

## Process

1. Load MobileNetV2 model.
2. Freeze all its 154 layers.
3. All a global_average_pooling2d layer.
4. Add two dense layers. Note that the number of neurons on the last layer must be the same as the number of classes
in the classifier.
5. Train the model and save each epoch with callbacks.


## Results

Accuracy of the trained model at each epoch.
![Accuracy of the trained model](./results/acc.PNG)

Loss of the trained model at each epoch.
![Loss of the trained model](./results/losss.PNG)

Showing the result of the last runs:
![Run details](./results/runs.PNG)
