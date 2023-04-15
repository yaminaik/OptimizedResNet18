# Optimized_ResNet18

This collection includes trials and application of a streamlined ResNet18 model architecture, which has been trained, validated, and assessed using the CIFAR-10 image classification dataset.


#Abstract
This report details our efforts to achieve maximum test accuracy for the ResNet18 model, while adhering to a constraint that limits the number of trainable parameters in the new architecture to 5 million. We conducted experiments on the ResNet18 model by making changes to its layer structure and varying the parameters and optimizers. Based on the results of our experiments, we recommend modified ResNet18 architectures that achieve the highest test accuracy while still meeting the parameter constraint. Our experiments were carried out using the CIFAR-10 image classification dataset.

#Best Models
After experimenting with several methodologies, combinations of optimizers, parameters, and layer structures, three model configurations were found to have high test accuracy.

The first architecture consists of Conv layers with a block configuration of [2,1,1,1] for 64, 128, 256 & 512 channels respectively.

The second architecture is similar to the first but with the 512 channel block removed, and instead has a [3,3,3] block configuration for 64, 128, 256 channels respectively. It also includes dropout for Convolutional layers and an extra hidden linear layer of 128 neurons.

#Respository Details


Code References:
https://github.com/kuangliu/pytorch-cifar
