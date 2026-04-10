# PyTorch Image Classifier: Transfer Learning with ResNet18

A binary image classification system (Dogs vs. Cats) built using PyTorch. This project demonstrates the practical application of **Transfer Learning** and **Fine-Tuning** on a deep residual network architecture.

## Overview
The goal of this project was to leverage a pre-trained ResNet18 model to achieve high accuracy with a limited dataset and minimal training time. By using weights optimized on the ImageNet dataset, the model acts as a robust feature extractor.

## Architecture & Methodology
*   **Base Model**: ResNet18 (Pre-trained on ImageNet).
*   **Feature Extraction**: Initial convolutional layers are frozen (`requires_grad = False`) to preserve learned spatial hierarchies.
*   **Custom Head**: The final fully connected (FC) layer is replaced with a new linear layer tailored for 2-class classification.
*   **Optimization**: Adam optimizer targeting only the new FC layer parameters.
*   **Preprocessing**: Images are resized to 224x224 and normalized using standard ImageNet statistics.

## Project Structure
*   `model.py` – Model definition and surgery (freezing/replacing layers).
*   `data_setup.py` – Data loading pipeline and torchvision transforms.
*   `train.py` – Main training loop and evaluation logic.
*   `predict_pets.py` – Standalone inference script for testing on single images.

## Performance
*   **Test Accuracy**: ~97.14% after 5 epochs.
*   **Loss Function**: Cross-Entropy Loss.
*   **Device**: Automated fallback between CUDA (GPU) and CPU.

## Key Takeaways & Lessons Learned

During the development of this project, I focused on several core Deep Learning concepts:

1. **Efficiency of Transfer Learning**: I observed how starting with pre-trained weights (ResNet18) allows for reaching high accuracy (>95%) in just a few minutes, which would be impossible when training from scratch on a small dataset.
2. **Layer Freezing**: I learned how to "freeze" the backbone of a neural network to act as a fixed feature extractor, only updating the gradient for the final classification head.
3. **Data Normalization**: I implemented ImageNet-specific normalization, which is critical for Transfer Learning to ensure input data distribution matches what the model was originally trained on.
4. **Inference Pipeline**: I built a separate inference script to demonstrate how a trained model can be deployed for real-world predictions on unseen images.
