# AutoEncoder Implementation for Image Reconstruction from scratch

This repo constructs various architectures of AutoEncoders Models for Image Reconstruction.

## Visaulization of the image reconstruction across the eopchs during training:

<p align="center">
  <img src="real_time_reconstruction_per_epoch.gif" alt="Reconstruction GIF" />
</p>

## What are Autoencoders?
<p align='center'><img width="772" height="470" alt="image" src="https://github.com/user-attachments/assets/8a8f7759-ac76-4beb-b053-881861c3feac" /></p>

An autoencoder is a neural network that learns to compress an image into a smaller representation (called the latent space) and then reconstruct the original image from that compressed version.
It has two parts:

1. Encoder

        Takes the input image, gradually reduces its dimensionality, learns a compact
        encoded vector (latent code) that captures the most important features

3. Decoder

        Takes the latent code, reconstructs the image back to its original shape.

The goal is the trained network to minimize the difference between the **input image** and the reconstructed one.

What autoencoders are used for:
- Denoising images
- Data Compression
- Anomaly detection
- Feature learning for usage in other tasks.

