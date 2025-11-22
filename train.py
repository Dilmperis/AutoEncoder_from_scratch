import torch
import os
from PIL import Image
import numpy as np
import tqdm
from torch.utils.data import Dataset, DataLoader, random_split

from AutoEncoder import AutoEncoder_Linear
from gif_reconstruction_per_epoch import snapshot_reconstruction, create_reconstruction_progress_video
# Pick 1 sample to track over time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomDataset(Dataset):
    def __init__(self, folder_path):
        self.images = []
        self.labels = []
        self.folder_path = folder_path

        for file in tqdm.tqdm(os.listdir(folder_path), desc="Loading images"):
            if file.endswith(".png"):
                # Extract label from filename
                label = int(file.split("_label_")[1].split(".")[0])

                img_path = os.path.join(folder_path, file)
                img = Image.open(img_path)

                img_array = np.array(img, dtype=np.float32) / 255.0
                # print(f'image max value: {img_array.max()} & min value: {img_array.min()}')
                # Store image + label
                self.images.append(img_array)
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.images[index]
        label = self.labels[index]

        # Convert to CHW format for PyTorch: (1, 28, 28)
        img = torch.tensor(img).unsqueeze(0)

        return img, label


# Usage:
path_folder = "mnist_images"
dataset = CustomDataset(path_folder)

track_sample, _ = dataset[100]  # Pick a sample for tracking
print('dataset length:', len(dataset))

# ---- split dataset ----
train_size = int(0.8 * len(dataset))
val_size   = int(0.1 * len(dataset))
test_size  = len(dataset) - train_size - val_size

train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_set, batch_size=64)
test_loader  = DataLoader(test_set, batch_size=64)


# Train 
model = AutoEncoder_Linear(inupt_size=28, latent_size=64)
model = model.to(device)
criterion = torch.nn.MSELoss()
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)  
epochs = 200

# Get a snapshot of what the reconstruction looks like before training starts
# snapshot_reconstruction(model, track_sample, device)

for epoch in range(epochs):
    model.train()

    train_loss_running = 0.0
    for imgs, labels in train_loader:
        imgs = imgs.float().to(device)  
        labels = labels.to(device)

        rectostructed_imgs = model(imgs)
        loss = criterion(rectostructed_imgs, imgs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_running += loss.item()


    # Validation
    model.eval()
    val_loader_loss = 0.0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.float().to(device)
            labels = labels.to(device)

            rectostructed_imgs = model(imgs)
            loss = criterion(rectostructed_imgs, imgs)
            val_loader_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {train_loss_running:.4f}, Validation Loss: {val_loader_loss:.4f}")

    # Snapshot reconstruction for GIF
    if (epoch + 1) % 10 == 0:
        snapshot_reconstruction(model, track_sample, device)


# After training, create reconstruction progress video
create_reconstruction_progress_video(model, track_sample, device, save_path="real_time_reconstruction_per_epoch.gif")


# Testing
model.eval()
test_loader_loss = 0.0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.float().to(device)
        labels = labels.to(device)

        rectostructed_imgs = model(imgs)
        loss = criterion(rectostructed_imgs, imgs)
        test_loader_loss += loss.item()

print(f"Test Loss: {test_loader_loss:.4f}")
    

# Visualization
from Visualization import visualize_reconstructions
# visualize_reconstructions(model, test_loader, device, num_samples=5)



'''
To do:

0) Play witht the arcitecture of the Autoencoder (number of layers, number of neurons, activation functions, etc.)
1) Fix CNN Autoencoder 
2) Fix RNN Autoencoder LIKE Boris

3) Play witht the epochs and ALWAYS Visualize the results after  to see how well it is doing.
    Preferablt make a GIF that shows live the reconstruction improving (Epochs and next to it the gt and reconstruction)
4) Upload your study on Github.
5) Explain every single "concept" (e.g Normalization... why the gradient exploded) and why it is done
6) Make a gif that shows how the reconstruction improves over epochs.

'''