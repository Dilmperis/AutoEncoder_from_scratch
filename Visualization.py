import torch
import matplotlib.pyplot as plt
import numpy as np

def visualize_reconstructions(model, test_loader, device, num_samples=5):
    model.eval()

    imgs_list = []
    recons_list = []
    # latents_list = []

    # ---- Take a few samples from the test set ----
    with torch.no_grad():
        for imgs, _ in test_loader:
            imgs = imgs.float().to(device)

            # Get latent + reconstruction
            latents = model.encoder(imgs)
            recons = model.decoder(latents).view(-1, 1, 28, 28)

            imgs_list.append(imgs.cpu().numpy())
            recons_list.append(recons.cpu().numpy())
            # latents_list.append(latents.cpu().numpy())

            break  # Only first batch

    imgs = imgs_list[0][:num_samples]
    recons = recons_list[0][:num_samples]
    # latents = latents_list[0][:num_samples]

    # ---- Plot each sample ----
    for i in range(num_samples):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        # GT
        axes[0].imshow(imgs[i].squeeze(), cmap='gray')
        axes[0].set_title("Ground Truth")
        axes[0].axis("off")

        # Reconstruction
        axes[1].imshow(recons[i].squeeze(), cmap='gray')
        axes[1].set_title("Reconstruction")
        axes[1].axis("off")

        # # Latent vector visualization (bar chart)
        # axes[2].bar(np.arange(len(latents[i])), latents[i])
        # axes[2].set_title("Latent Space")
        # axes[2].set_xlabel("Dimension")
        # axes[2].set_ylabel("Value")

        plt.tight_layout()
        plt.show()


# Example usage AFTER training:
# visualize_reconstructions(model, test_loader, device, num_samples=5)
