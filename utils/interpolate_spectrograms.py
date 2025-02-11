import torch
import numpy as np

# Function for latent interpolation
def interpolate_spectrograms(img1, img2, model, steps=10, device="cuda"):
    img1, img2 = img1.unsqueeze(0).to(device), img2.unsqueeze(0).to(device)
    mu1, logvar1 = model.encode(img1)
    mu2, logvar2 = model.encode(img2)

    z1 = model.reparameterize(mu1, logvar1)
    z2 = model.reparameterize(mu2, logvar2)

    interpolated_imgs = []
    for alpha in np.linspace(0, 1, steps):
        z_interp = (1 - alpha) * z1 + alpha * z2
        img_interp = model.decode(z_interp).detach().cpu()
        interpolated_imgs.append(img_interp)

    return torch.cat(interpolated_imgs, dim=0)


# How to use
# Loading two random images
# img1, _ = dataset[0]
# img2, _ = dataset[1]

# # Latent interpolation
# interpolated_images = interpolate_spectrograms(img1, img2, vae, steps=10)

# # Plot of interpolated spectrograms
# fig, ax = plt.subplots(1, 10, figsize=(20, 2))
# for i, img in enumerate(interpolated_images):
#     ax[i].imshow(img.permute(1, 2, 0).numpy())
#     ax[i].axis("off")
# plt.show()
