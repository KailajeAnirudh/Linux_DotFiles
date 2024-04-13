
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F

ValTransform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.where(x > 0.5, torch.tensor(1.0), torch.tensor(0.0))),
])


mnst = datasets.MNIST(root='./data', train=True, download=True)
val = datasets.MNIST(root='./data', train=False, download=True, ValTransform=ValTransform)


new_dataset_images = []
new_dataset_labels = []

for i in range(10):
    images = [img for img, label in mnist if label == i][:1000]
    
    for img in images:
        img = img.resize((14, 14))
        img = transforms.ToTensor()(img)
        img = torch.where(img > 0.5, torch.tensor(1.0), torch.tensor(0.0))
        
        new_dataset_images.append(img)
        new_dataset_labels.append(i)

# Convert the lists to PyTorch tensors
new_dataset_images = torch.stack(new_dataset_images)
new_dataset_labels = torch.tensor(new_dataset_labels)


dataloader = DataLoader(list(zip(new_dataset_images, new_dataset_labels)), batch_size=32, shuffle=True)


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(196, 128)
        self.fc2 = nn.Linear(128, 16)  # Outputs for both mean and logvar
        # Decoder
        self.fc3 = nn.Linear(8, 128)
        self.fc4 = nn.Linear(128, 196)

    def encode(self, x):
        h1 = torch.tanh(self.fc1(x))
        return self.fc2(h1).chunk(2, dim=1)  # Split the output into mu and logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = torch.tanh(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 196))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 196), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE, KLD


autoencoder = Autoencoder()
optimizer = torch.optim.Adam(autoencoder.parameters())
train_losses = []
BCE_losses = []
KLD_losses = []

for epoch in range(1, 51):
    autoencoder.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(dataloader):
        optimizer.zero_grad()
        recon_batch, mu, logvar = autoencoder(data)
        BCE, KLD = loss_function(recon_batch, data, mu, logvar)
        loss = BCE + KLD
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        BCE_losses.append(BCE.item())
        KLD_losses.append(KLD.item())
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(dataloader.dataset)))
    train_losses.append(train_loss / len(dataloader.dataset))


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(train_losses)
plt.title("Training Loss")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.show()


plt.figure(figsize=(10, 5))
plt.plot(BCE_losses, label='Reconstruction Loss')
plt.plot(KLD_losses, label='KL Divergence')
plt.title("Training Losses")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.legend()
plt.show()


images, labels = next(iter(dataloader))
images = images[:8]
labels = labels[:8]

# Run the images through the autoencoder
recon_images, _, _ = autoencoder(images)

# Convert the images to numpy arrays
images = images.detach().numpy()
recon_images = recon_images.detach().numpy()

# Plot the original and reconstructed images side by side
fig, axs = plt.subplots(2, 8, figsize=(16, 4))

for i in range(8):
    # Original images
    axs[0, i].imshow(images[i].reshape(14, 14), cmap='gray')
    axs[0, i].set_title(f'Original {labels[i]}')
    axs[0, i].axis('off')
    # Reconstructed images
    axs[1, i].imshow(recon_images[i].reshape(14, 14), cmap='gray')
    axs[1, i].set_title(f'Reconstructed {labels[i]}')
    axs[1, i].axis('off')

plt.tight_layout()
plt.show()


# Sample z from a standard Gaussian distribution
z = torch.randn(8, 8)

# Run the decoder network to synthesize an image
syn_images = autoencoder.decode(z)

# Convert the synthesized images to numpy arrays
syn_images = syn_images.detach().numpy()

# Plot the synthesized images
fig, axs = plt.subplots(1, 8, figsize=(16, 2))

for i in range(8):
    axs[i].imshow(syn_images[i].reshape(14, 14), cmap='gray')
    axs[i].axis('off')

plt.show()


val_dataloader = DataLoader(val, batch_size=100, shuffle=True)

def validate(autoencoder, dataloader):
    autoencoder.eval()
    val_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(dataloader):
            recon_batch, mu, logvar = autoencoder(data)
            BCE, KLD = loss_function(recon_batch, data, mu, logvar)
            val_loss += BCE.item() + KLD.item()
    return val_loss / len(dataloader.dataset)

all_train_losses = []
all_val_losses = []
update_count = 0

for epoch in range(1, 51):
    for batch_idx, (data, _) in enumerate(dataloader):
        optimizer.zero_grad()
        recon_batch, mu, logvar = autoencoder(data)
        BCE, KLD = loss_function(recon_batch, data, mu, logvar)
        loss = BCE + KLD
        loss.backward()
        optimizer.step()
        
        all_train_losses.append(BCE.item()/len(dataloader))
        
        update_count += 1
        if update_count % 100 == 0:
            val_loss = validate(autoencoder, val_dataloader)
            all_val_losses.append(val_loss/len(val_dataloader))


plt.figure(figsize=(10, 5))
plt.plot(all_train_losses[:-50], label='Train Loss')  # Exclude the last value
plt.plot(range(0, len(all_train_losses[:-50]), 100), all_val_losses, label='Validation Loss')
plt.title("Training and Validation Losses")
plt.xlabel("Weight Update")
plt.ylabel("Loss")
plt.legend()
plt.show()





