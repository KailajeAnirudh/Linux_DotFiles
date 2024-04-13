# %%
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F, BCELoss
import seaborn as sns
import matplotlib.pyplot as plt

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        """Encoder layers"""
        self.linear1 = nn.Linear(196, 128)
        self.linear2 = nn.Linear(128, 16)  
        """Decoder layers"""
        self.linear3 = nn.Linear(8, 128)
        self.linear4 = nn.Linear(128, 196)

    def encode(self, x):
        layer1 = torch.tanh(self.linear1(x))
        encoder_ouputs = self.linear2(layer1)
        mu = encoder_ouputs[:, :8]
        logsigma = encoder_ouputs[:, 8:]
        return mu, logsigma

    def reparameterize(self, mu, logsigma):
        std = torch.exp(0.5*logsigma)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, latent_factor):
        decodelayer1 = torch.tanh(self.linear3(latent_factor))
        return torch.sigmoid(self.linear4(decodelayer1))

    def forward(self, x):
        mu, logsigma = self.encode(x.view(-1, 196))
        latent_factor = self.reparameterize(mu, logsigma)
        return self.decode(latent_factor), mu, logsigma
    
def loss_function(recon_x, x, mu, logsigma):
    BCE_Loss = F.binary_cross_entropy(recon_x, x.view(-1, 196), reduction='sum')
    KL_Divergence = -0.5 * torch.sum(1 + logsigma - mu**2 - logsigma.exp())
    return BCE_Loss, KL_Divergence


traindata = datasets.MNIST(root='./data', train=True, download=True)
valdataset = datasets.MNIST(root='./data', train=False, download=True)
balanced_train_data = []; balanced_train_targets = []
for i in range(10):
    labelimgs = [labelimage for labelimage, label in traindata if label == i][:1000]
    for image in labelimgs:
        image = (transforms.ToTensor()(image.resize((14,14)))>0.5).float()
        balanced_train_data.append(image)
    balanced_train_targets.append(torch.ones(len(labelimgs))*i)
balanced_train_data = torch.cat(balanced_train_data)
balanced_train_targets = torch.cat(balanced_train_targets)

valdata = []; valtargets = []
for image, label in valdataset:
    image = (transforms.ToTensor()(image.resize((14,14)))>0.5).float()
    valdata.append(image)
    valtargets.append(torch.tensor(label))
valdata = torch.cat(valdata)
valtargets = torch.tensor(valtargets)
valdataset.data = valdata
valdataset.targets = valtargets
print(balanced_train_data.shape, balanced_train_targets.shape)
print(valdata.shape, valtargets.shape)


VariationalAE = VAE()
optimizer = torch.optim.Adam(VariationalAE.parameters(), lr = 0.001, weight_decay=1e-5)
train_losses = []
BCE_losses = []
KLD_losses = []
all_val_losses = []
epochs = 50
dataloader = DataLoader(list(zip(balanced_train_data, balanced_train_targets)), 
                        batch_size=32, shuffle=True)

val_dataloader = DataLoader(valdata, batch_size=100, shuffle=True)

def validate(VariationalAE, dataloader):
    VariationalAE.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in (dataloader):
            data = batch[0]
            recon_batch, mu, logvar = VariationalAE(data)
            BCE, KLD = loss_function(recon_batch, data, mu, logvar)
            val_loss += BCE.item() + KLD.item()
    return val_loss / len(dataloader.dataset)

weight_update = 0
for epoch in range(epochs):
    VariationalAE.train()
    train_loss = 0
    for _, (data, _) in enumerate(dataloader):
        optimizer.zero_grad()
        recon_batch, mu, logvar = VariationalAE(data)
        BCE_loss, KL_Divergence = loss_function(recon_batch, data, mu, logvar)
        loss = BCE_loss + KL_Divergence
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        BCE_losses.append(BCE_loss.item())
        KLD_losses.append(KL_Divergence.item())

        weight_update += 1
        if weight_update % 100 == 0:
            val_loss = validate(VariationalAE, val_dataloader)
            all_val_losses.append(val_loss/len(val_dataloader))
    print(f'Epoch: {epoch+1}, Loss: {train_loss / len(dataloader.dataset):.3f}')
    train_losses.append(train_loss / len(dataloader.dataset))



sns.set_style("whitegrid")  # Set seaborn style

plt.figure(figsize=(10, 10))

# Plot training loss
plt.subplot(2, 1, 1)
plt.plot(train_losses)
plt.title("Training Loss for the VAE")
plt.xlabel("Epochs")
plt.ylabel("Loss")

# Plot validation loss
plt.subplot(2, 1, 2)
plt.plot(all_val_losses)
plt.title("Validation Loss for the VAE")
plt.xlabel("Weight Update")
plt.ylabel("Loss")

plt.tight_layout()
plt.savefig("Training_Validation_Loss.png")
plt.show()


import pandas as pd
import seaborn as sns
plt.figure(figsize=(10, 10))
sns.set_style("whitegrid")

plt.subplot(2, 1, 1)
plt.plot(BCE_losses, label='Reconstruction Loss', alpha=0.9)
plt.plot(pd.Series(BCE_losses).rolling(window=100).mean(), label='Moving Average')
plt.title("Reconstruction Loss")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(KLD_losses, label='KL Divergence', alpha=0.9)
plt.plot(pd.Series(KLD_losses).rolling(window=100).mean(), label='Moving Average')
plt.title("KL Divergence")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("Reconstruction_KLD_Loss.png")
plt.show()


images, labels = next(iter(dataloader))
images = images[:8]; labels = labels[:8]

generated_images, _, _ = VariationalAE(images)
images = images.detach().numpy()
generated_images = generated_images.detach().numpy()

# Plot the original and reconstructed images side by side
fig, axs = plt.subplots(4, 4, figsize=(17, 16))
for i in range(8):
    # Original images
    axs[i//4, i%4].imshow(images[i].reshape(14, 14), cmap='gray')
    axs[i//4, i%4].set_title(f'Original {int(labels[i])}')
    axs[i//4, i%4].axis('off')
    # Reconstructed images
    axs[(i+8)//4, (i+8)%4].imshow(generated_images[i].reshape(14, 14), cmap='gray')
    axs[(i+8)//4, (i+8)%4].set_title(f'Generated {int(labels[i])}')
    axs[(i+8)//4, (i+8)%4].axis('off')

plt.subplots_adjust(top=0.9)  # Adjust the top position of the subplots
plt.suptitle("Original vs Generated Images")
plt.tight_layout()
plt.savefig("Original_vs_Generated.png")
plt.show()


"""Sampling from the latent space"""
fig, axs = plt.subplots(2, 4, figsize=(16, 8))

# Sample latent_factors from a standard Gaussian distribution
latent_factors = torch.randn(8, 8)

# Run the decoder network to synthesize an image
syn_images = VariationalAE.decode(latent_factors)

# Convert the synthesized images to numpy arrays
syn_images = syn_images.detach().numpy()

# Plot the synthesized images
for i in range(8):
    axs[i//4, i%4].imshow(syn_images[i].reshape(14, 14), cmap='gray')
    axs[i//4, i%4].axis('off')

plt.suptitle("Sampling from the latent space")
plt.savefig("Sampling_from_latent_space.png")
plt.show()




