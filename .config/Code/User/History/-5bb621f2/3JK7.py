import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
import cv2
import pandas as pd

def load_data(download=True, train=True):
    dataset = datasets.MNIST('./', download=download, train=train)
    return dataset, dataset.data.numpy(), np.array(dataset.targets)


class DataPreprocessor:
    """
    A class to preprocess image data for machine learning models.
    """

    @staticmethod
    def _get_balanced_subset(data, targets, group_size=1000):
        """
        Returns a balanced subset of the data based on the target labels.

        Parameters:
            group_size (int): The maximum size of each group in the subset.

        Returns:
            tuple: Tuple containing the subset of data and corresponding targets.
        """
        df = pd.DataFrame({'target': targets})
        balanced_df = df.groupby('target').head(group_size)
        subset_idx = balanced_df.index
        return data[subset_idx], targets[subset_idx]

    @staticmethod
    def resize_and_binarize_images(data, targets, group_size=1000, new_size=(14, 14)):
        """
        Resizes and binarizes images.

        Parameters:
            images (numpy.ndarray): The array of images to process.
            new_size (tuple): The new size for the images.

        Returns:
            numpy.ndarray: Array of processed images.
        """
        X, y = DataPreprocessor._get_balanced_subset(data, targets, group_size)
        resized_images = np.zeros((len(X), new_size[0] * new_size[1]))
        for i, img in enumerate(X):
            resized = cv2.resize(img.reshape(28, 28), new_size).flatten()
            resized_images[i] = np.where(resized > 128, 1, 0)
        return resized_images


class CustomVariationAutoEncoder(torch.nn.Module):
    def __init__(self):
        super(CustomVariationAutoEncoder, self).__init__()
        self.encoder_1 = nn.Sequential(
           nn.Linear(196, 128), 
           nn.Tanh()
        )
        self.encoder_mean = nn.Sequential(
           nn.Linear(128, 8)
        )
        self.encoder_variance = nn.Sequential(
           nn.Linear(128, 8)
        )

        self.decoder_1 = nn.Sequential(
            nn.Linear(8, 128),
            nn.Tanh()
        )
        self.decoder_2 = nn.Sequential(
            nn.Linear(128, 196),
            nn.Sigmoid()
        )

    def encoder(self, input):
        input = self.encoder_1( input )
        mean = self.encoder_mean( input )
        standard_dev = self.encoder_variance( input )
        return mean, standard_dev

    def decoder(self, input):
        input = self.decoder_1( input )
        input = self.decoder_2( input )
        return input

    def apply_reparameterization(self, mean, standard_dev):
        standard_dev = torch.exp( standard_dev*0.5 )
        sigma = torch.randn_like(standard_dev, device=device)
        return mean + sigma*standard_dev

    def forward(self, input):
        mean, standard_dev = self.encoder( input.view(-1, 196) )
        z = self.apply_reparameterization(mean, standard_dev)
        decoded_z = self.decoder(z)
        return decoded_z, mean, standard_dev


class Loss:
   
    @staticmethod
    def compute_reconstruction_loss(input, reconstructed_input):
        return nn.BCELoss(reduction='sum')(reconstructed_input.flatten(), input.flatten())

    @staticmethod
    def compute_kl_divergence_loss(mean, standard_dev):
        return 0.5 * ( torch.exp(standard_dev) + ( mean**2 ) - 1 - standard_dev).sum()


class Trainer:
   
    @staticmethod
    def run(model: CustomVariationAutoEncoder, epochs: int):
        
        total_training_lossses, reconstruction_training_losses, kl_divergence_training_losses, reconstruction_val_losses, weight_updates = list(), list(), list(), list(), list()
        iterations = 0

        for epoch in range(epochs):
            training_loss = 0

            for batch_idx, (input, _) in enumerate(training_dataloader):
                model.train()
                input = input.to(device).float()
                optimizer.zero_grad()

                # 1st sample
                mean, standard_dev = model.encoder(input)
                reconstructed_input_1 = model.decoder( model.apply_reparameterization(mean, standard_dev) )
                reconstruction_loss_1 = Loss.compute_reconstruction_loss(input, reconstructed_input_1)

                # 2nd sample
                reconstructed_input_2 = model.decoder( model.apply_reparameterization(mean, standard_dev) )
                reconstruction_loss_2 = Loss.compute_reconstruction_loss(input, reconstructed_input_2)
                kl_divergence_loss = Loss.compute_kl_divergence_loss(mean, standard_dev)
                
                total_reconstruction_loss = ( reconstruction_loss_1 + reconstruction_loss_2 ) / 2
                
                total_loss = total_reconstruction_loss + kl_divergence_loss
                
                total_loss.backward()

                total_training_lossses.append(total_loss.item())
                reconstruction_training_losses.append(total_reconstruction_loss.item())
                kl_divergence_training_losses.append(kl_divergence_loss.item())

                optimizer.step()
                iterations += 1

                if( iterations % 100 == 0 ):
                    model.eval()

                    ( test_batch, _ ) = next(iter( validation_dataloader ) )
                    test_batch = test_batch.to(device).float()

                    # 1st sample
                    val_mean, val_standard_dev = model.encoder(input)
                    reconstructed_val_input_1 = model.decoder( model.apply_reparameterization(val_mean, val_standard_dev) )
                    reconstruction_val_loss_1 = Loss.compute_reconstruction_loss(test_batch, reconstructed_val_input_1)

                    # 2nd sample
                    reconstructed_val_input_2 = model.decoder( model.apply_reparameterization(val_mean, val_standard_dev) )
                    reconstruction_val_loss_2 = Loss.compute_reconstruction_loss(test_batch, reconstructed_val_input_2)
                    kl_divergence_val_loss = Loss.compute_kl_divergence_loss(val_mean, val_standard_dev)
                    
                    total_val_reconstruction_loss = ( reconstruction_val_loss_1 + reconstruction_val_loss_2 ) / 2
                    reconstruction_val_losses.append(total_val_reconstruction_loss.item())

                    weight_updates.append(iterations)

            print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, training_loss / len(training_dataloader.dataset)))

        torch.save({'model': model.state_dict()}, '/Users/ayushgoel/Documents/UniversityOfPennsylvania/Courses/Fall_2023/ESE-5460/Homework_5/models/vae_{}.pt'.format(epochs))

        return total_training_lossses, reconstruction_training_losses, kl_divergence_training_losses, reconstruction_val_losses, weight_updates
  

if __name__ == "__main__":
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('mps')
    optimizer_lr = 1e-3
   
    training_dataset, X_train, y_train = load_data()
    validation_dataset, X_val, y_val = load_data(train=False)

    sampled_X_train, sampled_y_train = DataPreprocessor.resize_and_binarize_images(X_train, y_train)
    sampled_X_val, sampled_y_val = DataPreprocessor.resize_and_binarize_images(X_val, y_val, group_size=10)

    sampled_X_train_tensor, sampled_y_train_tensor = torch.from_numpy(sampled_X_train), torch.from_numpy(sampled_y_train)
    training_dataloader = torch.utils.data.DataLoader(
       torch.utils.data.TensorDataset(sampled_X_train_tensor, sampled_y_train_tensor), 
       batch_size=100, 
       shuffle=True
    )

    sampled_X_val_tensor, sampled_y_val_tensor = torch.from_numpy(sampled_X_val), torch.from_numpy(sampled_y_val)
    validation_dataloader = torch.utils.data.DataLoader(
       torch.utils.data.TensorDataset(sampled_X_val_tensor, sampled_y_val_tensor), 
       batch_size=100, 
       shuffle=False
    )

    model = CustomVariationAutoEncoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=optimizer_lr)
    
    total_training_lossses, reconstruction_training_losses, kl_divergence_training_losses, reconstruction_val_losses, weight_updates = Trainer.run(model, 50)

    plt.title("Training Losses")
    plt.plot(total_training_lossses, label="Total Training Loss")
    plt.xlabel("# weight updates")
    plt.ylabel("Loss")
    plt.show()

    plt.title("ELBO Loss first term")
    plt.plot(reconstruction_training_losses, label="first loss term")
    plt.xlabel("# weight updates")
    plt.ylabel("Loss")
    plt.show()

    plt.title("ELBO Loss second term")
    plt.plot(kl_divergence_training_losses, label="second loss term")
    plt.xlabel("# weight updates")
    plt.ylabel("Loss")
    plt.show()

    

    model.eval()
    for i in range(8):
        randidx = np.random.randint(0, len(sampled_X_val_tensor))
        images = sampled_X_val_tensor[randidx].to(device).float()
        reconstructed_images, mean, standard_dev = model(images)
        
        plt.figure()
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)
        
        ax1.imshow(images.view(14, 14), cmap='gray')
        ax2.imshow(reconstructed_images.detach().cpu().view(14, 14), cmap='gray')
        ax1.set_title('Input Images')
        ax2.set_title('Reconstructed Images')
        plt.savefig('q2_reconstructed_img_'+str(i)+'.pdf')


    model.eval()
    plt.figure(figsize=(8,8))
    noise = torch.randn(6**2, 8).to(device).float()
    reconstructed_images = model.decoder(noise).cpu()
    reconstructed_grid = vutils.make_grid(reconstructed_images.reshape(reconstructed_images.shape[0], 1, 14, 14), nrow=6, normalize=True)
    plt.imshow(reconstructed_grid.numpy().transpose(1, 2, 0))
    plt.savefig('q2_reconstructed_imgs_decoding_gaussian_noise.pdf')
    plt.show()

