'''
Problem:	Use an Autoencoder to image to image reconstruction. It uses real object (i.e. cat, dog, etc.) instead of numbers

Input:		RGB image 	(MNIST dataset)
Output:	Image similar to input
dataset:	MNIST - download in first execution


Results:	Input and corresponding output images
Directory:	
		
		input: 		./image_reconstruction/mnist_data
		Output images:		./image_reconstruction/images/


Helping material:

		link 1:	https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html
		link 2:	https://www.python-engineer.com/posts/pytorch-autoencoder/
		
		
Mohammad Zohaib
engr.mz@hotmail.com
Last Update:	June 3, 2021


'''



import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# dataset has PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1]
# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



mnist_data = datasets.MNIST('./image_reconstruction/mnist_data', train=True, download=True, transform=transforms.ToTensor())
mnist_data = list(mnist_data)[:4096]


def imsave(img, epoch, type='in'):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig('image_reconstruction/images/{}_{}.jpg'.format(epoch, type))


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()



class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential( # like the Composition layer you built
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train(model, num_epochs=5, batch_size=64, learning_rate=1e-3):
    torch.manual_seed(42)
    criterion = nn.MSELoss() # mean square error loss
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=1e-5) # <--
    train_loader = torch.utils.data.DataLoader(mnist_data,
                                               batch_size=batch_size,
                                               shuffle=True)
    outputs = []
    for epoch in range(num_epochs):
        for data in train_loader:
            img, _ = data
            recon = model(img)
            loss = criterion(recon, img)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print('Epoch:{}, Loss:{:.4f}'.format(epoch+1, float(loss)))
        outputs.append((epoch, img, recon),)
        imsave(torchvision.utils.make_grid(img.cpu()), epoch, 'in')
        imsave(torchvision.utils.make_grid(recon.cpu().detach()), epoch, 'out')

    return outputs


def main():
    # Hyper-parameters
    max_epochs = 150
    batch_size = 64
    learning_rate = 0.001

    model = Autoencoder()
    outputs = train(model, num_epochs=max_epochs, batch_size=batch_size, learning_rate= learning_rate)

    for k in range(0, max_epochs, 5):
        plt.figure(figsize=(9, 2))
        imgs = outputs[k][1].detach().numpy()
        recon = outputs[k][2].detach().numpy()
        for i, item in enumerate(imgs):
            if i >= 9: break
            plt.subplot(2, 9, i + 1)
            plt.imshow(item[0])

        for i, item in enumerate(recon):
            if i >= 9:
                break
            plt.subplot(2, 9, 9 + i + 1)
            plt.imshow(item[0])


if __name__ == '__main__':
    main()
