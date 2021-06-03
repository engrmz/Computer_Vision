
'''
Problem:	Use an ResNet18 encoder and a decoder to image to image reconstruction. It uses real object (i.e. cat, dog, etc.) instead of numbers.

Input:		RGB image 	(CIFAR10 dataset)
Output:	Image similar to input
dataset:	CIFAR10 - download in first execution


Results:	Output images, training summary (for tensorboard), models (weights of the best model)

Directory:	
		
		input: 		./classification/classification_data
		Output images:		./image_reconstruction_CIFAR/images
		training summary:	./image_reconstruction_CIFAR/train_summary
		best model:		./image_reconstruction_CIFAR/models


Helping material:

		link 1:	https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html
		link 2:	https://www.python-engineer.com/posts/pytorch-autoencoder/
		
		
Mohammad Zohaib
engr.mz@hotmail.com
Last Update:	June 3, 2021


'''


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision.models as models
from torchsummary import summary

from torchvision import datasets, transforms as T

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# dataset has PILImage images of range [0, 1].
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# CIFAR10: 60000 32x32 color images in 10 classes, with 6000 images per class
train_dataset = torchvision.datasets.CIFAR10(root='./classification/classification_data', train=True,
                                             download=True, transform=transform)

test_dataset = torchvision.datasets.CIFAR10(root='./classification/classification_data', train=False,
                                            download=True, transform=transform)


def imsave(img, epoch, type='in'):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    PATH = './image_reconstruction_CIFAR/images'
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    plt.savefig(PATH+'/{}_{}.jpg'.format(epoch, type))


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()



    model_resent50 = models.resnet50(pretrained=False)
    model = torch.nn.Sequential(*(list(model_resent50.children())[:-1]))
    print(model)

class ImageReconstruction_Resnet18(nn.Module):
    def __init__(self):
        super(ImageReconstruction_Resnet18, self).__init__()
        # Input: N x 3 x 32 x 32
        model_resent18 = models.resnet18(pretrained=False)
        self.encoder_resen =  torch.nn.Sequential(*(list(model_resent18.children())[:-1]))  # N x 512 x 1 x 1
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=1, padding=0),  # N x 256 x 04 x 04
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # N x 128 x 08 x 08
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # N x 64 x 16 x 16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # N x 32 x 32 x 32
            nn.ReLU(),

            nn.ConvTranspose2d(32, 3, 1, stride=1, padding=0),  # N x 3 x 32 x 32   <== OUTPUT
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder_resen(x)
        # x = torch.flip(x, dims=(0,))
        x = self.decoder(x)
        return x


def test(model, criterion, batch_size=64):
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                              shuffle=False)

    test_loss = 0
    with torch.no_grad():
        for i, (img, labels) in enumerate(test_loader):
            img = img.to(device)
            # for data in train_loader:
            #     img, _ = data
            recon = model(img)
            loss = criterion(recon, img)
            test_loss += loss.item()

        test_loss /= len(test_loader)

    return test_loss



def train(model, start_epoch=0, num_epochs=5, batch_size=64, learning_rate=1e-3):
    writer = SummaryWriter("image_reconstruction_CIFAR/train_summary")
    torch.manual_seed(42)
    criterion = nn.MSELoss() # mean square error loss
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=1e-5)     # <--

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True)

    # outputs = []
    for epoch in range(num_epochs + start_epoch):
        epoch += start_epoch     # start epoch from the last best model
        train_loss = 0
        for i, (img, labels) in enumerate(tqdm(train_loader)):
            img = img.to(device)
            recon = model(img)
            loss = criterion(recon, img)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()

        train_loss /= len(train_loader)


        # outputs.append((epoch, img, recon),)
        imsave(torchvision.utils.make_grid(img.cpu()), epoch, 'in')
        imsave(torchvision.utils.make_grid(recon.cpu().detach()), epoch, 'out')

        test_loss = test(model, criterion, batch_size)
        writer.add_scalar('train', train_loss, epoch+1)  # write training loss
        writer.add_scalar('valid', test_loss, epoch + 1)  # write training loss
        print('Epoch:{}, train_loss:{:.4f}, test_loss:{:.4f}'.format(epoch+1, float(train_loss),  float(test_loss)))

        PATH = './image_reconstruction_CIFAR/models/'
        if not os.path.exists(PATH):
            os.makedirs(PATH)
        if test_loss <= train_loss:
            file_name = 'cnn_best.pth'
            torch.save(model.state_dict(), PATH + file_name)
        elif epoch % 10 == 0:
            file_name = 'cnn_{}.pth'.format(epoch)
            torch.save(model.state_dict(), PATH + file_name)

    # return outputs


''' Find last saved model -> start of epoch'''
def find_epoch():
    if os.path.isdir("./image_reconstruction_CIFAR/models"):
        lst = os.listdir("./image_reconstruction_CIFAR/models")
        if len(lst) != 0:
            x = [a.split('.')[0].split('_')[1] for a in lst]
            y = [int(a) for a in x if not a.endswith('best')]
            y.sort()
            return y[-1]
        else:
            return 0
    else:
        return 0



def main():
    # Hyper-parameters
    max_epochs = 1000
    batch_size = 256
    learning_rate = 0.001

    model = ImageReconstruction_Resnet18().to(device)


    start_epoch = find_epoch()
    print("Epoch starts from {}".format(start_epoch))

    best_model_path = "./image_reconstruction_CIFAR/models/cnn_best.pth"
    if os.path.isfile(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print("Loading best model from: {}".format(best_model_path))
    else:
        print("Best model NOT found. Starting from the beginning")

    train(model, start_epoch, num_epochs=max_epochs, batch_size=batch_size, learning_rate= learning_rate)


if __name__ == '__main__':
    main()
