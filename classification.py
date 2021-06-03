
'''
Problem:	Use an Autoencoder for object classification

Input:		RGB image 	(CIFAR10 dataset)
Output:	Classification of the object - accuracy of the model
dataset:	CIFAR10 - download in first execution


Results:	Output images, training summary (for tensorboard), models (weights of the best model)


		
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
import os
# Device configuration
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]='0'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = "cuda"
# Hyper-parameters
num_epochs = 150
batch_size = 64
learning_rate = 0.001

# dataset has PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# CIFAR10: 60000 32x32 color images in 10 classes, with 6000 images per class
train_dataset = torchvision.datasets.CIFAR10(root='./classification/classification_data', train=True,
                                             download=True, transform=transform)

test_dataset = torchvision.datasets.CIFAR10(root='./classification/classification_data', train=False,
                                            download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                          shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imsave(img, epoch, type='in'):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig('/classification/{}_{}.jpg'.format(epoch, type))


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()


# The model "ImageReconstruction" not working accurate - Unable to reconstruct image accurate
class ImageReconstruction(nn.Module):
    # Auto encoder for image to image reconstruction
    def __init__(self):
        super(ImageReconstruction, self).__init__()
        # input: Nx3x32x32
        ''' Layers for encoder'''
        self.conv1 = nn.Conv2d(3, 6, 5, stride=1, padding=0)    # N x 6 x 28 x 28
        self.pool1= nn.MaxPool2d(2, stride=2, return_indices=True)     # N x 6 x 14 x !4
        # Layer 2: NN and Pool
        self.conv2 = nn.Conv2d(6, 16, 5, stride=1, padding=0)   # N x 16 x 10 x 10
        self.pool2 = nn.MaxPool2d(2, stride=2, return_indices=True)     # N x 16 x 5 x 5

        # Input: N x 16 x 5 x 5
        '''Layers for decoder'''
        # Layer 1: Pool and NN
        self.unpool2 = nn.MaxUnpool2d(2, stride=2)   # N x 16 x 10 x 10
        self.conv2_T = nn.ConvTranspose2d(16, 6, 5, stride=1, padding=0)  # N x 6 x 14 x 14
        # Layer 2: Pool and NN
        self.unpool1 = nn.MaxUnpool2d(2, stride=2)  # N x 16 x 28 x 28
        self.conv1_T = nn.ConvTranspose2d(6, 3, 5, stride=1, padding=0)  # N x 3 x 32 x 32


    def forward(self, x):
        '''encoder -> pool returns indices for unpool '''
        x, ind1 = self.pool1(F.relu(self.conv1(x)))
        x, ind2 = self.pool2(F.relu(self.conv2(x)))
        y = self.unpool2(x, ind2)
        y = F.relu(self.conv2_T(y))
        y = self.unpool2(y, ind1)
        y = F.relu(self.conv1_T(y))
        y = F.sigmoid(y)

        return y



# The model "ConvNet" is working properly. 
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # -> n, 3, 32, 32
        x = self.pool(F.relu(self.conv1(x)))  # -> n, 6, 14, 14
        x = self.pool(F.relu(self.conv2(x)))  # -> n, 16, 5, 5
        x = x.view(-1, 16 * 5 * 5)  # -> n, 400
        x = F.relu(self.fc1(x))  # -> n, 120
        x = F.relu(self.fc2(x))  # -> n, 84
        x = self.fc3(x)  # -> n, 10
        return x


''' for classification '''
model = ConvNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

''' for image reconstruction '''
# model = ImageReconstruction().to(device)
# # model = ImageReconstruction().cuda()
#
# # model.load_state_dict(torch.load(
# #     "/home/mzohaib/code/kpnet/Results/Saliency Key-points/saliency_log_1653 TO 3033/chair/pointnet/debug_best_3033.pth"))
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(),
#                              lr=1e-3,
#                              weight_decay=1e-5)

n_total_steps = len(train_loader)
outputs_bank = []
for epoch in range(num_epochs):
    train_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [4, 3, 32, 32] = 4, 3, 1024
        # input_layer: 3 input channels, 6 output channels, 5 kernel size
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        # loss = criterion(outputs, images)


        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= n_total_steps

        #
        # if (i + 1) % 2000 == 0:
        #     print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')

    # print(f'Epoch:{epoch + 1}, Loss:{loss.item():.4f}')
    outputs_bank.append((epoch, images, outputs))

    ''' Computing test loss after every epoch '''
    with torch.no_grad():
        avg_loss = 0
        n_samples = 0
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            # images = images.cuda()

            outputs = model(images)
            loss = criterion(outputs, labels)
            avg_loss += loss.item()

        avg_loss /= len(test_loader)

    print('Epoch: {},    train_loss:  {}   test_loss: {}'.format(epoch,  train_loss, avg_loss))

    # print(f'Epoch:{epoch + 1}, Training Loss:{loss.item():.4f}')
    # print('Saving model states')
    PATH = './classification/models/cnn_{}.pth'.format(epoch)
    torch.save(model.state_dict(), PATH)



with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')

