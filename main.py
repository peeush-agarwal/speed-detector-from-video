from store_images_from_video import store_images_from_video
from speed_detector_dataset import SpeedDetectorDataset
from cnn_model import SpeedChallengeModel, train

import os
import time

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision


def visualize_images(trainloader):
    images, labels = next(iter(trainloader))

    print(labels)
    # print(images.shape)

    images = torchvision.utils.make_grid(images)
    plt.imshow(images.numpy().transpose((1,2,0)))
    plt.show()    
    # images = images.numpy()
    # img_0_0 = images[0, 0, :, :] # First image, 1st channel, all pixels
    # plt.imshow(img_0_0)
    # plt.show()

def get_device():
    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        a = torch.zeros((2,2),device=device)
        b = torch.ones((2,2),device=device)
        c = a+b
        return device
    except:
        return torch.device("cpu")
    


if __name__ == "__main__":
    input_folder = './Data/train'
    images_folder = os.path.join(input_folder, 'images')
    video_filename = 'train.mp4'
    text_filename = os.path.join(input_folder, 'train.txt')
    # store_images_from_video(os.path.join(input_folder, video_filename), images_folder)
    # print('Images fetched from video:'+video_filename)
    
    trainset = SpeedDetectorDataset(images_folder, text_filename)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,shuffle=False,num_workers=0)
    # visualize_images(trainloader)
    
    device = get_device()
    # print(device)
    
    '''Instantiate model'''
    model = SpeedChallengeModel()
    model.to(device)

    '''Define Loss'''
    criterion = nn.MSELoss()

    '''Define Optimizer'''
    learning_rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

    since = time.time()
    print('Training initiated')
    train(model, trainloader, criterion, optimizer, device, num_epochs = 5)
    print(f'Training completed in {time.time() - since}')
