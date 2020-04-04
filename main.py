from store_images_from_video import store_images_from_video
import os
from speed_detector_dataset import SpeedDetectorDataset
import torchvision
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from cnn_model import SpeedChallengeModel, train

def visualize_images(trainloader):
    images, labels = next(iter(trainloader))

    print(labels)

    # plt.imshow(torchvision.utils.make_grid(images))
    # plt.show()    
    images = images.numpy()
    img_0_0 = images[0, 0, :, :] # First image, 1st channel, all pixels
    plt.imshow(img_0_0)
    plt.show()


if __name__ == "__main__":
    input_folder = './Data/train'
    output_folder = './Data/output'
    video_filename = 'train.mp4'
    text_filename = os.path.join(input_folder, 'train.txt')
    # store_images_from_video(os.path.join(input_folder, video_filename), output_folder)
    print('Images fetched from video:'+video_filename)
    
    trainset = SpeedDetectorDataset(output_folder, text_filename)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,shuffle=False,num_workers=0)
    visualize_images(trainloader)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    '''Instantiate model'''
    model = SpeedChallengeModel()
    model.to(device)

    '''Define Loss'''
    criterion = nn.MSELoss()

    '''Define Optimizer'''
    learning_rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

    train(model, trainloader, criterion, optimizer, num_epochs = 1)

    print('Training completed')
