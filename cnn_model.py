import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets

'''Define class'''
class SpeedChallengeModel(nn.Module):
    def __init__(self):
        super(SpeedChallengeModel, self).__init__()
        
        #Input : 300 x 300 x 3

        # Convolution 1 : 296 x 296 x 16
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        # Max pool 1 : 148 x 148 x 16
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
     
        # Convolution 2 : 144 x 144 x 32
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        # Max pool 2 : 72 x 72 x 32
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
     
        # Convolution 3 : 68 x 68 x 64
        self.cnn3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=0)
        self.relu3 = nn.ReLU()
        # Max pool 3 : 34 x 34 x 64
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
     
        # Fully connected 1 (readout)
        self.fc1 = nn.Linear(64*34*34, 1)
        self.relu4 = nn.ReLU() 
    
    def forward(self, x):
      
        # Convolution 1
        out = self.maxpool1(self.relu1(self.cnn1(x)))
        
        # Convolution 2
        out = self.maxpool2(self.relu2(self.cnn2(out)))
        
        # Convolution 3
        out = self.maxpool3(self.relu3(self.cnn3(out)))
        
        out = out.view(out.size(0), -1)

        # Linear function (readout)
        out = self.fc1(out)
        out = self.relu4(out)
        
        return out

'''Train model'''
def train(model, train_loader, criterion, optimizer, num_epochs = 1):
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            print(f'i = {i}.................')
            if (i > 0):
                break
            
            # Load images
            images = images.requires_grad_().to(device)
            labels = labels.to(device)
                
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
                
            # Forward pass to get output/logits
            outputs = model(images)
                
            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, labels)
                
            # Getting gradients w.r.t. parameters
            loss.backward()
                
            # Updating parameters
            optimizer.step()
         
        print('Epoch: {}. Loss: {}.'.format(epoch+1, loss.item()))