import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image

class SpeedDetectorDataset(Dataset):
  def __init__(self, folder_path, label_file):
    self.folder = folder_path
    self.label = []
    f = open(label_file, "r")
    for speed in f:
        self.label.append(speed.replace('\n',''))

  def __len__(self):
    """
    Denotes the total number of samples.
    """
    return len(self.label)

  def __getitem__(self, idx):
    """
    Generates one sample of data.
    """
    filename = os.path.join(self.folder, 'image'+str(idx)+'.jpg')
    img = Image.open(filename)
    trans = transforms.ToPILImage()
    trans1 = transforms.ToTensor()

    return trans1(img), self.label[idx]




#   @classmethod
#   def from_dataframe(cls, data, feature_cols=None, label_col=None):
#     """
#     Characterizes a Dataset for PyTorch

#     Parameters
#     ----------

#     data: pandas data frame
#       The data frame object for the input data. It must
#       contain all the continuous, categorical and the
#       output columns to be used.

#     feature_cols: List of strings
#       The names of the columns in the data.
#       These columns will be passed through the embedding
#       layers in the model.

#     label_col: string
#       The name of the output variable column in the data
#       provided.
#     """

#     n = data.shape[0]

#     if label_col:
#       y = data[label_col].astype(np.float32).values.reshape(-1, 1)
#     else:
#       y =  np.zeros((self.n, 1))

#     if feature_cols:
#       X = data[feature_cols].astype(np.float32).values
#     else:
#       X = np.zeros((self.n, 1))
#     return cls(X, y)


