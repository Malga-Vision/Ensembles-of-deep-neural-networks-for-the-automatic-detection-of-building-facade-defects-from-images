import numpy as np
import PIL
from torch.utils.data import Dataset
from torchvision.transforms import transforms

trainTransform = transforms.Compose([
  transforms.ToTensor(),
	transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
  ),
	transforms.RandomHorizontalFlip(p = 0.6),
  transforms.RandomVerticalFlip(p = 0.4),
	transforms.RandomRotation((5, 5))
])

valTransform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
  )
])

class FBD(Dataset):
  def __init__(self, data, labels=None, transform = None):
      self.data = data
      self.labels = labels
      self.transform = transform
  def __len__(self):
      return len(self.data)

  def __getitem__(self, index):
      x = self.data[index]
      x = PIL.Image.fromarray(x.astype(np.uint8))
      y = self.labels[index]


      if self.transform:
        x = self.transform(x)

      return x, y
  

class LogitsDataset(Dataset):
  def __init__(self, logits, labels=None):
      self.data = logits
      self.labels = labels
  def __len__(self):
      return len(self.data)

  def __getitem__(self, index):
      x = self.data[index]
      y = self.labels[index]

      return x, y
  


  

def count_samples_per_class(dataset, num_classes):
    class_counts = np.zeros(num_classes, dtype=int)
    for index in range(len(dataset)):
        _, label = dataset[index]
        class_counts[label] += 1
    return class_counts



def load_images_and_labels(location):
    images = np.load(location + "_images.npy")
    labels = np.load(location + "_labels.npy")

    return images, labels