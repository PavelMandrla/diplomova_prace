import cv2
from PIL import Image
import torch
from models.model import MyModel
from torchvision.transforms import transforms
from datasets.fdst import FDST
from torch.utils.data import DataLoader

dataset = FDST("../datasets/our_dataset", training=True, sequence_len=5)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
a, b, c = next(iter(dataloader))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
model = MyModel()
model.eval().to(device)

print(model)

image = a.to(device)

#output = model(image)
#output.cpu()
#print(output.shape)

mu, mu_normed = model(image)
print(mu.shape, mu_normed.shape)
# torch.Size([1, 3, 448, 796]) -> # torch.Size([1, 512, 7, 7])

# torch.Size([5, 512, 7, 7])




