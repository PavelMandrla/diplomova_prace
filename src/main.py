import torch
import cv2
import numpy as np
from models.model import MyModel
from datasets.fdst import FDST
from torch.utils.data import DataLoader
from testing.utils import show_image, animate_range, range_real_time
import matplotlib.pyplot as plt


model_path = './save_dir/40_ckpt.tar'

dataset = FDST("../datasets/our_dataset", training=False, sequence_len=5)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MyModel(model_path)
model = model.eval().to(device)

range_real_time(model, dataset, device, 220, 700)
#animate_range(model, dataset, device)

'''
for i in range(149 - 5):
    data = dataset[i]
    #dataset.get_unnormed_item(i).show()

    image = data[0].unsqueeze(dim=0)
    print(type(data[1]))
    print(data[1])
    #show_image(image, data[1])

    image = image.to(device)
    with torch.set_grad_enabled(False):
        mu, mu_normed = model(image)
    image.cpu()

    count = torch.sum(mu).item()
    print(count)

    vis_img = mu[0, 0].cpu().numpy()
    vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-5)
    vis_img = (vis_img * 255).astype(np.uint8)
    vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_JET)
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)

    fig, axs = plt.subplots(1, 2)

    fig.set_figwidth(15)
    fig.suptitle('Result')
    fig.set_figwidth(25)
    axs[0].imshow(vis_img)
    axs[1].imshow(dataset.get_unnormed_item(i))

    plt.show()

'''




