import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import torch
import cv2
import numpy as np
import timeit
from timeit import default_timer as timer


def show_image(input_tensor, head_pos=None):
    fig, axs = plt.subplots(1, input_tensor.shape[1])
    fig.suptitle('Image sequence')
    fig.set_figwidth(25)

    for i, ax in enumerate(axs):
        ax.imshow(input_tensor[0][i].permute(1, 2, 0))
        if i == len(axs) - 1 and head_pos is not None:
            for point in head_pos:
                ax.plot(point[0], point[1], "or")
    plt.show()


def colorize_heat_map(mu):
    vis_img = mu[0, 0].cpu().numpy()
    vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-5)
    vis_img = (vis_img * 255).astype(np.uint8)
    vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_JET)
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)[:, :, ::-1].copy()
    return vis_img


def animate_range(model, dataset, device):
    fig, axs = plt.subplots(1, 2)
    times = []

    def animate(i):
        print(i)
        data = dataset[i]
        image = data[0].unsqueeze(dim=0)

        start = timer()
        image = image.to(device)
        with torch.set_grad_enabled(False):
            mu, mu_normed = model(image)
        end = timer()
        times.append(end - start)

        image.cpu()
        vis_img = colorize_heat_map(mu)

        axs[0].set_title('%.2f' % (torch.sum(mu).item()), fontsize = 14)
        im1 = axs[0].imshow(vis_img)
        axs[1].set_title('{}'.format(data[1].shape[0]), fontsize=14)
        im2 = axs[1].imshow(dataset.get_unnormed_item(i))
        return im1, im2

    ani = FuncAnimation(fig, animate, interval=40, blit=True, repeat=True, frames=100)
    ani.save("TLI.gif", dpi=300, writer=PillowWriter(fps=25))
    print(sum(times) / len(times))


def range_real_time(model, dataset, device, range_from, range_to):

    for i in range(range_from, range_to):
        original = np.array(dataset.get_unnormed_item(i))[:, :, ::-1].copy()
        data = dataset[i]
        image = data[0].unsqueeze(dim=0)

        image = image.to(device)
        with torch.set_grad_enabled(False):
            mu, mu_normed = model(image)

        vis_img = colorize_heat_map(mu)

        cv2.imshow('original', original)
        cv2.imshow('heat map', vis_img)

        cv2.waitKey(5)



