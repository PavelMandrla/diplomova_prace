import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import torch
import cv2
import numpy as np
from timeit import default_timer as timer
from torchvision import transforms


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


def eval_video(model, vid_path, device, sequence_len=5, stride=1):
    cap = cv2.VideoCapture(vid_path)

    if not cap.isOpened():
        raise "could not open video file %s" % vid_path

    frame_buffer = []
    buffer_size = sequence_len * stride

    norm =  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (1280, 720))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype('float32')
            frame /= 255.0

            frame_tensor = torch.from_numpy(frame)
            frame_tensor = torch.permute(frame_tensor, (2, 0, 1))
            frame_tensor = norm(frame_tensor)
            frame_buffer.append(frame_tensor)
            if len(frame_buffer) < buffer_size:
                continue

            input = torch.stack(frame_buffer[:buffer_size:stride]).unsqueeze(0)
            input = input.to(device)

            with torch.set_grad_enabled(False):
                mu, mu_normed = model(input)

            vis_img = colorize_heat_map(mu)

            cv2.imshow('heat map', vis_img)
            cv2.imshow('frame', frame)
            print(torch.sum(mu).item())
            cv2.waitKey(20)

            frame_buffer.pop(0)
        else:
            break


def animate_video(model, vid_path, device, sequence_len=5, stride=1):
    cap = cv2.VideoCapture(vid_path)
    results = []

    if not cap.isOpened():
        raise "could not open video file %s" % vid_path

    frame_buffer = []
    buffer_size = sequence_len * stride

    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    #while cap.isOpened():
    for i in range(1000):
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (1280, 720))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype('float32')
            frame /= 255.0

            frame_tensor = torch.from_numpy(frame)
            frame_tensor = torch.permute(frame_tensor, (2, 0, 1))
            frame_tensor = norm(frame_tensor)
            frame_buffer.append(frame_tensor)
            if len(frame_buffer) < buffer_size:
                continue

            input = torch.stack(frame_buffer[:buffer_size:stride]).unsqueeze(0)
            input = input.to(device)

            with torch.set_grad_enabled(False):
                mu, mu_normed = model(input)

            vis_img = colorize_heat_map(mu)
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)

            results.append((cv2.resize(frame, (640, 368)), vis_img, torch.sum(mu).item()))
            print(len(results))

            frame_buffer.pop(0)
        else:
            break

    fig, axs = plt.subplots(1, 2)
    def animate(i):
        axs[0].set_title('%.2f' % (results[i][2]), fontsize = 14)
        im1 = axs[0].imshow(results[i][1])
        axs[1].set_title('input', fontsize=14)
        im2 = axs[1].imshow(results[i][0])
        return im1, im2

    ani = FuncAnimation(fig, animate, interval=40, blit=True, repeat=True, frames=len(results))
    ani.save("TLI.gif", dpi=300, writer=PillowWriter(fps=25))


def plot_timeseries(model, dataset, device, range_from, range_to):
    counts = []

    def plot_results(results):
        plt.plot([x for x, _ in results])
        plt.plot([x for _, x in results])
        plt.ylabel('some numbers')
        plt.show()

    for i in range(range_from, range_to):
        print(i - range_from, range_to - range_from)
        data = dataset[i]
        image = data[0].unsqueeze(dim=0)
        real_count = len(data[1])

        image = image.to(device)
        with torch.set_grad_enabled(False):
            mu, mu_normed = model(image)
        counts.append((torch.sum(mu).item(), real_count))

    with open('counts.csv', 'w+') as file:
        for count in counts:
            file.write('%.3f, %.3f\n' % (count[1], count[0]))
    #plot_results(counts)

