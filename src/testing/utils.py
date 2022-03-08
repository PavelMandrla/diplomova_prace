import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import torch
import cv2
import numpy as np
from timeit import default_timer as timer
from torchvision import transforms
from tqdm import tqdm


def show_image(input_tensor, head_pos=None):
    fig, axs = plt.subplots(1, input_tensor[0].shape[0])
    fig.suptitle('Image sequence')
    fig.set_figwidth(25)

    if input_tensor[0].shape[0] == 1:
        axs.imshow(input_tensor[0][0].permute(1, 2, 0))
    else:
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


def cv_frame_to_tensor(frame, size):
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    frame = cv2.resize(frame, size)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype('float32')
    frame /= 255.0

    frame_tensor = torch.from_numpy(frame)
    frame_tensor = torch.permute(frame_tensor, (2, 0, 1))
    frame_tensor = norm(frame_tensor)

    return frame_tensor


def animate_video(model, device, src_path, save_path):
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise "video file %s not found" % src_path

    frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, 30, (frame_width, frame_height))

    frame_buffer = []
    buffer_size = model.seq_len * model.stride

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame_buffer.append(cv_frame_to_tensor(frame, model.input_size))

            if len(frame_buffer) < buffer_size:
                continue

            sequence = torch.stack(frame_buffer[:buffer_size:model.stride]).unsqueeze(0).to(device)

            with torch.set_grad_enabled(False):
                mu, mu_normed = model(sequence)

            heat_map = colorize_heat_map(mu)

            x_offset = frame.shape[1] - heat_map.shape[1]
            y_offset = frame.shape[0] - heat_map.shape[0]
            frame[y_offset:frame.shape[0], x_offset:frame.shape[1]] = heat_map

            text_org = (25, 50)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2
            text_color = (255, 0, 0)
            text_thickness = 2
            frame = cv2.putText(frame, '%.2f' % torch.sum(mu).item(), text_org, font, font_scale, text_color, text_thickness, cv2.LINE_AA)

            out.write(frame)

            # cv2.imshow("frame", frame)
            # cv2.waitKey(20)
            frame_buffer.pop(0)
        else:
            break

    cap.release()
    out.release()


def evaluate_dataset(model, dataloader, device, filename):
    counts = []
    times = []

    t = tqdm(dataloader, leave=False, total=len(dataloader))
    for i, data in enumerate(t):
        image = data[0]
        real_count = data[1].size()[1]

        start = timer()
        image = image.to(device)
        with torch.set_grad_enabled(False):
            mu, mu_normed = model(image)
        end = timer()
        times.append(end - start)

        counts.append((torch.sum(mu).item(), real_count))

    with open(filename, 'w+') as file:
        for i, count in enumerate(counts):
            file.write('%f, %f, %f\n' % (count[1], count[0], times[i]))

