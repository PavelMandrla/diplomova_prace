import matplotlib.pyplot as plt


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
