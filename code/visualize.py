import matplotlib
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


# adapted from Actor anim.py
# take xyz coordinates (Nj x 3 x T)
def plot_motion(motion, kinematic_tree, interval=50, save_path="/home/halinh/test.gif"):
    matplotlib.use('Agg') # non-interactive
    fig = plt.figure(figsize=[3, 3])
    ax = fig.add_subplot(111, projection='3d')

    def init():
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)
        ax.set_zlim(-100, 100)

        ax.view_init(azim=-90, elev=110)
        # ax.set_axis_off()
        ax.xaxis._axinfo["grid"]['color'] = (0.5, 0.5, 0.5, 0.25)
        ax.yaxis._axinfo["grid"]['color'] = (0.5, 0.5, 0.5, 0.25)
        ax.zaxis._axinfo["grid"]['color'] = (0.5, 0.5, 0.5, 0.25)

    colors = ['red', 'magenta', 'black', 'green', 'blue']
    def update(index):
        ax.lines.clear()
        ax.collections.clear()
        if kinematic_tree is not None:
            for chain, color in zip(kinematic_tree, colors):
                ax.plot(motion[chain, 0, index],
                        motion[chain, 1, index],
                        motion[chain, 2, index], linewidth=1.0, color=color)
        else:
            ax.scatter(motion[1:, 0, index], motion[1:, 1, index],
                       motion[1:, 2, index], c="red")
            ax.scatter(motion[:1, 0, index], motion[:1, 1, index],
                       motion[:1, 2, index], c="blue")

    ani = FuncAnimation(fig, update, frames=motion.shape[-1], interval=interval, repeat=False, init_func=init)
    plt.tight_layout()
    ani.save(save_path, writer='ffmpeg', fps=1000/interval)
    plt.close()
