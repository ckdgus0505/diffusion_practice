import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from spiraldataset import spiral

def plot_figure(x, fname="fig.png"):
    x = x[0].detach().numpy()
    plt.cla()
    plt.xlim([-4, 4])      # X축의 범위: [xmin, xmax]
    plt.ylim([-4, 4])     # Y축의 범위: [ymin, ymax]
    plt.scatter(x[:,0], x[:,1])
    plt.savefig(fname)

# def plot_forward_gif(n_point=400, time_step=300, fname="fig.gif"):
#     x = spiral(n_point)

#     mean = torch.tensor([0.0, 0.0])
#     covariance_matrix = torch.eye(2)
#     multivariate_normal = dist.MultivariateNormal(mean, covariance_matrix)

#     ani = FuncAnimation(plt.gcf(), forward_process_for_gif, frames=100, interval=10)
#     ani.save(fname, fps=60)

