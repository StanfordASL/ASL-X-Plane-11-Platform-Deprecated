import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from PIL import Image

plt.rc('font', size=20)
plt.rc('font', family='Times New Roman')
plt.rcParams['legend.fontsize'] = 'large'
plt.rcParams['axes.labelsize'] = 'large'

def extract_episode_details(filename):
    match = re.search(r'(\d+)_(\d+).png', filename)
    if match:
        episode_number = int(match.group(1))
        timestamp = int(match.group(2))
        return (episode_number, timestamp)
    else:
        raise ValueError(f"Could not extract episode number and timestamp from {filename}")

def get_episode_dict(df):
    episodes = {}
    for i, filename in enumerate(df.loc[:,"image_filename"]):
        episode, timestep = extract_episode_details(filename)
        if not episode in episodes:
            episodes[episode] = [i]
        else:
            episodes[episode].append(i)
    return episodes

def animate_episode_with_ood(data_dir, save_dir, df, episodes, episode_num, simulator_params, experiment_params, model, monitor, real_time_x=5, triggers_fallback=True):
    ctes = df.loc[episodes[episode_num], "distance_to_centerline_meters"]
    dps = df.loc[episodes[episode_num], "downtrack_position_meters"]
    hes = df.loc[episodes[episode_num], "heading_error_degrees"]
    dp_min = np.min(dps) - 10
    dp_max = np.max(dps) + 10
    constraint = simulator_params["simulator"]["runway_width"]
    time_step = simulator_params["simulator"]['time_step']
    failure_times = [int(t / time_step) for t in experiment_params["ood"]["transient_time_range"]]
    # import pdb; pdb.set_trace()
    images = [Image.open(data_dir + df.loc[i, "image_filename"]) for i in episodes[episode_num]]
    ood_scores = [monitor.monitor(image, None) for image in images]
    fallback_triggered = [score[1] for score in ood_scores]
    ood_scores = [score[0] for score in ood_scores]
    
    estimates = np.array([model.get_estimate(img) for img in images])
    perception_errors = np.linalg.norm(estimates - np.vstack((ctes, hes)).T, axis=1)
    print(estimates.shape, perception_errors.shape)
    t_max = len(dps)
    fig = plt.figure(figsize=(15,9))
    gs = GridSpec(3, 6, figure=fig)
    ax1 = fig.add_subplot(gs[:2, 1:5])
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)
    

    ax2 = fig.add_subplot(gs[2, :2])
    ax2.plot([-1,1000], [-constraint,-constraint], "r--", lw=3, label="constraint")
    ax2.plot([-1,1000], [constraint,constraint], "r--", lw=3)
    traj_line = ax2.plot([],[], color="tab:blue", lw=3)[0]
    ax2.set_xlim(dp_min, dp_max)
    ax2.set_ylim(-5,11)
    ax2.grid()
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    ax2.set_title("Taxi Trajectory")
    ax2.legend(loc="lower right", fontsize=18)

    ax3 = fig.add_subplot(gs[2, 2:4])
    alpha=.5
    if experiment_params["ood"]["corruption"][0] != "None":
        ax3.fill_between(failure_times, [-100, -100], [100,100],color="tab:red", alpha=alpha)
        if triggers_fallback:
            ax3.plot(np.arange(0,t_max), np.ones(t_max) * 4.4, "--", color="tab:red", lw=3, label="error bound")
            ax3.legend(loc="upper left", fontsize=18)
    error_line = ax3.plot([],[], lw=3, color="tab:blue", alpha=1)[0]
    ax3.grid()
    ax3.set_ylim(0,19)
    ax3.set_xlim(0,t_max)
    ax3.set_xlabel("time (s)")
    ax3.set_ylabel(r"$\|\hat{\mathbf{x}} - \mathbf{x}\|$")
    ax3.set_title("NN Estimate Errors")

    ax4 = fig.add_subplot(gs[2, 4:])
    ood_line = ax4.plot([],[], lw=3, color="tab:blue", alpha=1)[0]
    # ax.set_xlim(0,50)
    ax4.set_ylim(0,4)
    ax4.set_xlim(0,t_max)
    if np.any(fallback_triggered) and triggers_fallback:
        ax4.plot(np.arange(0,t_max), np.ones(t_max) * monitor.threshold, "--", color="tab:red", lw=3, label="ood trigger")
        ax4.legend(loc="lower right", fontsize=18)
    ax4.grid()
    ax4.set_xlabel("time (s)")
    ax4.set_ylabel("Anomaly Score")
    ax4.set_title("OOD Detection")
    if experiment_params["ood"]["corruption"][0] != "None":
        ax4.fill_between(failure_times, [-100, -100], [100,100],color="tab:red", label="ood", alpha=alpha)
    plt.tight_layout()
    # fig.subplots_adjust(wspace=1, hspace=.2)

    def animate(i):
        ax1.set_title("Observation, t = %d (s)" % i)
        ax1.imshow(images[i])

        if fallback_triggered[i] and triggers_fallback:
            ax1.text(120, 150, "Fallback Triggered", color="white", fontsize=30)
            
        traj_line.set_data(dps[:i+1],ctes[:i+1])
        ood_line.set_data(np.arange(min(i+1, len(ood_scores))), ood_scores[:i+1])
        error_line.set_data(np.arange(min(i+1, len(ood_scores))), perception_errors[:i+1])

    print("animating episode %d" % episode_num)

    ani = FuncAnimation(fig, animate, frames=len(episodes[episode_num]))

    print("saving")
    writer = animation.FFMpegWriter(fps=real_time_x / simulator_params["simulator"]["time_step"])

    ani.save(data_dir + save_dir + "episode_%d_ood.mp4" % episode_num, writer=writer)


def animate_episode_with_traj(data_dir, save_dir, df, episodes, episode_num, simulator_params, real_time_x=5):
    ctes = df.loc[episodes[episode_num], "distance_to_centerline_meters"]
    dps = df.loc[episodes[episode_num], "downtrack_position_meters"]
    dp_min = np.min(dps) - 10
    dp_max = np.max(dps) + 10
    dps_constraints = np.linspace(dp_min, dp_max)
    constraint = simulator_params["simulator"]["runway_width"]

    fig, axs = plt.subplots(1, 2, figsize=(13,5), gridspec_kw={'width_ratios':[3,2]})
    axs[0].get_xaxis().set_visible(False)
    axs[0].get_yaxis().set_visible(False)
    axs[1].grid()
    axs[1].set_ylim(-15,15)
    axs[1].set_xlim(dp_min, dp_max)
    axs[1].plot(dps_constraints, np.ones(len(dps_constraints)) * constraint, "r--", lw=2)
    axs[1].plot(dps_constraints, -1 * np.ones(len(dps_constraints)) * constraint, "r--", lw=2)
    axs[1].set_xlabel("x (m)")
    axs[1].set_ylabel("y (m)")
    axs[1].set_title("taxi trajectory")

    lines = []
    lines.append(axs[1].plot([],[],"-o", label="Ours", lw=3, color="tab:blue", markersize=6)[0])
    # axs[1].legend(loc="upper right",ncol=3, fontsize=15)# bbox_to_anchor=(1.05,1.05))
    plt.tight_layout()

    def animate(j):
        # import pdb; pdb.set_trace()
        lines[0].set_data(dps[:j + 1], ctes[:j + 1])
        i = episodes[episode_num][j]
        axs[0].imshow(img.imread(data_dir + df.loc[i, "image_filename"]))
        time_of_day = simulator_params["simulator"]["time_of_day"][df.loc[i,"period_of_day"]]["label"]
        corruption = simulator_params["screenshot_camera"]["corruption_types"][df.loc[i,"image_corruption"]]
        axs[0].set_title("Episode %d: " % episode_num + time_of_day+ ", " + corruption)

    print("animating episode %d" % episode_num)

    ani = FuncAnimation(fig, animate, frames=len(episodes[episode_num]))

    print("saving")
    writer = animation.FFMpegWriter(fps=real_time_x / simulator_params["simulator"]["time_step"])

    ani.save(data_dir + save_dir + "episode_%d_traj.mp4" % episode_num, writer=writer)


def animate_episode(data_dir, save_dir, df, episodes, episode_num, simulator_params, real_time_x=5):
    fig = plt.figure()
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.tight_layout()

    def animate(j):
        # import pdb; pdb.set_trace()
        i = episodes[episode_num][j]
        ax.imshow(img.imread(data_dir + df.loc[i, "image_filename"]))
        time_of_day = simulator_params["simulator"]["time_of_day"][df.loc[i,"period_of_day"]]["label"]
        corruption = simulator_params["screenshot_camera"]["corruption_types"][df.loc[i,"image_corruption"]]
        ax.set_title("Episode %d: " % episode_num + time_of_day+ ", " + corruption)
    print("animating")

    ani = FuncAnimation(fig, animate, frames=len(episodes[episode_num]))

    print("saving")
    writer = animation.FFMpegWriter(fps=real_time_x / simulator_params["simulator"]["time_step"])

    ani.save(data_dir + save_dir + "episode_%d.mp4" % episode_num, writer=writer)



