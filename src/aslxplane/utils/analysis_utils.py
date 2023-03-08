import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
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



