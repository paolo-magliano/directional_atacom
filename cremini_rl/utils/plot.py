from contourpy import contour_generator

import wandb
import os
from mushroom_rl.utils.plot import get_mean_and_confidence
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset, inset_axes
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# # Spring Pastels from https://www.heavy.ai/blog/12-color-palettes-for-telling-better-stories-with-your-data
COLOR_PALETTE = ["#fd7f6f", "#bd7ebe", "#3293db", "#7cc202", "#04a777", "#ffb55a", "#bd7ebe", "#3f423e"]

# COLOR_PALETTE = ["#e60049", "#0bb4ff", "#50e991", "#e6d800", "#9b19f5", "#ffa300", "#dc0ab4", "#b3d4ff", "#00bfa0",
#                  "#3f423e"]

title_dict = {"atacom_sac_0.5": "D-ATACOM@0.5", "wc_lag_sac_0.5": "WCSAC@0.5", "wc_lag_sac_0.9": "WCSAC@0.9",
              "wc_lag_sac_0.1": "WCSAC@0.1", "lag_sac": "LagSAC", "PPOLag": "PPOLag", "RCPO": "RCPO",
              "OnCRPO": "OnCRPO", "CPO": "CPO", "PCPO": "PCPO", "TRPOLag": "TRPOLag", "sac": "SAC",
              "atacom_sac_0.9": "D-ATACOM@0.9", "atacom_sac_0.1": "D-ATACOM@0.1",
              "baselineatacom_sac": "ATACOM + non-FI",
              "baselineatacom_sac_viability": "ATACOM + FI",
              "CAPPETS": "CAPPETS", "SafeLOOP": "SafeLOOP", "clbf_sac": "CBF-SAC", "safelayer_td3": "SafeLayerTD3",
              }

colour_dict = {"atacom_sac_0.9": 0, "atacom_sac_0.5": 0, "atacom_sac_0.1": 0, "wc_lag_sac_0.5": 1, "wc_lag_sac_0.9": 1,
               "wc_lag_sac_0.1": 1, "lag_sac": 2, "PPOLag": 3, "RCPO": 4, "OnCRPO": 5, "CPO": 6, "PCPO": 7,
               "TRPOLag": 8, "sac": 9, "baselineatacom_sac": 8, "baselineatacom_sac_viability": 7,
               "CAPPETS": 1, "SafeLOOP": 2, "clbf_sac": 3, "safelayer_td3": 4}


def plot_zero_level(ax, X, Y, Z, **kwargs):
    contour_gen = contour_generator(X, Y, Z)
    lines = contour_gen.lines(0)

    for line in lines:
        if len(line) > 20:
            ax.plot(*line.T, **kwargs)


def smooth(scalars, weight: float):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value

    return smoothed


def plot_learning_curve(grouped_runs, metric_key, title, xlabel, ylabel, steps_per_epoch, save_dir=None, linewidth=8,
                        smooth_weight=None):
    # Spring Pastels from https://www.heavy.ai/blog/12-color-palettes-for-telling-better-stories-with-your-data

    plt.rcParams["font.size"] = 24 # 55
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["DejaVu Serif"]
    # plt.rcParams["mathtext.fontset"] = "cm"
    # plt.rcParams['axes.linewidth'] = 2

    plt.figure(figsize=(16, 10))
    color_idx = 0

    # names = ["no decay", "decay=$0.97^{epoch}$", "decay=$0.98^{epoch}$"]

    for group_key in sorted(grouped_runs[metric_key].keys()):
        metric_df = grouped_runs[metric_key][group_key]

        mean, interval = get_mean_and_confidence(metric_df.transpose())

        if  "ijrr_air_hockey" in save_dir:
            mean = mean[:150]
            interval = interval[:150]

        if smooth_weight is not None:
            mean = np.array(smooth(mean, smooth_weight))
            interval = np.array(smooth(interval, smooth_weight))

        x = np.arange(mean.shape[-1]) * steps_per_epoch

        # temp = {
        #     "atacom_sac_fixed_delta_mushroomv1_e4f9d1277326ef29d4fef6dc35144c20107c9365": "0.3",
        #     "atacom_sac_fixed_delta_mushroomv1_64116e77ce76c431e35ad2c047d8a149d64822e9": "0.1",
        #     "atacom_sac_fixed_delta_mushroomv1_b7ae2a47d34831205930b91c71a1127f211a8125": "0.5",
        #     "atacom_sac_fixed_delta_mushroomv1_3e5089655030c94bf6db8aa7523301fc41125d57": "1",
        #     "atacom_sac_fixed_delta_mushroomv1_88a16eff42f0d896af418b577dc45234dd4fb21b": "3",
        #     "atacom_sac_fill_up_1145e0587f424822aedbf9ed683748dec3297361": "learned"

        # }

        # temp = {"atacom_sac_model_missmatch_0.8_mushroomv1_88a16eff42f0d896af418b577dc45234dd4fb21b": "0.8",
        #         "atacom_sac_model_missmatch_0.4_mushroomv1_88a16eff42f0d896af418b577dc45234dd4fb21b": "0.4",
        #         "atacom_sac_model_missmatch_0.2_mushroomv1_88a16eff42f0d896af418b577dc45234dd4fb21b": "0.2",
        #         "atacom_sac_model_miss_0.1_88a16eff42f0d896af418b577dc45234dd4fb21b": "0.1",
        #         "atacom_sac_fill_up_1145e0587f424822aedbf9ed683748dec3297361": "0"}

        group_label = group_key.replace("r3", "").replace("iros", "").replace("tmp", "").replace("_", " ").title()

        plt.plot(x, mean, label=group_label, color=COLOR_PALETTE[color_idx], linewidth=linewidth, alpha=0.7)
        plt.fill_between(x, mean - interval, mean +
                         interval, alpha=0.2, color=COLOR_PALETTE[color_idx], label="_nolegend_")
        color_idx += 1
    plt.xlabel(xlabel)
    # plt.ylabel(ylabel)
    # leg = plt.legend(ncol=3)
    plt.title(title)
    if "log" in ylabel.lower():
        plt.yscale("log")
    # leg_lines = leg.get_lines()
    # plt.setp(leg_lines, linewidth=linewidth)
    plt.tight_layout()
    ax = plt.gca()
    ax.tick_params('both', length=20, width=4, which='major')
    ax.tick_params('both', length=10, width=2, which='minor')
    # if metric_key == "sum_cost" or metric_key == "max_violation":
    #     axins = inset_axes(ax, 8, 4,
    #                        loc=1)  # , bbox_to_anchor=(0.2, 0.55), bbox_transform=ax.figure.transFigure)  # no zoom
    #     color_idx = 0
    #     for group_key in sorted(grouped_runs[metric_key].keys()):
    #         metric_df = grouped_runs[metric_key][group_key]

    #         mean, interval = get_mean_and_confidence(metric_df.transpose())

    #         axins.plot(x, mean, color=COLOR_PALETTE[color_idx], linewidth=linewidth)
    #         axins.fill_between(x, mean - interval, mean +
    #                            interval, alpha=0.2, color=COLOR_PALETTE[color_idx])
    #         color_idx += 1

    #     axins.set_xlim(0.5e6, 2e6)
    #     if metric_key == "sum_cost":
    #         axins.set_ylim(-0.1, 2)
    #     else:
    #         axins.set_ylim(0, 0.05)
    #     axins.xaxis.set_major_locator(ticker.NullLocator())
    #     box, c1, c2 = mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    #     plt.setp([box, c1, c2], linewidth=2)
    # plt.show()

    if save_dir is not None:
        # plt.savefig(save_dir + f"/{ylabel}.pdf", dpi=1000)
        plt.legend()
        plt.savefig(save_dir + f"/{ylabel}.png")

    # handles, labels = plt.gca().get_legend_handles_labels()
    # # order = [0, 3, 1, 2, 5]
    # order = [0, 2, 5, 4, 1, 3]
    # leg = plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], ncol=6, loc='center left',
    #                  bbox_to_anchor=(1, 0.5), frameon=False)

    #leg = plt.legend(ncol=6, loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    # leg_lines = leg.get_lines()
    # plt.setp(leg_lines, linewidth=linewidth + 2)

    #export_legend(leg, filename=os.path.join(save_dir, "legend.pdf"))

    # plt.show()


def export_legend(legend, filename="legend.png"):
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)


def download_run_history(entity, project, save_path, samples, filters):
    api = wandb.Api()

    runs = api.runs(f"{entity}/{project}", filters=filters)

    for run in runs:
        run_hist = run.history(samples=samples)

        if "Reward/R" in run_hist.keys():
            run_hist["R"] = run_hist["Reward/R"]

        if "Reward/J" in run_hist.keys():
            run_hist["J"] = run_hist["Reward/J"]

        if "ep_violation_rate" in run_hist.keys():
            run_hist["sum_cost"] = run_hist["ep_cost"]
            run_hist["violation_rate"] = run_hist["ep_violation_rate"]

        if not "success" in run_hist.keys():
            run_hist["success"] = 0

        if "puck_vel_cross" in run_hist.keys():
            run_hist["puck_vel"] = run_hist["puck_vel_cross"]

        try:
            metrics = ["J", "R", "episode_length"]
            if "max_cost" in run_hist.keys():
                metrics.append("max_cost")
            if "sum_cost" in run_hist.keys():
                metrics.append("sum_cost")
            if "violation_rate" in run_hist.keys():
                metrics.append("violation_rate")
            if "dist_to_target" in run_hist.keys():
                metrics.append("dist_to_target")
            if "success_rate" in run_hist.keys():
                metrics.append("success_rate")
            if "puck_vel" in run_hist.keys():
                metrics.append("puck_vel")
            
            run_hist = run_hist[metrics]

            run_hist.to_csv(f"{save_path}/{run.id}.csv", index=False)
        except:
            print(f"Failed to save {run.group} {run.id}")


def group_run_histories_by_key(entity, project, save_path, group_key, filters):
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", filters=filters)

    metrics = {}
    for run in runs:
        hist = pd.read_csv(f"{save_path}/{run.id}.csv")
        temp = run.config
        temp["group"] = run.group
        if "algo" in temp.keys():
            temp["alg"] = temp["algo"]
        for el in group_key:
            temp = temp.get(el)

        # group_key_val = float(temp)
        group_key_val = temp
        for key in hist.keys():
            if key not in metrics:
                metrics[key] = {}

            if group_key_val not in metrics[key]:
                metrics[key][group_key_val] = pd.DataFrame()

            # Only take 10 seeds for ablation studies
            if run.state == "finished": # and metrics[key][group_key_val].shape[1] < 10:
                metrics[key][group_key_val][run.id] = hist[key]
    return metrics


def make_path(project, name):
    data_path = os.path.join("data", project, name)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    plot_path = os.path.join("plots", project, name)
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    return data_path, plot_path


def plot_air_hockey_constraint(agent):
    from air_hockey_challenge.utils.kinematics import inverse_kinematics
    grid_pos_x = np.linspace(-0.5, 0.5, 10)
    grid_pos_y = np.linspace(0.6, 1.2, 10)
    X, Y = np.meshgrid(grid_pos_x, grid_pos_y)
    joint_pos = []

    pass

if __name__ == '__main__':
    entity = "paolo-magliano"
    project = "quadrotor_traj"
    name = "sac_vs_dc"

    data_path, plot_path = make_path(project, name)

    filters = {
        "$and": [
            {"group": {"$regex": "iros"}},
            {"$or": [
                {"group": {"$regex": "^sac_iros"}},
                {"group": {"$regex": "^atacom_sac_dc"}}
            ]},
        ]
    }

    download_run_history(entity, project, data_path, samples=1000, filters=filters)

    metrics = group_run_histories_by_key(entity, project, data_path, group_key=["group"], filters=filters)

    plot_learning_curve(metrics, "R", "Return", "Steps", "Return", steps_per_epoch=10000, save_dir=plot_path, smooth_weight=None)
    plot_learning_curve(metrics, "J", "Discounted Return", "Steps", "Discounted return", steps_per_epoch=10000, save_dir=plot_path, smooth_weight=None)
    plot_learning_curve(metrics, "episode_length", "Length of episodes", "Steps", "Espisode length", steps_per_epoch=10000, save_dir=plot_path, smooth_weight=None)
    
    if "sum_cost" in metrics.keys():
        plot_learning_curve(metrics, "sum_cost", "Total episodic cost", "Steps", "Total cost", steps_per_epoch=10000, save_dir=plot_path, smooth_weight=None) 

    if "violation_rate" in metrics.keys():
        plot_learning_curve(metrics, "violation_rate", "Violation rate", "Steps", "Violation rate", steps_per_epoch=10000, save_dir=plot_path, smooth_weight=None)

    if "max_cost" in metrics.keys():
        plot_learning_curve(metrics, "max_cost", "Maximum episodic violation", "Steps", "Max violation", steps_per_epoch=10000, save_dir=plot_path, smooth_weight=None)
    
    if "quadrotor" in project:
        plot_learning_curve(metrics, "dist_to_target", "Distance from target", "Steps", "Log target distance", steps_per_epoch=10000, save_dir=plot_path, smooth_weight=None)

    if "hockey" in project:
        plot_learning_curve(metrics, "success_rate", "Success rate of goal", "Steps", "Success rate", steps_per_epoch=10000, save_dir=plot_path, smooth_weight=None)
        plot_learning_curve(metrics, "puck_vel", "Puck velocity", "Steps", "Puck velocity", steps_per_epoch=10000, save_dir=plot_path, smooth_weight=None)


    # plot_learning_curve(metrics, "sum_cost", "Cost", "Steps", "Cost", steps_per_epoch=10000, save_dir=plot_path)
