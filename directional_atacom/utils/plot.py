from contourpy import contour_generator

import wandb
import os
from mushroom_rl.utils.plot import get_mean_and_confidence
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import random
import re
from adjustText import adjust_text
from scipy import interpolate

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
})

# # Spring Pastels from https://www.heavy.ai/blog/12-color-palettes-for-telling-better-stories-with-your-data
COLOR_PALETTE = ["#fd7f6f", "#bd7ebe", "#3293db", "#7cc202", "#04a777", "#ffb55a", "#5e60ce", "#2f4858"]

ALTERNATIVE_COLOR_PALETTE = [
    "#e4572e",  # warm red-orange
    "#6a4c93",  # deep purple (different from your violet)
    "#118ab2",  # strong cyan-blue
    "#8ac926",  # fresh green (not lime)
    "#f4a261",  # soft amber
    "#264653",  # deep blue-gray anchor
]

# COLOR_PALETTE = ["#e60049", "#0bb4ff", "#50e991", "#e6d800", "#9b19f5", "#ffa300", "#dc0ab4", "#b3d4ff", "#00bfa0",
#                  "#3f423e"]

# title_dict = {"atacom_sac_0.5": "D-ATACOM@0.5", "wc_lag_sac_0.5": "WCSAC@0.5", "wc_lag_sac_0.9": "WCSAC@0.9",
#               "wc_lag_sac_0.1": "WCSAC@0.1", "lag_sac": "LagSAC", "PPOLag": "PPOLag", "RCPO": "RCPO",
#               "OnCRPO": "OnCRPO", "CPO": "CPO", "PCPO": "PCPO", "TRPOLag": "TRPOLag", "sac": "SAC",
#               "atacom_sac_0.9": "D-ATACOM@0.9", "atacom_sac_0.1": "D-ATACOM@0.1",
#               "baselineatacom_sac": "ATACOM + non-FI",
#               "baselineatacom_sac_viability": "ATACOM + FI",
#               "CAPPETS": "CAPPETS", "SafeLOOP": "SafeLOOP", "clbf_sac": "CBF-SAC", "safelayer_td3": "SafeLayerTD3",
#               }

# colour_dict = {"atacom_sac_0.9": 0, "atacom_sac_0.5": 0, "atacom_sac_0.1": 0, "wc_lag_sac_0.5": 1, "wc_lag_sac_0.9": 1,
#                "wc_lag_sac_0.1": 1, "lag_sac": 2, "PPOLag": 3, "RCPO": 4, "OnCRPO": 5, "CPO": 6, "PCPO": 7,
#                "TRPOLag": 8, "sac": 9, "baselineatacom_sac": 8, "baselineatacom_sac_viability": 7,
#                "CAPPETS": 1, "SafeLOOP": 2, "clbf_sac": 3, "safelayer_td3": 4}

colour_dict = {label: index  for index, label in enumerate(["atacom_dc", "datacom_dc", "atacom", "datacom", "sac", "datacom_dc_constraint", "datacom_constraint"])}
colour_idx = 0

metrics_label = {
    "R": "Return",
    "J": "Discounted return",
    "episode_length": "Length of episodes",
    "dist_to_target": "Distance from target",
    "success_rate": "Success rate",
    "puck_vel": "Puck velocity",
    "sum_cost": "Episodic cost",
    "max_cost": "Max violation",
    "violation_rate": "Violation rate",
    "steps": "Steps"
}

def format_label(label):
    label = label.replace("_sac", "").replace("r3", "").replace("iros", "").replace("tmp", "").replace("baseline-", "")
    label = label.replace("constr", "constraint").replace("array", "").replace("nod", "").replace("auto", "")
    label = re.sub(r"beta_\d+(\.\d+)?", "", label)
    label = re.sub(r"delta_\d+(\.\d+)?", "", label)
    label = re.sub(r"lambda_\d+(\.\d+)?", "", label)
    label = re.sub(r"_+", "_", label)

    return label.strip("_")

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

def plot(mode, grouped_runs, y_metric, x_metric="steps", steps_per_epoch=None, save_dir=None, linewidth=8, smooth_weight=None, logscale=False):
    # Spring Pastels from https://www.heavy.ai/blog/12-color-palettes-for-telling-better-stories-with-your-data

    plt.rcParams["font.size"] = 40 # 55

    if mode == "learning":
        plt.figure(figsize=(16, 8))
    else:
        plt.figure(figsize=(10, 8))
    # color_idx = 0
    y_min, y_max = float("inf"), -float("inf")

    group_keys = sorted(grouped_runs[y_metric].keys()) if mode == "learning" else sorted(grouped_runs["performance"].keys())

    p_xs = []
    p_ys = []
    p_labels = []

    global colour_idx
    colour_idx = 0
    
    for group_key in group_keys:
        label = format_label(group_key)
        if mode == "learning":
            group_y_min, group_y_max, _, _ = _plot_learning_curve(grouped_runs, y_metric, group_key, label, steps_per_epoch, save_dir, linewidth, smooth_weight)
        if mode == "beta" or mode == "delta":
            group_y_min, group_y_max, p_x, p_y, p_texts = _plot_hp(grouped_runs, group_key, label, linewidth, smooth_weight)
            
            p_xs.append(p_x)
            p_ys.append(p_y)
            p_labels.extend(p_texts)
        
        if group_y_min is not None:
            y_min = min(y_min, group_y_min)
        if group_y_max is not None:
            y_max = max(y_max, group_y_max)

    ax = plt.gca()

    xlabel = metrics_label[x_metric]
    ylabel = metrics_label[y_metric]

    log_label = "_log" if "cost" in y_metric and logscale else ""
    if np.isfinite(y_min) and np.isfinite(y_max):
        if log_label:
            plt.yscale("log")
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(bottom=max(y_min, 1e-6), top=ymax)
        else:
            y_range = y_max - y_min
            if y_range == 0:
                y_range = abs(y_max) if y_max != 0 else 1.0
            
            margin = 0.1 * y_range
            plt.ylim(y_min - margin, y_max + margin)


    plt.xlabel(xlabel)
    if mode == "learning":
        plt.title(ylabel)
        plt.tight_layout()
    if mode == "beta" or mode == "delta":
        plt.ylabel(ylabel)
        plt.title(f"{mode.title()} analysis")

        plt.tight_layout()

        margin = 0.17
        ax = plt.gca()

        for getter, setter in [(ax.get_xlim, ax.set_xlim),
                            (ax.get_ylim, ax.set_ylim)]:
            lo, hi = getter()
            span = hi - lo
            setter(lo - margin * span, hi + margin * span)

        px = np.array([np.linspace(min(p_x), max(p_x), 100) for p_x in p_xs]).flatten()
        py = np.array([interpolate.interp1d(p_x, p_y)(np.linspace(min(p_x), max(p_x), 100)) for p_x, p_y in zip(p_xs, p_ys)]).flatten()

        # ---- Auto-adjust to remove overlaps ----
        adjust_text(
            p_labels,
            x=px,
            y=py,
            ax=plt.gca(),
            only_move={"text": "xy"},
            # arrowprops=dict(arrowstyle="-", lw=3),
            # min_arrow_len=70,
            expand=(1.7, 1.7),
            # force_text=(0.3, 0.3),
            # force_static=(0.3, 0.3),
            lim=500
        )


    if mode == "learning" and y_metric in {"sum_cost", "max_violation"} and not logscale and "planar_air_hockey_vel/atacom" not in save_dir and "ijrr_air_hockey/sac" not in save_dir: # and "sac" not in [format_label(g) for g in group_keys]:
        axins = inset_axes(
            ax,
            width="55%",   
            height="45%",
            loc="upper right",
            borderpad=1.2
        )

        color_idx = 0
        zoom_y_min, zoom_y_max = float("inf"), -float("inf")

        zoom_idx = (0.5, 1.)
        if "planar_air_hockey_vel/datacom" in save_dir:
            zoom_idx = (0.05, 0.5)


        for group_key in group_keys:
            label = format_label(group_key)
            g_y_min, g_y_max, g_x_min, g_x_max = _plot_learning_curve(grouped_runs, y_metric, group_key, label, steps_per_epoch, save_dir, linewidth, smooth_weight, axins=axins, zoom_idx=zoom_idx)

            zoom_y_min = min(zoom_y_min, g_y_min)
            zoom_y_max = max(zoom_y_max, g_y_max)
        
        axins.set_xlim((g_x_min, g_x_max))
        axins.set_ylim(zoom_y_min - 0.10 * (zoom_y_max - zoom_y_min), zoom_y_max + 0.10 * (zoom_y_max - zoom_y_min))

        axins.xaxis.set_major_locator(ticker.NullLocator())
        axins.tick_params(axis="both", which="major", length=10, width=2)

        box, c1, c2 = mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="0.35")
        plt.setp([box, c1, c2], linewidth=2.5)

    ax.tick_params('both', length=20, width=4, which='major')
    ax.tick_params('both', length=10, width=2, which='minor')

    if save_dir is not None:
        if mode == "learning":
            plt.savefig(save_dir + f"/{y_metric + log_label}.pdf", dpi=1000)
        if mode == "beta" or mode == "delta":
            plt.savefig(save_dir + f"/{x_metric}.pdf", dpi=1000)

    leg = plt.legend(ncol=4, loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

    export_legend(leg, filename=os.path.join(save_dir, "legend.pdf"))

def _plot_learning_curve(grouped_runs, key_metric, group_key, label, steps_per_epoch, save_dir, linewidth=8, smooth_weight=None, axins=None, zoom_idx=(0., 1.)):
    metric_df = grouped_runs[key_metric][group_key].sort_index(axis=1)

    if len(metric_df.columns) >= 15:
        metric_df = metric_df.iloc[:, :15]
    # else:
    #     print(f"Method {group_key} for metric {key_metric} has only {len(metric_df.columns)} number of seeds")

    mean, interval = get_mean_and_confidence(metric_df.transpose())

    # if "ijrr_air_hockey" in save_dir:
    #     mean = mean[:150]
    #     interval = interval[:150]

    if "datacom" in save_dir:
        mean = mean[:150]
        interval = interval[:150]

    # if "fixed_delta" in save_dir and "cost" in key_metric:
    #     mean = mean[:50]
    #     interval = interval[:50]

    if smooth_weight is not None:
        mean = np.array(smooth(mean, smooth_weight))
        interval = np.array(smooth(interval, smooth_weight))

    x = np.arange(mean.shape[-1]) * steps_per_epoch

    if axins:
        linewidth = linewidth * 0.5
    
    zoom_idx_start = int(zoom_idx[0] * len(x))
    zoom_idx_end = int(zoom_idx[1] * len(x))


    if label in colour_dict.keys():
        plot_color = COLOR_PALETTE[colour_dict[label]]
    else:
        global colour_idx
        plot_color = ALTERNATIVE_COLOR_PALETTE[colour_idx]
        colour_idx += 1

    plotter = axins if axins else plt

    plotter.plot(x, mean, label=label.replace("_", "-").upper().replace("CONSTRAINT", "constraint"), color=plot_color, linewidth=linewidth, alpha=0.7)
    plotter.fill_between(x, mean - interval, mean +
                        interval, alpha=0.2, color=plot_color, label="_nolegend_")

    return mean[zoom_idx_start:zoom_idx_end].min(), mean[zoom_idx_start:zoom_idx_end].max(), x[zoom_idx_start:zoom_idx_end].min(), x[zoom_idx_start:zoom_idx_end].max()

def _plot_hp(grouped_runs, group_key, label, linewidth=8, smooth_weight=None):
    performance_metric_df = grouped_runs['performance'][group_key]
    safety_metric_df = grouped_runs['safety'][group_key]

    performance_mean, performance_interval = get_mean_and_confidence(performance_metric_df)
    safety_mean, safety_interval = get_mean_and_confidence(safety_metric_df)

    if smooth_weight is not None:
        performance_mean = np.array(smooth(performance_mean, smooth_weight))
        performance_interval = np.array(smooth(performance_interval, smooth_weight))

        safety_mean = np.array(smooth(safety_mean, smooth_weight))
        safety_interval = np.array(smooth(safety_interval, smooth_weight))

    plt.plot(safety_mean, performance_mean, color=COLOR_PALETTE[colour_dict[label]], linewidth=linewidth, alpha=0.9)
    plt.scatter(
        safety_mean, performance_mean,
        color=COLOR_PALETTE[colour_dict[label]],
        s=500, 
        zorder=3,
        label=label.replace("_", "-").upper(),
        alpha=0.9
    )

    ax = plt.gca()

    texts = []
    for x, y, p_label in zip(safety_mean, performance_mean, safety_mean.index):
        # texts.append(ax.text(x, y, str(p_label)))
        texts.append(ax.annotate(
            f"{p_label}", (x, y),
            ha="left", va="center",
            fontsize=32,
        ))


    return None, None, safety_mean, performance_mean, texts

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

        if "joint_pos_constr/0" in run_hist.keys() and not "sum_cost" in run_hist.keys():
            joint_pos_constr = np.array([run_hist[f"joint_pos_constr/{i}"] for i in range(14)])
            link_pos_constr = np.array([run_hist[f"link_pos_constr/{i}"] for i in range(7)])
            constraints = np.maximum(np.concatenate([joint_pos_constr, link_pos_constr]), 0)
            run_hist["sum_cost"] = np.mean(constraints, axis=0)

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


def group_run_histories_by_key(entity, project, save_path, group_key, filters, key_performance_metric=None, key_safety_metric=None):
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", filters=filters)

    metrics = {}
    run_metrics = {}
    running = {}
    failed = {}
    for run in runs:
        try:
            hist = pd.read_csv(f"{save_path}/{run.id}.csv")
        except:
            continue
        temp = run.config
        temp["group"] = run.group
        if "algo" in temp.keys():
            temp["alg"] = temp["algo"]
        for el in group_key:
            temp = temp.get(el)

        # group_key_val = float(temp)
        group_key_val = temp

        seed = run.config.get("seed")

        args = (run.metadata or {}).get("args")
        if seed is None and args and "--seed" in args:
            seed = int(args[args.index("--seed") + 1])

        if seed is None:
            seed = - random.randint(1, 1000)


        if run.state == "finished":
            for key in hist.keys():
                if key not in metrics:
                    metrics[key] = {}
                    run_metrics[key] = {
                        'last': {},
                        'mean': {}
                    }

                if group_key_val not in metrics[key]:
                    metrics[key][group_key_val] = pd.DataFrame()
                    run_metrics[key]['last'][group_key_val] = pd.Series(dtype=float)
                    run_metrics[key]['mean'][group_key_val] = pd.Series(dtype=float)

                # Only take 10 seeds for ablation studies
                if seed not in metrics[key][group_key_val]:
                    metrics[key][group_key_val][seed] = hist[key]
                    if  run_metrics[key]['last'][group_key_val].shape[-1] < 5:
                        run_metrics[key]['last'][group_key_val].at[seed] = hist[key].mean()
                        run_metrics[key]['mean'][group_key_val].at[seed] = hist[key].mean()
        elif run.state == "running":
            if group_key_val not in running.keys():
                running[group_key_val] = {}
            running[group_key_val][seed] = run.summary.get("_step", None)
        else:
            if group_key_val not in failed.keys():
                failed[group_key_val] = {}
            failed[group_key_val][seed] = run.summary.get("_step", None)

    for method in failed.keys():
        print(f"\t{method} failed with {len(failed[method].items())} seeds: {sorted(failed[method].items(), key=lambda x: x[1])}")

    for method in running.keys():
        print(f"\t{method} is running with {len(running[method].items())} seeds: {sorted(running[method].items(), key=lambda x: x[1])}")
        

    return metrics, run_metrics

def process_run_metrics(run_metrics, method, hp, performance_metric_key, safety_metric_key):
    metrics = {
        'performance': {},
        'safety': {}
    }

    for group_key in sorted(run_metrics[performance_metric_key]['last'].keys()):      
        match = re.search(rf"{hp}_([0-9.]+)", group_key)
        hp_value = float(match.group(1)) if match else None

        if "dc" in group_key:
            method_key = method + "_dc" if hp == "beta" else method + "_dc_constr"
        else:
            method_key = method if hp == "beta" else method + "_constr"

        if method_key not in metrics['performance']:
            metrics['performance'][method_key] = pd.DataFrame()
            metrics['safety'][method_key] = pd.DataFrame()
        
        if group_key not in run_metrics[performance_metric_key]['last'].keys() or group_key not in run_metrics[safety_metric_key]['mean'].keys():
            continue

        performance_df = run_metrics[performance_metric_key]['last'][group_key].to_frame(name=hp_value).reset_index(drop=True)
        safety_df = run_metrics[safety_metric_key]['mean'][group_key].to_frame(name=hp_value).reset_index(drop=True)

        metrics['performance'][method_key] = pd.concat([metrics['performance'][method_key], performance_df], axis=1).sort_index(axis=1)
        metrics['safety'][method_key] = pd.concat([metrics['safety'][method_key], safety_df], axis=1).sort_index(axis=1)
        
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

def filter_dict(and_args=[], or_args=[], nor_args=[]):
    base = {"$and": []}

    for arg in and_args:
        base["$and"].append({"group": {"$regex": arg}})

    or_array = []
    for arg in or_args:
        or_array.append({"group": {"$regex": arg}})

    if or_array:
        base["$and"].append({"$or": or_array})

    nor_array = []
    for arg in nor_args:
        nor_array.append({"group": {"$regex": arg}})

    if nor_array:
        base["$and"].append({"$nor": nor_array})

    return base

plots_list = [
    # ("air_hockey_vel", "atacom_vs_dc", filter_dict([], ["^(baseline-)?atacom_sac$", "^(baseline-)?atacom_sac_dc$"])),
    # ("air_hockey_vel", "sac_vs_dc", filter_dict([], ["^sac$", "^(baseline-)?atacom_sac_dc$"])),

    # ("quadrotor", "atacom_vs_dc", filter_dict([], ["^(baseline-)?atacom_sac$", "^(baseline-)?atacom_sac_dc$"])),
    # ("quadrotor", "sac_vs_dc", filter_dict([], ["^sac$", "^(baseline-)?atacom_sac_dc$"])),

    # ("planar_air_hockey", "atacom_vs_dc", filter_dict([], ["^(baseline-)?atacom_sac$", "^(baseline-)?atacom_sac_dc$"])),
    # ("planar_air_hockey", "sac_vs_dc", filter_dict([], ["^sac$", "^(baseline-)?atacom_sac_dc$"])),
    # ("planar_air_hockey", "datacom_vs_dc", filter_dict([], ["^datacom_sac$", "^datacom_sac_dc$"])),

    # ("planar_air_hockey_vel", "atacom_vs_dc", filter_dict([], ["^(baseline-)?atacom_sac$", "^(baseline-)?atacom_sac_dc$"])),
    # ("planar_air_hockey_vel", "sac_vs_dc", filter_dict([], ["^sac$", "^(baseline-)?atacom_sac_dc$"])),
    # ("planar_air_hockey_vel", "datacom_vs_dc", filter_dict([], ["^datacom_sac$", "^datacom_sac_dc$"])),
]


if __name__ == '__main__':
    entity = os.environ.get("WANDB_ENTITY", "paolo-magliano")

    performance_metrics = ["R", "J"]
    safety_metrics = ["sum_cost", "max_cost", "violation_rate"]

    for project, name, filter in plots_list:
        data_path, plot_path = make_path(project, name)

        download_run_history(entity, project, data_path, samples=1000, filters=filter)\

        print(f"Plot {project} project")

        metrics, run_metrics = group_run_histories_by_key(entity, project, data_path, group_key=["group"], filters=filter)

        for method in metrics["R"].keys():
            print(f"\t{method} with {len(metrics['R'][method].columns)} seeds: {sorted(metrics['R'][method].columns)}")

        if "quadrotor" in project:
            performance_metrics.append("dist_to_target")
        elif "hockey" in project:
            performance_metrics.append("puck_vel")
            performance_metrics.append("success_rate")

        all_metrics = performance_metrics + safety_metrics + ["episode_length"]

        hp = None
        if "beta" in name:
            hp = "beta"
            hp_method = "atacom"
        if "delta" in name and "fixed_delta" not in name:
            hp = "delta"
            hp_method = "datacom"

        if hp is None:
            for key in all_metrics:
                if key in metrics.keys():
                    plot("learning", metrics, key, steps_per_epoch=10000, save_dir=plot_path)
                    if "cost" in key:
                        plot("learning", metrics, key, steps_per_epoch=10000, save_dir=plot_path, logscale=True)
        else:   
            for p_metric in performance_metrics:
                path = plot_path + "/" + p_metric
                if not os.path.exists(path):
                    os.makedirs(path)
                for s_metric in safety_metrics:
                    if s_metric in run_metrics.keys() and p_metric in run_metrics.keys():
                        hp_metrics = process_run_metrics(run_metrics, hp_method, hp, p_metric, s_metric)
                        plot(hp, hp_metrics, p_metric, s_metric, save_dir=path)
                    # plot_hp(hp_metrics, f'Beta {s_metric.replace("_", " ").title()}', s_metric.replace("_", " ").title(), p_metric.replace("_", " ").title(), save_dir=path)




