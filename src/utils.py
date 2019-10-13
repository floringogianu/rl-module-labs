""" Plot utils.

    Code original from: https://github.com/eeml2019/PracticalSessions/tree/master/rl
"""
import matplotlib.pyplot as plt
import numpy as np


map_from_action_to_subplot = lambda a: (2, 6, 8, 4)[a]
map_from_action_to_name = lambda a: ("up", "right", "down", "left")[a]


def plot_values(values, colormap="pink", vmin=-1, vmax=10):
    plt.imshow(
        values, interpolation="nearest", cmap=colormap, vmin=vmin, vmax=vmax
    )
    plt.yticks([])
    plt.xticks([])
    plt.colorbar(ticks=[vmin, vmax])


def plot_state_value(action_values, epsilon=0.1):
    q = action_values
    plt.figure(figsize=(4, 4))
    vmin = np.min(action_values)
    vmax = np.max(action_values)
    v = (1 - epsilon) * np.max(q, axis=-1) + epsilon * np.mean(q, axis=-1)
    plot_values(v, colormap="summer", vmin=vmin, vmax=vmax)
    plt.title("$v(s)$")


def plot_action_values(action_values, epsilon=0.1, large=False):
    q = action_values
    figsize = (12, 12) if large else (8, 8)
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    vmin = np.min(action_values)
    vmax = np.max(action_values)
    dif = vmax - vmin
    for a in [0, 1, 2, 3]:
        plt.subplot(3, 3, map_from_action_to_subplot(a))

        plot_values(q[..., a], vmin=vmin - 0.05 * dif, vmax=vmax + 0.05 * dif)
        action_name = map_from_action_to_name(a)
        plt.title(r"$q(s, \mathrm{" + action_name + r"})$")

    plt.subplot(3, 3, 5)
    v = (1 - epsilon) * np.max(q, axis=-1) + epsilon * np.mean(q, axis=-1)
    plot_values(v, colormap="summer", vmin=vmin, vmax=vmax)
    plt.title("$v(s)$")


def smooth(x, window=10):
    return (
        x[: window * (len(x) // window)]
        .reshape(len(x) // window, window)
        .mean(axis=1)
    )


def plot_stats(stats, window=10):
    plt.figure(figsize=(16, 4))
    plt.subplot(121)
    xline = range(0, len(stats.episode_lengths), window)
    plt.plot(xline, smooth(stats.episode_lengths, window=window))
    plt.ylabel("Episode Length")
    plt.xlabel("Episode Count")
    plt.subplot(122)
    plt.plot(xline, smooth(stats.episode_rewards, window=window))
    plt.ylabel("Episode Return")
    plt.xlabel("Episode Count")


def plot_policy(grid, policy, large=False):
    action_names = [
        r"$\uparrow$",
        r"$\rightarrow$",
        r"$\downarrow$",
        r"$\leftarrow$",
    ]
    grid.plot_grid(large=large)
    plt.title("Policy Visualization")
    for i in range(9):
        for j in range(10):
            action_name = action_names[policy[i, j]]
            plt.text(j, i, action_name, ha="center", va="center")


def plot_greedy_policy(grid, q, large=True, ax=None, title="Greedy Policy"):
    action_names = [
        r"$\uparrow$",
        r"$\rightarrow$",
        r"$\downarrow$",
        r"$\leftarrow$",
    ]
    greedy_actions = np.argmax(q, axis=2)

    grid.plot_grid(large=large, ax=ax)

    if ax is None:
        plt.title(title)
        for i in range(9):
            for j in range(10):
                action_name = action_names[greedy_actions[i, j]]
                plt.text(j, i, action_name, ha="center", va="center")
    else:
        ax.set_title(title)
        for i in range(9):
            for j in range(10):
                action_name = action_names[greedy_actions[i, j]]
                ax.text(j, i, action_name, ha="center", va="center")

