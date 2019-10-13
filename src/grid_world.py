""" Defines two different Grid worlds.

    Code original from: https://github.com/eeml2019/PracticalSessions/tree/master/rl
"""
import numpy as np
import matplotlib.pyplot as plt


class Grid:
    def __init__(self, discount=0.9, penalty_for_walls=-5):
        """ Gridworld

            Map definition:
                -1: wall
                0: empty, episode continues
                other: number indicates reward, episode will terminate
        """

        # fmt: off
        self._layout = np.array([
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1,  0,  0,  0,  0,  0, -1,  0,  0, -1],
            [-1,  0,  0,  0, -1,  0,  0,  0, 10, -1],
            [-1,  0,  0,  0, -1, -1,  0,  0,  0, -1],
            [-1,  0,  0,  0, -1, -1,  0,  0,  0, -1],
            [-1,  0,  0,  0,  0,  0,  0,  0,  0, -1],
            [-1,  0,  0,  0,  0,  0,  0,  0,  0, -1],
            [-1,  0,  0,  0,  0,  0,  0,  0,  0, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        ])
        # fmt: on
        self._start_state = (2, 2)
        self._goal_states = [(2, 8)]
        self._state = self._start_state
        self._number_of_states = np.prod(np.shape(self._layout))
        self._discount = discount
        self._penalty_for_walls = penalty_for_walls
        self._layout_dims = self._layout.shape

    @property
    def number_of_states(self):
        return self._number_of_states

    def plot_grid(self, large=False, ax=None):
        if ax is None:
            figsize = (4, 4) if large else (3, 3)
            plt.figure(figsize=figsize)
            ax = plt.gca()
        ax.imshow(self._layout <= -1, interpolation="nearest")
        ax.grid(0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"The {self.__class__.__name__}.")
        ax.text(
            self._start_state[1],
            self._start_state[0],
            r"$\mathbf{S}$",
            ha="center",
            va="center",
        )
        for goal in self._goal_states:
            ax.text(goal[1], goal[0], r"$\mathbf{G}$", ha="center", va="center")
        h, w = self._layout.shape
        for y in range(h - 1):
            ax.plot([-0.5, w - 0.5], [y + 0.5, y + 0.5], "-k", lw=2)
        for x in range(w - 1):
            ax.plot([x + 0.5, x + 0.5], [-0.5, h - 0.5], "-k", lw=2)

    def get_obs(self):
        y, x = self._state
        return y * self._layout.shape[1] + x

    def int_to_state(self, int_obs):
        x = int_obs % self._layout.shape[1]
        y = int_obs // self._layout.shape[1]
        return y, x

    def step(self, action):
        y, x = self._state

        if action == 0:  # up
            new_state = (y - 1, x)
        elif action == 1:  # right
            new_state = (y, x + 1)
        elif action == 2:  # down
            new_state = (y + 1, x)
        elif action == 3:  # left
            new_state = (y, x - 1)
        else:
            raise ValueError(
                "Invalid action: {} is not 0, 1, 2, or 3.".format(action)
            )

        new_y, new_x = new_state
        if self._layout[new_y, new_x] == -1:  # wall
            if self._penalty_for_walls is not None:
                reward = self._penalty_for_walls
            else:
                # rw is 0 if regular wall (instead of -1) else the actual value
                reward = (
                    0
                    if self._layout[new_y, new_x] == -1
                    else self._layout[new_y, new_x]
                )
            discount = self._discount
            new_state = (y, x)
        elif self._layout[new_y, new_x] == 0:  # empty cell
            reward = 0.0
            discount = self._discount
        else:  # a goal
            reward = self._layout[new_y, new_x]
            discount = 0.0
            new_state = self._start_state

        if hasattr(self, "_step_penalty"):
            reward += self._step_penalty

        self._state = new_state
        return reward, discount, self.get_obs()


class AltGrid(Grid):
    def __init__(self, discount=0.9, penalty_for_walls=-5):
        # fmt: off
        self._layout = np.array([
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1,  0,  0,  0,  0,  0,  0,  0,  0, -1],
            [-1,  0,  0,  0, -1, -1,  0,  0,  0, -1],
            [-1,  0,  0,  0, -1, -1,  0,  0,  0, -1],
            [-1,  0,  0,  0, -1, -1,  0,  0,  0, -1],
            [-1,  0,  0,  0,  0,  0,  0,  0,  0, -1],
            [-1,  0,  0,  0,  0,  0,  0,  0,  0, -1],
            [-1,  0, 10,  0,  0,  0,  0,  0,  0, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        ])
        # fmt: on
        self._start_state = (2, 2)
        self._goal_states = [(7, 2)]
        self._state = self._start_state
        self._number_of_states = np.prod(np.shape(self._layout))
        self._discount = discount
        self._penalty_for_walls = penalty_for_walls
        self._layout_dims = self._layout.shape


class TMaze(Grid):
    def __init__(self, discount=0.9, penalty_for_walls=-0.1):
        # fmt: off
        self._layout = np.array([
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1,  5, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1,  0, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1,  0, -1],
            [-1,  0,  0,  0,  0,  0,  0,  0,  0, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1,  0, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1,  0, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, 10, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        ])
        # fmt: on
        self._start_state = (4, 1)
        self._goal_states = [(1, 8), (7, 8)]
        self._state = self._start_state
        self._number_of_states = np.prod(np.shape(self._layout))
        self._discount = discount
        self._penalty_for_walls = penalty_for_walls
        self._layout_dims = self._layout.shape


class Cliff(Grid):
    def __init__(self, discount=0.9, penalty_for_walls=None, step_penalty=-1):
        # fmt: off
        self._layout = np.array([
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1,  0,  0,  0,  0,  0,  0,  0,  0, -1],
            [-1,  0,  0,  0,  0,  0,  0,  0,  0, -1],
            [-1,  0,  0,  0,  0,  0,  0,  0,  0, -1],
            [-1,  0,  0,  0,  0,  0,  0,  0,  0, -1],
            [-1,  0, -100, -100, -100, -100, -100, -100,  1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        ])
        # fmt: on
        self._start_state = (5, 1)
        self._goal_states = [(5, 8)]
        self._state = self._start_state
        self._number_of_states = np.prod(np.shape(self._layout))
        self._discount = discount
        self._penalty_for_walls = penalty_for_walls
        self._step_penalty = step_penalty
        self._layout_dims = self._layout.shape
