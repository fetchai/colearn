# ------------------------------------------------------------------------------
#
#   Copyright 2021 Fetch.AI Limited
#
#   Licensed under the Creative Commons Attribution-NonCommercial International
#   License, Version 4.0 (the "License"); you may not use this file except in
#   compliance with the License. You may obtain a copy of the License at
#
#       http://creativecommons.org/licenses/by-nc/4.0/legalcode
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------
import matplotlib.axes._axes as mpl_ax
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from colearn.utils.results import Results


class ColearnPlot:
    def __init__(self, score_name: str = "user-defined score", draw_time=0.01):
        self.score_name = score_name
        plt.ion()
        self.results_axes: mpl_ax.Axes = plt.subplot(2, 1, 1, label="sub1")
        self.votes_axes: mpl_ax.Axes = plt.subplot(2, 1, 2, label="sub2")
        plt.show()
        self.fig_invalid = False
        self.draw_time = draw_time  # pause required for plt.draw() to complete

    def _remake_axes(self):
        self.results_axes: mpl_ax.Axes = plt.subplot(2, 1, 1, label="sub1")
        self.votes_axes: mpl_ax.Axes = plt.subplot(2, 1, 2, label="sub2")
        plt.show()
        self.fig_invalid = False

    def plot_results_and_votes(self, results):
        self.plot_results(results)
        self.plot_votes(results)

    def plot_results(self, results):
        if self.fig_invalid:
            self._remake_axes()

        # Prepare data for plotting
        results.process_statistics()

        self.results_axes.clear()

        self.results_axes.set_xlabel("training round")
        self.results_axes.set_ylabel(self.score_name)

        self.results_axes.set_xlim(-0.5, len(results.mean_test_scores) - 0.5)
        self.results_axes.set_xticks(np.arange(0, len(results.mean_test_scores), step=1))

        n_rounds = len(results.data)
        n_learners = len(results.data[0].vote_scores)
        for i in range(n_learners):
            self.results_axes.plot(
                range(n_rounds),
                results.h_test_scores[i],
                "b--",
                alpha=0.5,
                label=f"test {self.score_name}",
            )
            self.results_axes.plot(
                range(n_rounds),
                results.h_vote_scores[i],
                "r--",
                alpha=0.5,
                label=f"vote {self.score_name}",
            )

        (line_mean_test_score,) = self.results_axes.plot(
            range(n_rounds),
            results.mean_test_scores,
            "b",
            linewidth=3,
            label=f"mean test {self.score_name}",
        )
        (line_mean_vote_score,) = self.results_axes.plot(
            range(n_rounds),
            results.mean_vote_scores,
            "r",
            linewidth=3,
            label=f"mean vote {self.score_name}",
        )
        self.results_axes.legend(handles=[line_mean_test_score, line_mean_vote_score])

        plt.draw()
        plt.pause(self.draw_time)

    def plot_votes(self, results: Results):
        if self.fig_invalid:
            self._remake_axes()
        self.votes_axes.clear()

        results_list = results.data

        votes_array = np.array([res.votes for res in results_list])

        votes_array = votes_array.transpose()

        coloured_votes_array = np.zeros((votes_array.shape[0], votes_array.shape[1], 3), dtype=np.int)

        green_colour = np.array([204, 255, 204], dtype=np.int)
        red_colour = np.array([255, 153, 153], dtype=np.int)
        grey_colour = np.array([53, 53, 53], dtype=np.int)

        # Default fill: grey colour. Green and red will not match to 'None' in the votes_array
        coloured_votes_array[votes_array == 1] = green_colour
        coloured_votes_array[votes_array == 0] = red_colour
        coloured_votes_array[(votes_array != 1) & (votes_array != 0)] = grey_colour

        # make extra legend entries
        red_patch_handle = mpatches.Patch(color=red_colour / 256, label='Negative vote')
        green_patch_handle = mpatches.Patch(color=green_colour / 256, label='Positive vote')
        grey_patch_handle = mpatches.Patch(color=grey_colour / 256, label='Did not vote')

        self.votes_axes.imshow(coloured_votes_array, aspect="auto", interpolation='nearest')

        n_learners = votes_array.shape[0]
        n_rounds = votes_array.shape[1]

        self.votes_axes.set_xticks(range(n_rounds))

        ticks = ["Learner " + str(i) for i in range(n_learners)]
        self.votes_axes.set_yticks(range(n_learners))
        self.votes_axes.set_yticklabels(ticks)

        pos_xs = []
        pos_ys = []
        neg_xs = []
        neg_ys = []
        for i, res in enumerate(results.data[1:]):
            if res.vote:
                pos_xs.append(i + 1)
                pos_ys.append(res.block_proposer)
            else:
                neg_xs.append(i + 1)
                neg_ys.append(res.block_proposer)

        self.votes_axes.scatter(pos_xs, pos_ys, marker=r"$\checkmark$", c="green", s=150,
                                label="Proposer and positive overall vote")
        self.votes_axes.scatter(neg_xs, neg_ys, marker="X", c="red", s=150,
                                label="Proposer and negative overall vote")
        self.votes_axes.set_xlabel("training round")

        handles, _ = self.votes_axes.get_legend_handles_labels()
        self.votes_axes.legend(handles=handles + [green_patch_handle, red_patch_handle, grey_patch_handle])

        # Gridlines based on minor ticks
        self.votes_axes.set_xticks(np.arange(-0.5, n_rounds, 1), minor=True)
        self.votes_axes.set_yticks(np.arange(-0.5, n_learners, 1), minor=True)
        self.votes_axes.grid(which="minor", color="w", linestyle="-", linewidth=2)

        plt.draw()
        plt.pause(self.draw_time)

    def block(self):
        if self.fig_invalid:
            raise Exception("Previous call to .block() made the plot invalid.")
        plt.show(block=True)
        # closing the plot makes the figure and axes invalid
        self.fig_invalid = True
