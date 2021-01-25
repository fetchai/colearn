import matplotlib.axes._axes as mpl_ax
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from colearn.utils.results import Results


class ColearnPlot:
    def __init__(self, n_learners: int, score_name: str = "user-defined score"):
        self.score_name = score_name
        self.n_learners = n_learners
        self.results_axes: mpl_ax.Axes = plt.subplot(2, 1, 1, label="sub1")
        self.votes_axes: mpl_ax.Axes = plt.subplot(2, 1, 2, label="sub2")

    def _process_statistics(self, results: Results):
        results.h_test_scores = []
        results.h_vote_scores = []

        results.mean_test_scores = []
        results.mean_vote_scores = []

        results.max_test_scores = []
        results.max_vote_scores = []

        for r in range(len(results.data)):
            results.mean_test_scores.append(
                np.mean(np.array(results.data[r].test_scores))
            )
            results.mean_vote_scores.append(
                np.mean(np.array(results.data[r].vote_scores))
            )
            results.max_test_scores.append(np.max(np.array(results.data[r].test_scores)))
            results.max_vote_scores.append(np.max(np.array(results.data[r].vote_scores)))

        # gather individual scores
        for i in range(self.n_learners):
            results.h_test_scores.append([])
            results.h_vote_scores.append([])

            for r in range(len(results.data)):
                results.h_test_scores[i].append(results.data[r].test_scores[i])
                results.h_vote_scores[i].append(results.data[r].vote_scores[i])

        results.highest_test_score = np.max(np.array(results.h_test_scores))
        results.highest_vote_score = np.max(np.array(results.h_vote_scores))

        results.highest_mean_test_score = np.max(results.mean_test_scores)
        results.highest_mean_vote_score = np.max(results.mean_vote_scores)

        results.current_mean_test_score = results.mean_test_scores[-1]
        results.current_mean_vote_score = results.mean_vote_scores[-1]

        results.current_max_test_score = results.max_test_scores[-1]
        results.current_max_vote_score = results.max_vote_scores[-1]

        results.mean_mean_test_score = np.mean(np.array(results.h_test_scores))
        results.mean_mean_vote_score = np.mean(np.array(results.h_vote_scores))

    def plot_results(self, results, block=False):
        # Prepare data for plotting
        self._process_statistics(results)

        plt.ion()
        plt.show(block=False)

        self.results_axes.clear()

        self.results_axes.set_xlabel("training round")
        self.results_axes.set_ylabel(self.score_name)

        self.results_axes.set_xlim(-0.5, len(results.mean_test_scores) - 0.5)
        self.results_axes.set_xticks(np.arange(0, len(results.mean_test_scores), step=1))

        rounds = range(len(results.mean_test_scores))

        for i in range(self.n_learners):
            self.results_axes.plot(
                rounds,
                results.h_test_scores[i],
                "b--",
                alpha=0.5,
                label=f"test {self.score_name}",
            )
            self.results_axes.plot(
                rounds,
                results.h_vote_scores[i],
                "r--",
                alpha=0.5,
                label=f"vote {self.score_name}",
            )

        (line_mean_test_score,) = self.results_axes.plot(
            rounds,
            results.mean_test_scores,
            "b",
            linewidth=3,
            label=f"mean test {self.score_name}",
        )
        (line_mean_vote_score,) = self.results_axes.plot(
            rounds,
            results.mean_vote_scores,
            "r",
            linewidth=3,
            label=f"mean vote {self.score_name}",
        )
        self.results_axes.legend(handles=[line_mean_test_score, line_mean_vote_score])

        if block is False:
            plt.draw()
            plt.pause(0.01)
        else:
            plt.show(block=True)

    def plot_votes(self, results: Results, block=False):
        plt.ion()
        plt.show(block=False)

        self.votes_axes.clear()

        results_list = results.data

        data = np.array([res.votes for res in results_list])

        data = data.transpose()
        self.votes_axes.matshow(data, aspect="auto", vmin=0, vmax=1)

        n_learners = data.shape[0]
        n_rounds = data.shape[1]

        # draw gridlines
        self.votes_axes.set_xticks(range(n_rounds))

        ticks = [""] + ["Learner " + str(i) for i in range(n_learners)] + [""]
        ticks_loc = self.votes_axes.get_yticks().tolist()
        self.votes_axes.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
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

        self.votes_axes.scatter(pos_xs, pos_ys, marker="*", s=150, label="Positive overall vote")
        self.votes_axes.scatter(neg_xs, neg_ys, marker="X", s=150, label="Negative overall vote")
        self.votes_axes.set_xlabel("training round")
        self.votes_axes.legend()

        # Gridlines based on minor ticks
        self.votes_axes.set_xticks(np.arange(-0.5, n_rounds, 1), minor=True)
        self.votes_axes.set_yticks(np.arange(-0.5, n_learners, 1), minor=True)
        self.votes_axes.grid(which="minor", color="w", linestyle="-", linewidth=2)

        if block is False:
            plt.draw()
            plt.pause(0.01)
        else:
            plt.show(block=True)
