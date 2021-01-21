import matplotlib.axes._axes as mpl_ax
import matplotlib.pyplot as plt
import numpy as np

from colearn.utils.results import Results

import sys

# Global module-wide axes instances
this = sys.modules[__name__]
this.results_axes = None
this.vote_axes = None


def process_statistics(results: Results, n_learners: int):
    results.h_test_accuracies = []
    results.h_vote_accuracies = []

    results.mean_test_accuracies = []
    results.mean_vote_accuracies = []

    results.max_test_accuracies = []
    results.max_vote_accuracies = []

    for r in range(len(results.data)):
        results.mean_test_accuracies.append(
            np.mean(np.array(results.data[r].test_scores))
        )
        results.mean_vote_accuracies.append(
            np.mean(np.array(results.data[r].vote_scores))
        )
        results.max_test_accuracies.append(np.max(np.array(results.data[r].test_scores)))
        results.max_vote_accuracies.append(np.max(np.array(results.data[r].vote_scores)))

    # gather individual scores
    for i in range(n_learners):
        results.h_test_accuracies.append([])
        results.h_vote_accuracies.append([])

        for r in range(len(results.data)):
            results.h_test_accuracies[i].append(results.data[r].test_scores[i])
            results.h_vote_accuracies[i].append(results.data[r].vote_scores[i])

    results.highest_test_accuracy = np.max(np.array(results.h_test_accuracies))
    results.highest_vote_accuracy = np.max(np.array(results.h_vote_accuracies))

    results.highest_mean_test_accuracy = np.max(results.mean_test_accuracies)
    results.highest_mean_vote_accuracy = np.max(results.mean_vote_accuracies)

    results.current_mean_test_accuracy = results.mean_test_accuracies[-1]
    results.current_mean_vote_accuracy = results.mean_vote_accuracies[-1]

    results.current_max_test_accuracy = results.max_test_accuracies[-1]
    results.current_max_vote_accuracy = results.max_vote_accuracies[-1]

    results.mean_mean_test_accuracy = np.mean(np.array(results.h_test_accuracies))
    results.mean_mean_vote_accuracy = np.mean(np.array(results.h_vote_accuracies))


def plot_results(results,
                 n_learners: int,
                 block=False,
                 score_name="user-defined score"):
    # Prepare data for plotting
    process_statistics(results, n_learners)

    plt.ion()
    plt.show(block=False)

    if this.results_axes is None:
        this.results_axes = plt.subplot(2, 1, 1, label="sub1")

    assert isinstance(this.results_axes, mpl_ax.Axes)  # gets rid of IDE errors
    this.results_axes.clear()

    this.results_axes.set_xlabel("training epoch")
    this.results_axes.set_ylabel(score_name)

    this.results_axes.set_xlim(-0.5, len(results.mean_test_accuracies) - 0.5)
    this.results_axes.set_xticks(np.arange(0, len(results.mean_test_accuracies), step=1))

    epochs = range(len(results.mean_test_accuracies))

    for i in range(n_learners):
        this.results_axes.plot(
            epochs,
            results.h_test_accuracies[i],
            "b--",
            alpha=0.5,
            label=f"test {score_name}",
        )
        this.results_axes.plot(
            epochs,
            results.h_vote_accuracies[i],
            "r--",
            alpha=0.5,
            label=f"vote {score_name}",
        )

    (line_mean_test_acc,) = this.results_axes.plot(
        epochs,
        results.mean_test_accuracies,
        "b",
        linewidth=3,
        label=f"mean test {score_name}",
    )
    (line_mean_vote_acc,) = this.results_axes.plot(
        epochs,
        results.mean_vote_accuracies,
        "r",
        linewidth=3,
        label=f"mean vote {score_name}",
    )
    this.results_axes.legend(handles=[line_mean_test_acc, line_mean_vote_acc])

    if block is False:
        plt.draw()
        plt.pause(0.01)
    else:
        plt.show(block=True)


def plot_votes(results: Results, block=False):
    plt.ion()
    plt.show(block=False)

    if this.vote_axes is None:
        this.vote_axes = plt.subplot(2, 1, 2, label="sub2")
    assert isinstance(this.vote_axes, mpl_ax.Axes)  # gets rid of IDE errors
    this.vote_axes.clear()

    results_list = results.data

    data = np.array([res.votes for res in results_list])

    data = data.transpose()
    this.vote_axes.matshow(data, aspect="auto", vmin=0, vmax=1)

    n_learners = data.shape[0]
    n_epochs = data.shape[1]

    # draw gridlines
    this.vote_axes.set_xticks(range(n_epochs))
    this.vote_axes.set_yticklabels([""] + ["Learner " + str(i) for i in range(n_learners)])

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

    this.vote_axes.scatter(pos_xs, pos_ys, marker="*", s=150, label="Positive overall vote")
    this.vote_axes.scatter(neg_xs, neg_ys, marker="X", s=150, label="Negative overall vote")
    this.vote_axes.set_xlabel("training epoch")
    this.vote_axes.legend()

    # Gridlines based on minor ticks
    this.vote_axes.set_xticks(np.arange(-0.5, n_epochs, 1), minor=True)
    this.vote_axes.set_yticks(np.arange(-0.5, n_learners, 1), minor=True)
    this.vote_axes.grid(which="minor", color="w", linestyle="-", linewidth=2)

    if block is False:
        plt.draw()
        plt.pause(0.01)
    else:
        plt.show(block=True)
