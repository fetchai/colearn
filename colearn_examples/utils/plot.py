import os
import tempfile
from pathlib import Path

import matplotlib.axes._axes as mpl_ax
import matplotlib.pyplot as plt

from numpy import arange, array, max, mean

from colearn_examples.config import TrainingMode, ModelConfig
from colearn_examples.utils.results import Results


def process_statistics(results: Results, n_learners: int):
    results.h_test_accuracies = []
    results.h_vote_accuracies = []

    results.mean_test_accuracies = []
    results.mean_vote_accuracies = []

    results.max_test_accuracies = []
    results.max_vote_accuracies = []

    for r in range(len(results.data)):
        results.mean_test_accuracies.append(
            mean(array(results.data[r].test_scores))
        )
        results.mean_vote_accuracies.append(
            mean(array(results.data[r].vote_scores))
        )
        results.max_test_accuracies.append(max(array(results.data[r].test_scores)))
        results.max_vote_accuracies.append(max(array(results.data[r].vote_scores)))

    # gather individual scores
    for i in range(n_learners):
        results.h_test_accuracies.append([])
        results.h_vote_accuracies.append([])

        for r in range(len(results.data)):
            results.h_test_accuracies[i].append(results.data[r].test_scores[i])
            results.h_vote_accuracies[i].append(results.data[r].vote_scores[i])

    results.highest_test_accuracy = max(array(results.h_test_accuracies))
    results.highest_vote_accuracy = max(array(results.h_vote_accuracies))

    results.highest_mean_test_accuracy = max(results.mean_test_accuracies)
    results.highest_mean_vote_accuracy = max(results.mean_vote_accuracies)

    results.current_mean_test_accuracy = results.mean_test_accuracies[-1]
    results.current_mean_vote_accuracy = results.mean_vote_accuracies[-1]

    results.current_max_test_accuracy = results.max_test_accuracies[-1]
    results.current_max_vote_accuracy = results.max_vote_accuracies[-1]

    results.mean_mean_test_accuracy = mean(array(results.h_test_accuracies))
    results.mean_mean_vote_accuracy = mean(array(results.h_vote_accuracies))


def display_statistics(
        results: Results,
        n_learners: int,
        mode: TrainingMode,
        vote_threshold: float,
        model_config: ModelConfig,
        current_epoch,
        filename=Path(tempfile.gettempdir()) / "stats_mnist.tsv",
):
    print("Statistics")

    # Prepare data for statistics
    process_statistics(results, n_learners)

    header_str = (
        "MODEL_TYPE\tHOSPITALS\tEPOCHS\tL_RATE\tCOLLAB\tVOTE_THRESHOLD"
        "\tTRAIN_RATIO\tVAL_BATCHES\tTEST_RATIO\tHIGHEST_TEST_ACCURACY"
        "\tHIGHEST_VOTE_ACCURACY"
        "\tHIGHEST_MEAN_TEST_ACCURACY\tHIGHEST_MEAN_VOTE_ACCURACY"
        "\tTRAIN_AUGMENTATION\tBATCH_SIZE\tBATCHES_PER_EPOCH"
        "\tCURRENT_MEAN_TEST_ACCURACY\tCURRENT_MEAN_VOTE_ACCURACY"
        "\tCURRENT_MAX_TEST_ACCURACY\tCURRENT_MAX_VOTE_ACCURACY"
        "\tMEAN_MEAN_TEST_ACCURACY\tMEAN_MEAN_VOTE_ACCURACY\tNON_IID"
    )

    data_str = "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t" "%s\n" % (
        model_config.model_type,
        n_learners,
        current_epoch,
        model_config.l_rate,
        mode,
        vote_threshold,
        model_config.train_ratio,
        model_config.val_batches,
        model_config.test_ratio,
        results.highest_test_accuracy,
        results.highest_vote_accuracy,
        results.highest_mean_test_accuracy,
        results.highest_mean_vote_accuracy,
        model_config.train_augment,
        model_config.batch_size,
        model_config.steps_per_epoch,
        results.current_mean_test_accuracy,
        results.current_mean_vote_accuracy,
        results.current_max_test_accuracy,
        results.current_max_vote_accuracy,
        results.mean_mean_test_accuracy,
        results.mean_mean_vote_accuracy,
    )

    print(header_str)
    print(data_str)

    new_file = False
    if not os.path.isfile(filename):
        new_file = True

    f = open(filename, "a")
    # Write header if file was empty
    if new_file:
        f.write(header_str + "\n")

    f.write(data_str)
    f.close()


def plot_results(results,
                 n_learners: int,
                 block=False,
                 score_name="user-defined score"):
    # Prepare data for plotting
    process_statistics(results, n_learners)

    plt.ion()
    plt.show(block=False)
    axes = plt.subplot(2, 1, 1, label="sub1")
    assert isinstance(axes, mpl_ax.Axes)  # gets rid of IDE errors
    axes.clear()

    axes.set_xlabel("training epoch")
    axes.set_ylabel(score_name)

    axes.set_xlim(-0.5, len(results.mean_test_accuracies) - 0.5)
    axes.set_xticks(arange(0, len(results.mean_test_accuracies), step=1))

    epochs = range(len(results.mean_test_accuracies))

    for i in range(n_learners):
        axes.plot(
            epochs,
            results.h_test_accuracies[i],
            "b--",
            alpha=0.5,
            label=f"test {score_name}",
        )
        axes.plot(
            epochs,
            results.h_vote_accuracies[i],
            "r--",
            alpha=0.5,
            label=f"vote {score_name}",
        )

    (line_mean_test_acc,) = axes.plot(
        epochs,
        results.mean_test_accuracies,
        "b",
        linewidth=3,
        label=f"mean test {score_name}",
    )
    (line_mean_vote_acc,) = axes.plot(
        epochs,
        results.mean_vote_accuracies,
        "r",
        linewidth=3,
        label=f"mean vote {score_name}",
    )
    axes.legend(handles=[line_mean_test_acc, line_mean_vote_acc])

    if block is False:
        plt.draw()
        plt.pause(0.01)
    else:
        plt.show(block=True)


def plot_votes(results: Results, block=False):
    plt.ion()
    plt.show(block=False)
    axes = plt.subplot(2, 1, 2, label="sub2")
    assert isinstance(axes, mpl_ax.Axes)  # gets rid of IDE errors
    axes.clear()

    results_list = results.data

    data = array([res.votes for res in results_list])

    data = data.transpose()
    axes.matshow(data, aspect="auto", vmin=0, vmax=1)

    n_learners = data.shape[0]
    n_epochs = data.shape[1]

    # draw gridlines
    axes.set_xticks(range(n_epochs))
    axes.set_yticklabels([""] + ["Learner " + str(i) for i in range(n_learners)])

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

    axes.scatter(pos_xs, pos_ys, marker="*", s=150, label="Positive overall vote")
    axes.scatter(neg_xs, neg_ys, marker="X", s=150, label="Negative overall vote")
    axes.set_xlabel("training epoch")
    axes.legend()

    axes1 = plt.subplot(2, 1, 1, label="sub1")
    assert isinstance(axes1, mpl_ax.Axes)
    pos = axes1.get_position()
    pos2 = axes.get_position()
    axes.set_position([pos.x0, pos2.y0, pos.width, pos2.height])

    # Gridlines based on minor ticks
    axes.set_xticks(arange(-0.5, n_epochs, 1), minor=True)
    axes.set_yticks(arange(-0.5, n_learners, 1), minor=True)
    axes.grid(which="minor", color="w", linestyle="-", linewidth=2)

    if block is False:
        plt.draw()
        plt.pause(0.01)
    else:
        plt.show(block=True)
