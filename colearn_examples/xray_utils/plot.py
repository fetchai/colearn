import os
import tempfile
from pathlib import Path

# pylint: disable=W0622
from numpy import arange, array, max, mean

import matplotlib.axes._axes as mpl_ax
import matplotlib.pyplot as plt


from colearn_examples.config import TrainingMode, ModelConfig
from colearn_examples.utils.results import Results


def process_statistics(results, n_learners: int):
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
        results.max_test_accuracies.append(
            max(array(results.data[r].test_scores)))
        results.max_vote_accuracies.append(
            max(array(results.data[r].vote_scores)))

    # gather individual scores
    for i in range(n_learners):
        results.h_test_accuracies.append([])
        results.h_vote_accuracies.append([])

        for r in range(len(results.data)):
            results.h_test_accuracies[i].append(
                results.data[r].test_scores[i])
            results.h_vote_accuracies[i].append(
                results.data[r].vote_scores[i])

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


def plot_results(results: Results,
                 n_learners: int,
                 block=False):
    # Prepare data for plotting
    process_statistics(results, n_learners)

    plt.ion()
    plt.show(block=False)
    axes = plt.subplot(2, 1, 1, label="sub1")
    assert isinstance(axes, mpl_ax.Axes)  # gets rid of IDE errors
    axes.clear()

    axes.set_xlabel("training epoch")
    axes.set_ylabel("classification accuracy")
    axes.set_xlim(-0.5, len(results.mean_test_accuracies) - 0.5)

    axes.set_xticks(arange(0, len(results.mean_test_accuracies), step=1))

    epochs = range(len(results.mean_test_accuracies))

    for i in range(n_learners):
        axes.plot(
            epochs,
            results.h_test_accuracies[i],
            "b--",
            alpha=0.5,
            label="test accuracy",
        )
        axes.plot(
            epochs,
            results.h_vote_accuracies[i],
            "r--",
            alpha=0.5,
            label="vote accuracy",
        )

    (line_mean_test_acc,) = axes.plot(
        epochs,
        results.mean_test_accuracies,
        "b",
        linewidth=3,
        label="mean test accuracy",
    )
    (line_mean_vote_acc,) = axes.plot(
        epochs,
        results.mean_vote_accuracies,
        "r",
        linewidth=3,
        label="mean vote accuracy",
    )
    axes.legend(handles=[line_mean_test_acc, line_mean_vote_acc])

    if block is False:
        plt.draw()
        plt.pause(0.01)
    else:
        plt.show(block=True)


def display_statistics(
        results: Results,
        n_learners: int,
        mode: TrainingMode,
        vote_threshold: float,
        model_config: ModelConfig,
        current_epoch,
        filename=Path(tempfile.gettempdir()) / "stats_xray.tsv",
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

    data_str = "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t" \
               "%s\t%s\t%s\t%s\t%s\t%s\t" "%s\n" % (
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
