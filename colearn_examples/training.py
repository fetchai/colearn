# type: ignore [no-redef]
from typing import Callable, List

from colearn_examples.config import ColearnConfig, ModelConfig, TrainingData, TrainingMode
from colearn_examples.utils.results import Result, Results

from colearn.basic_learner import BasicLearner, LearnerData
from colearn.ml_interface import ProposedWeights
from colearn.standalone_driver import run_one_epoch


def setup_models(config: ModelConfig, client_data_folders_list: List[str],
                 data_loading_func: Callable[[ModelConfig, str], LearnerData], test_data_dir=None):
    learner_datasets = []
    n_learners = len(client_data_folders_list)
    for i in range(n_learners):
        learner_datasets.append(
            data_loading_func(config, client_data_folders_list[i], test_data_dir=test_data_dir)
        )

    all_learner_models = []
    clone_model = config.model_type(config, data=learner_datasets[0])

    for i in range(n_learners):
        model = clone_model.clone(data=learner_datasets[i])

        all_learner_models.append(model)

    return all_learner_models


def initial_result(learners: List[BasicLearner]):
    result = Result()
    for learner in learners:
        proposed_weights = learner.test_model()  # type: ProposedWeights
        result.test_accuracies.append(proposed_weights.test_accuracy)
        result.vote_accuracies.append(proposed_weights.vote_accuracy)
        result.votes.append(True)
    return result


def collaborative_training_pass(learners: List[BasicLearner], vote_threshold,
                                epoch):
    print("Doing collaborative training pass")
    result = Result()

    proposed_weights_list, vote = run_one_epoch(epoch, learners,
                                                vote_threshold)
    result.vote = vote
    result.votes = [pw.vote for pw in proposed_weights_list]
    result.vote_accuracies = [pw.vote_accuracy for pw in
                              proposed_weights_list]
    result.test_accuracies = [pw.test_accuracy for pw in proposed_weights_list]
    result.block_proposer = epoch % len(learners)

    idx=0
    for l in learners:
        print("Eval config for node ", idx, ": ", l.evaluate_model(l.config.evaluation_config))
        idx+=1

    return result


def individual_training_pass(learners, epoch):
    print("Doing individual training pass")
    result = Result()

    # train all models
    for i, learner in enumerate(learners):
        print(f"Training learner #{i} epoch {epoch}")
        weights = learner.train_model()
        proposed_weights = learner.test_model(weights)
        learner.accept_weights(weights)

        result.votes.append(True)
        result.vote_accuracies.append(proposed_weights.vote_accuracy)
        result.test_accuracies.append(proposed_weights.test_accuracy)

    return result


def main(colearn_config: ColearnConfig, data_dir):
    results = Results()
    kwargs={}

    # pylint: disable=C0415
    if colearn_config.data == TrainingData.XRAY:
        from colearn_examples.xray import split_to_folders, display_statistics, \
            plot_results, plot_votes, prepare_single_client, XrayConfig
        model_config = XrayConfig(colearn_config.shuffle_seed)
    elif colearn_config.data == TrainingData.MNIST:
        from colearn_examples.mnist import split_to_folders, display_statistics, \
            plot_results, plot_votes, prepare_single_client, MNISTConfig
        model_config = MNISTConfig(colearn_config.shuffle_seed)
    elif colearn_config.data == TrainingData.FRAUD:
        from colearn_examples.fraud import split_to_folders, display_statistics, \
            plot_results, plot_votes, prepare_single_client, FraudConfig
        model_config = FraudConfig(colearn_config.shuffle_seed)
    elif colearn_config.data == TrainingData.CIFAR10:
        from colearn_examples.cifar10 import split_to_folders, display_statistics, \
            plot_results, plot_votes, prepare_single_client, CIFAR10Config
        model_config = CIFAR10Config(colearn_config.shuffle_seed)
    elif colearn_config.data == TrainingData.COVID:
        from colearn_examples.covid_xray import split_to_folders, display_statistics, \
            plot_results, plot_votes, prepare_single_client, CovidXrayConfig
        model_config = CovidXrayConfig(colearn_config.shuffle_seed)
        kwargs["test_ratio"] = 0.2
    else:
        raise Exception("Unknown task: %s" % colearn_config.data)

    # load, shuffle, clean, and split the data into n_learners
    data_split = [1 / colearn_config.n_learners for _ in range(colearn_config.n_learners)]
    client_data_folders_list = split_to_folders(
        data_dir, colearn_config.shuffle_seed, data_split, colearn_config.n_learners, **kwargs
    )
    test_data_dir = None
    if colearn_config.data == TrainingData.COVID:
        test_data_dir = client_data_folders_list[-1]
        del client_data_folders_list[-1]

    # setup n_learners duplicate models before training
    all_learner_models = setup_models(
        model_config, client_data_folders_list, prepare_single_client, test_data_dir
    )  # type: List[BasicLearner]

    # Get initial accuracy
    results.data.append(initial_result(all_learner_models))

    for i in range(colearn_config.n_epochs):
        if colearn_config.mode == TrainingMode.COLLABORATIVE:
            results.data.append(
                collaborative_training_pass(all_learner_models,
                                            colearn_config.vote_threshold, i)
            )
        elif colearn_config.mode == TrainingMode.INDIVIDUAL:
            results.data.append(individual_training_pass(all_learner_models, i))
        else:
            raise Exception("Unknown training mode")

        display_statistics(results, colearn_config, model_config, i + 1)

        if colearn_config.plot_results:
            # then make an updating graph
            plot_results(results, colearn_config, block=False)
            if colearn_config.mode == TrainingMode.COLLABORATIVE:
                plot_votes(results, block=False)

    if colearn_config.plot_results:
        if colearn_config.mode == TrainingMode.COLLABORATIVE:
            plot_results(results, colearn_config, block=False)
            plot_votes(results, block=True)
        else:
            plot_results(results, colearn_config, block=True)

if __name__ == "__main__":
    config = ColearnConfig(task=TrainingData.COVID, n_learners=5, n_epochs=30)
    config.vote_threshold=0.5
    main(config, "/Users/qati/code/colearn/examples/covid/covid_big_dset2")