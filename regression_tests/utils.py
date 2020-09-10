import pytest
from training import setup_models
from colearn_examples.config import TrainingData


@pytest.fixture
def data_provider(request):
    def wrapper(config):
        name = str(config.n_learners)+getattr(config, "_test_id", "")
        path = "data/{}/{}".format(config.data, name)
        val = request.config.cache.get(path, None)
        print("************* PATH ", path, " cache? ", val is None)
        if val is None:
            if config.data == TrainingData.XRAY:
                from colearn_examples.xray import split_to_folders
            elif config.data == TrainingData.MNIST:
                from colearn_examples.mnist import split_to_folders
            elif config.data == TrainingData.FRAUD:
                from colearn_examples.fraud import split_to_folders
            else:
                raise Exception("Unknown task: %s" % config.data)
            val = split_to_folders(
                config, config.data_dir
            )
            request.config.cache.set(path, val)
        return val
    return wrapper


@pytest.fixture
def learner_provider(request, data_provider):
    def wrapper(config):
        name = str(config.n_learners)+getattr(config, "_test_id", "")
        path = "learners/{}/{}".format(config.data, name)
        val = request.config.cache.get(path, None)
        if val is None:
            if config.data == TrainingData.XRAY:
                from colearn_examples.xray import prepare_single_client, XrayConfig
                model_config = XrayConfig()
            elif config.data == TrainingData.MNIST:
                from colearn_examples.mnist import prepare_single_client, MNISTConfig
                model_config = MNISTConfig()
            elif config.data == TrainingData.FRAUD:
                from colearn_examples.fraud import prepare_single_client, FraudConfig
                model_config = FraudConfig()
            else:
                raise Exception("Unknown task: %s" % config.data)
            client_data_folders_list = data_provider(config)
            val = setup_models(model_config, client_data_folders_list, prepare_single_client)
        return val
    return wrapper
