import pytest
from training import setup_models


@pytest.fixture
def data_provider(request):
    def wrapper(config):
        name = str(config.n_learners)+getattr(config, "_test_id", "")
        path = "data/{}/{}".format(config.data, name)
        val = request.config.cache.get(path, None)
        print("************* PATH ", path, " cache? ", val is None)
        if val is None:
            val = config.dataset.split_to_folders(
                config, config.main_data_dir
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
            client_data_folders_list = data_provider(config)
            all_learner_data = []
            for i in range(config.n_learners):
                print(type(client_data_folders_list[i]), client_data_folders_list[i])
                all_learner_data.append(
                    config.dataset.prepare_single_client(config, client_data_folders_list[i])
                )
            val = []
            clone_model = config.model_type(config, data=all_learner_data[0])
            for i in range(config.n_learners):
                model = clone_model.clone(data=all_learner_data[i])
                val.append(model)
        return val
    return wrapper
