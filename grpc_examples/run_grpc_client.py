import json
import pickle

from colearn_grpc.grpc_learner_client import GRPCLearnerClient


def main():
    ml_system = GRPCLearnerClient("test", "127.0.0.1:9995")
    started = ml_system.start()
    if not started:
        print("Didn't start!")
        return

    ml_info = ml_system.get_supported_system()
    print(ml_info)

    model_architecture = "KERAS_MNIST"
    dataset = "KERAS_MNIST"

    model_parameters = ml_info['model_architectures'][model_architecture]
    data_parameters = ml_info['datasets'][dataset]
    print(model_parameters)
    print(data_parameters)
    d = json.loads(data_parameters)
    # d["location"] = "gs://mlfabric/0"
    d["location"] = "/tmp/mnist/0"
    data_parameters = json.dumps(d)

    mp = json.loads(model_parameters)
    # mp["model_type"] = "CONV2D"
    mp["model_type"] = "RESNET"
    mp["steps_per_epoch"] = 1
    model_parameters = json.dumps(mp)

    print(model_parameters)
    print(data_parameters)

    response = ml_system.setup_ml(
        dataset, data_parameters, model_architecture, model_parameters
    )
    print("SETUP: ", response)
    weights = ml_system.mli_propose_weights()
    # print(len(weights.weights))
    print(len((pickle.loads(weights.weights)).weights))

    test = ml_system.mli_test_weights(weights)
    print(test.vote, test.vote_score, test.test_score)

    accept = ml_system.mli_accept_weights(weights)
    print("accept response: ", accept)

    print("Done!")
    # ml_system.stop()  # todo: find out why stop hangs forever


if __name__ == "__main__":
    main()
