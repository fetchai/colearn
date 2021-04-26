# gRPC tutorial
This tutorial explains how to set up the gRPC learner server.
It assumes that you can already run colearn locally, and that you have already defined your own models and dataloaders 
(if you're going to do so).
If you haven't done this then see the tutorials in the [Getting Started](./intro_tutorial_mli.md) section.

## Architecture of colearn
There are two main parts to a collective learning system: the learner and the backend.
The backend controls the learner, and manages the smart contracts and IPFS, and acts as a control hub for 
all the associated learners.
The learner is the part that executes machine learning code. 
This consists of proposing, evaluating and accepting new weights as detailed in the Machine Learning Interface.
The learner and the backend communicate via [gRPC](https://grpc.io); 
the learner runs a gRPC server, and the backend runs a gRPC client that makes requests of the learner.
This separation means that the learner can run on specialised hardware (e.g. a compute server) and does not need to 
be co-located with the backend.

## Architecture of gRPC server
The gRPC interface is defined in 
[colearn_grpc/proto/interface.proto]({{ repo_root }}/colearn_grpc/proto/interface.proto).
This defines the functions that the gRPC server exposes and the format for messages between the server and the client.

As we covered in the earlier tutorials, the machine learning part of colearn is contained inside the
`MachineLearningInterface` (MLI).
To recap: the MLI provides methods for proposing, evaluating and accepting weights.
If you want to use your own models with colearn then you need to write an object that implements the MLI 
(for example, an instance of a python class that inherits from `MachineLearningInterface`).
For more about the MLI see the [MLI tutorial](./intro_tutorial_mli.md).

The gRPC server has an MLI factory, and it uses its MLI factory to make objects that implement 
the `MachineLearningInterface`.
The MLI factory needs to implement the MLI factory interface.
You could write your own MLI factory, but it's easier to use the one we provide.
Below we will discuss the MLI factory interface and then talk about how to use the example factory.

## MLI Factory interface
The MLI Factory (as the name suggests) is a factory class for creating objects that implement the machine learning 
interface:
```Python 
{!../colearn_grpc/mli_factory_interface.py!} 
```
The MLI Factory stores the constructors for dataloaders and models and also a list of the dataloaders that 
are compatible with each model.
Each constructor is stored under a specific name.
For example, "KERAS_MNIST_MODEL" is the model for keras mnist.
The gRPC server uses the MLI factory to construct MLI objects.
The MLI Factory needs to implement four methods:

* get_models - returns the names of the models that are registered with the factory and their parameters.
* get_dataloaders - returns the names of the dataloaders that are registered with the factory and their parameters.
* get_compatibilities - returns a list of dataloaders for each model that can be used with that model.
* get_mli - takes the name and parameters for the model and dataloader and constructs the MLI object. 
  Returns the MLI object.


## Using the example MLI Factory
The example MLI factory is defined in 
[colearn_grpc/example_mli_factory.py]({{ repo_root }}/colearn_grpc/example_mli_factory.py).
It stores the models and dataloaders that it knows about in factoryRegistry.py
To add a new model and dataloader to the factory you need to do the following things:

1. Define a function that loads the dataset given the location of the dataset.
2. Define a function that takes in the dataset and loads the MLI model. 
3. Register both these functions with the factory registry. 

Registering a dataloader looks like this:
```python
@FactoryRegistry.register_dataloader(dataloader_tag)
def prepare_data_loaders(location: str,
                         train_ratio: float = 0.9,
                         batch_size: int = 32) -> Tuple[PrefetchDataset, PrefetchDataset]:
```
Registering a model is similar, but you additionally have to specify the dataloaders that this model is compatible with.
```python
@FactoryRegistry.register_model_architecture(model_tag, [dataloader_tag])
def prepare_learner(data_loaders: Tuple[PrefetchDataset, PrefetchDataset],
                    steps_per_epoch: int = 100,
                    vote_batches: int = 10,
                    learning_rate: float = 0.001
                    ) -> KerasLearner:
```

You can see an example of how to do this in [colearn_examples/grpc/mnist_grpc.py]({{ repo_root }}/colearn_examples/grpc/mnist_grpc.py).
The FactoryRegistry decorators get evaluated when the functions are imported, so ensure that the functions are imported 
before constructing the gRPC server (more on that later). 
   
Constraints on the dataloader function:

1. The first parameter should be a mandatory parameter called "location" which stores the location of the dataset.
2. The subsequent parameters should have default arguments.
3. The return type should be specified with a type annotation, and this should be the same type that is expected by the 
   model functions that use this dataloader.
4. The arguments that you pass to the dataloader function must be 
   [JSON-encodable](https://docs.python.org/3.7/library/json.html). 
   Native python types are fine (e.g. str, dict, list, float). 

Constraints on the model function:

1. The first parameter should be a mandatory parameter called "data_loaders". 
   This must have the same type as the return type of the compatible dataloaders.
2. The subsequent parameters should have default arguments.
3. The return type of model_function should be `MachineLearningInterface` or a subclass of it (e.g. `KerasLearner`).
4. The dataloaders listed as being compatible with the model should already be registered with FactoryRegistry before
   the model is registered. 
4. The arguments that you pass to the model function must be 
   [JSON-encodable](https://docs.python.org/3.7/library/json.html).
   Native python types are fine (e.g. str, dict, list, float). 

## Making it all work together
It can be challenging to ensure that all the parts talk to each other, so we have provided some examples and 
helper scripts.
It is recommended to first make an all-in-one script following the example of 
[colearn_examples/grpc/mnist_grpc.py]({{ repo_root }}/colearn_examples/grpc/mnist_grpc.py).
Once this is working you can run [colearn_grpc/scripts/run_n_servers.py]({{ repo_root }}/colearn_grpc/scripts/run_n_servers.py) or 
[colearn_grpc/scripts/run_grpc_server.py]({{ repo_root }}/colearn_grpc/scripts/run_server.py) to run the server(s).
The script [colearn_grpc/scripts/probe_grpc_server.py]({{ repo_root }}/colearn_grpc/scripts/probe_grpc_server.py) will connect to a 
gRPC server and print the dataloaders and models that are registered on it (pass in the address as a parameter).
The client side of the gRPC communication can then be run using 
[colearn_examples/grpc/run_grpc_demo.py]({{ repo_root }}/colearn_examples/grpc/run_grpc_demo.py).
More details are given below.

A note about running tensorflow in multiple processes: on a system with a GPU, tensorflow will try to get all the GPU
memory when it starts up. 
This means that running tensorflow in multiple processes on the same machine will fail.
To prevent this happening, tensorflow should be told to use only the CPU by setting the environment variable
`CUDA_VISIBLE_DEVIES` to `-1`.
This can be done in a python script (before importing tensorflow) by using:
```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
```

## Testing locally with an all-in-one script
You can test this locally by following the example in 
[colearn_examples/grpc/mnist_grpc.py]({{ repo_root }}/colearn_examples/grpc/mnist_grpc.py).
Define your dataloader and model functions as specified above, and register them with the factory.
Then create n_learners gRPC servers:
```python
n_learners = 5
first_server_port = 9995
# make n servers
for i in range(n_learners):
    port = first_server_port + i
    server = GRPCServer(mli_factory=ExampleMliFactory(),
                        port=port)
    server_process = Process(target=server.run)
    server_process.start()
```

And then create n_learners gRPC clients:

```python
all_learner_models = []
for i in range(n_learners):
   port = first_server_port + i
   ml_system = ExampleGRPCLearnerClient(f"client {i}", f"127.0.0.1:{port}")
   ml_system.start()
   dataloader_params = {"location": data_folders[i]}
   ml_system.setup_ml(dataset_loader_name=dataloader_tag,
                      dataset_loader_parameters=json.dumps(dataloader_params),
                      model_arch_name=model_tag,
                      model_parameters=json.dumps({}))
   all_learner_models.append(ml_system)
```

`ExampleGRPCLearnerClient` inherits from the `MachineLearningInterface` so you can use it with the training functions 
as before:
```python
for round_index in range(n_rounds):
    results.data.append(
        collective_learning_round(all_learner_models,
                                  vote_threshold, round_index)
    )
```

## Testing remotely
We expect that the gRPC learner part will often be on a compute cluster and be separate from the gRPC client side.
To test the gRPC in a setup like this you can start the servers on the computer side and the client part separately.
For one gRPC server:
```bash
python3 ./grpc_examples/run_grpc_server.py --port 9995 --metrics_port 9091
```

For multiple gRPC servers:
```bash
python3 ./grpc_examples/run_n_grpc_servers.py --n_learners 5 --port 9995 --metrics_port 9091
```
The servers by default will start on port 9995 and use subsequent ports from there, so if three
servers are required they will run on ports 9995, 9996 and 9997.

If you have written your own dataloaders and models then you need to make sure that those functions are defined or 
imported before the server is created.
These are the imports of the default dataloaders and models in `grpc_examples/run_grpc_server.py`:
```python
# These are imported so that they are registered in the FactoryRegistry
import colearn_keras.keras_mnist
import colearn_keras.keras_cifar10
import colearn_pytorch.pytorch_xray
import colearn_pytorch.pytorch_covid_xray
import colearn_other.fraud_dataset
```

Once the gRPC server(s) are running, set up whatever networking and port forwarding is required.
You can check that the gRPC server is accessible by using the probe script:
```bash
python3 ./grpc_examples/probe_grps_server.py --port 9995
```
If the connection is successful this will print a list of the models and datasets registered on the server.
These are the defaults that are registered:
```
info: Attempt number 0 to connect to 127.0.0.1:9995
info: Successfully connected to 127.0.0.1:9995!
{'compatibilities': {'FRAUD': ['FRAUD'],
                     'KERAS_CIFAR10': ['KERAS_CIFAR10'],
                     'KERAS_MNIST': ['KERAS_MNIST'],
                     'KERAS_MNIST_RESNET': ['KERAS_MNIST'],
                     'PYTORCH_COVID_XRAY': ['PYTORCH_COVID_XRAY'],
                     'PYTORCH_XRAY': ['PYTORCH_XRAY']},
 'data_loaders': {'FRAUD': '{"train_ratio": 0.8}',
                  'KERAS_CIFAR10': '{"train_ratio": 0.9, "batch_size": 32}',
                  'KERAS_MNIST': '{"train_ratio": 0.9, "batch_size": 32}',
                  'PYTORCH_COVID_XRAY': '{"train_ratio": 0.8, "batch_size": 8, '
                                        '"no_cuda": false}',
                  'PYTORCH_XRAY': '{"test_location": null, "train_ratio": 0.96, '
                                  '"batch_size": 8, "no_cuda": false}'},
 'model_architectures': {'FRAUD': '{}',
                         'KERAS_CIFAR10': '{"steps_per_epoch": 100, '
                                          '"vote_batches": 10, '
                                          '"learning_rate": 0.001}',
                         'KERAS_MNIST': '{"steps_per_epoch": 100, '
                                        '"vote_batches": 10, "learning_rate": '
                                        '0.001}',
                         'KERAS_MNIST_RESNET': '{"steps_per_epoch": 100, '
                                               '"vote_batches": 10, '
                                               '"learning_rate": 0.001}',
                         'PYTORCH_COVID_XRAY': '{"learning_rate": 0.001, '
                                               '"steps_per_epoch": 40, '
                                               '"vote_batches": 10, "no_cuda": '
                                               'false, "vote_on_accuracy": '
                                               'true}',
                         'PYTORCH_XRAY': '{"learning_rate": 0.001, '
                                         '"steps_per_epoch": 40, '
                                         '"vote_batches": 10, "no_cuda": '
                                         'false, "vote_on_accuracy": true}'}}

```

Then run `python -m colearn_examples/grpc/run_grpc_demo.py` on the other side to run the usual demo.
The script takes as arguments the model name and dataset name that should be run, along with the number of learners
and the data location for each learner.
```bash
python -m colearn_examples/grpc/run_grpc_demo.py --n_learners 5 --dataloader_tag KERAS_MNIST --model_tag KERAS_MNIST \
--data_locations /tmp/mnist/0,/tmp/mnist/1,/tmp/mnist/2,/tmp/mnist/3,/tmp/mnist/4
```

## Using the MLI Factory interface
An alternative method of using your own dataloaders and models with the gRPC server is to use the MLI Factory interface.
This is defined in `colearn_grpc/mli_factory_interface.py`.
An example is given in `colearn_examples/grpc/mlifactory_grpc_mnist.py`.
The MLI Factory is implemented as shown:
```python
dataloader_tag = "KERAS_MNIST_EXAMPLE_DATALOADER"
model_tag = "KERAS_MNIST_EXAMPLE_MODEL"

class SimpleFactory(MliFactory):
    def get_dataloaders(self) -> Dict[str, Dict[str, Any]]:
        return {dataloader_tag: dict(train_ratio=0.9,
                                     batch_size=32)}

    def get_models(self) -> Dict[str, Dict[str, Any]]:
        return {model_tag: dict(steps_per_epoch=100,
                                vote_batches=10,
                                learning_rate=0.001)}

    def get_compatibilities(self) -> Dict[str, Set[str]]:
        return {model_tag: {dataloader_tag}}

    def get_mli(self, model_name: str, model_params: str, dataloader_name: str,
                dataset_params: str) -> MachineLearningInterface:
        dataloader_params = json.loads(dataset_params)
        data_loaders = prepare_data_loaders(**dataloader_params)

        model_params = json.loads(model_params)
        mli_model = prepare_learner(data_loaders=data_loaders, **model_params)
        return mli_model
```

An instance of the `SimpleFactory` class needs to be passed to the gRPC server on creation:
```python
n_learners = 5
first_server_port = 9995
# make n servers
server_processes = []
for i in range(n_learners):
    port = first_server_port + i
    server = GRPCServer(mli_factory=SimpleFactory(),
                        port=port)
    server_process = Process(target=server.run)
    print("starting server", i)
    server_process.start()
    server_processes.append(server_process)
```
The rest of the example follows the `grpc_mnist.py` example.
