# gRPC tutorial
This tutorial explains how to set up the gRPC learner server.
It assumes that you can already run colearn locally, and that you have already defined your own models and dataloaders 
(if you're going to do so).
If you haven't done this then see the tutorials in the Getting Started section.

## Architecture of colearn
There are two main parts to a collective learning system: the learner and the backend.
The backend controls the learner, and manages networking, communication with other collective learning nodes, IPFS and smart contracts. 
The learner is the part that executes machine learning code. 
This consists of proposing, evaluating and accepting new weights as detailed in the Machine Learning Interface.
The learner and the backend communicate via gRPC; 
the learner runs a gRPC server, and the backend runs a gRPC client that makes requests of the learner.
This separation means that the learner can run on specialised hardware, e.g. a compute server, and does not need to be co-located with the backend.

## Architecture of gRPC server
The gRPC interface is defined in colearn_grpc/proto/interface.proto.
This defines the functions that the gRPC server exposes and the format for messages between the server and the client.

grpc server has a mli factory, and it uses the mli factory to make the learner
mli factory needs to implement mli factory interface.
So you could write your own mli factory,
but it's easier to use the one we provide
Below we will discuss the mli factory and then talk about how to use the example factory.

## MLI Factory interface:
mli factory (as the name suggests) is a factory class for creating objects that implement the machine learning interface.
mlif stores the constructors for dataloaders and models and also the dataloaders that are compatible with each model.
Each construcotr is stored under a specific name.
For example, "KERAS_MNIST_MODEL" is the model for keras mnist.
The grpc server uses the mli factory to construct mli objects.
mlif needs to implement four methods:
* get_models - returns the names of the models that are registered with the factory and their parameters
* get_dataloaders - returns the names of the dataloaders that are registered with the factory and their parameters
* get compatibilities - returns a list of dataloaders for each model that can be used with that model
* get_mli - takes the name and parameters for the model and dataloader and contructs the mli object. Returns the mli object


## Using the example MLI Factory
The example mli factory is defined in colearn_grpc/example_mli_factory.py.
It stores the models and dataloaders that it knows about in factoryRegistry.py
To add a new model and dataloader to the factory you need to do the folowing things:
1. Define a function that loads the dataset given the location of the dataset.
2. Define a function that takes in the dataset and loads the mli model. 
2. Register both these functions with the factory registry. 

Registering a dataloader looks like this:
```python
@FactoryRegistry.register_dataloader(dataloader_tag)
def prepare_data_loaders(train_folder: str,
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

You can see an example of how to do this in {{ repo_root }}/colearn_grpc/mnist_grpc.py.
The FactoryRegistry decorators get evaluated when the functions are imported, so ensure that the functions are imported 
before constructing the gRPC server (more on that later). 
   
Constraints on the dataloader function:
1. The first parameter should be a mandatory parameter called "train_folder" which stores the location of the dataset.
2. The subsequent parameters should have default arguments.
3. The return type should be specified with a type annotation, and this should be the same type that is expected by the 
   model functions that use this dataloader.

Constraints on the model function:
1. The first parameter should be a mandatory parameter called "data_loaders". 
   This must have the same type as the return type of the compatible dataloaders.
2. The subsequent parameters should have default arguments.
3. The return type of model_function should be `MachineLearningInterface` or a subclass of it (e.g. `KerasLearner`).
4. The dataloaders listed as being compatible with the model should already be registered with FactoryRegistry before
   the model is registered. 

## Making it all work together
It can be challenging to ensure that all the parts talk to each other, so we have provided some examples and 
helper scripts.
It is recommended to first make an all-in-one script following the example of 
{{ repo_root }}/colearn_grpc/mnist_grpc.py.
Once this is working you can run grpc_examples/run_n_servers.py or run_server.py to run the server(s).
The script probe_grpc_server will connect to a gRPC server and print the dataloaders and models that are registered
on it (pass in the address as a parameter).


## Testing locally
You can test the locally by following the example in mnist_grpc.
Create n_learners gRPC servers:
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
    ml_system = GRPCLearnerClient(f"client {i}", f"127.0.0.1:{port}")
    started = ml_system.start()
    dataloader_params = {"train_folder": data_folders[i]}
    ml_system.setup_ml(dataset_loader_name=dataloader_tag,
                       dataset_loader_parameters=json.dumps(dataloader_params),
                       model_arch_name=model_tag,
                       model_parameters=json.dumps({}))
    all_learner_models.append(ml_system)
```

GRPCLearnerClient implements the machine learning interface, so you can use it with the training functions as before:
```python
for round_index in range(n_rounds):
    results.data.append(
        collective_learning_round(all_learner_models,
                                  vote_threshold, round_index)
    )
```

## Testing remotely
We expect that the grpc learner part will often be on e.g. a compute cluster and be separate from the grpc client side.
To test the grpc is a setup like this, use run_n_servers.py on the compute cluster to run just the server part of the demo.
Then set up whatever port forwarding etc is required.
The servers by default will start on port 9995 and use subsequent ports from there, so if three
servers are required they will run on ports 9995, 9996 and 9997.
Then run grpc_demo on the other side to run the usual demo.
grpc_demo takes as arguments the model name and dataset name that should be run.

There are a few default dataset and models that we have defined that you can use as well. 
**put names here**

