# gRPC tutorial

## Architecture of colearn
There's the orchestrator that does communication with smart contracts and IPFS stuff.
This communicates with the learner via gRPC.
The learner (what is the proper name??) runs a gRPC server that exposes methods for proposing, evaluating and testing weights.

## Architecture of gRPC server
grpc server has a mli factory, and it uses the mli factory to make the learner
mli factory needs to implement mli factory interface.
So you could write your own mli factory,
but it's easier to use the one we provide
Below we will discuss the mli factory and then talk about how to use the example factory.

## mli factory interface:
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


## Using the example mlif
The example mlif is defined in colearn_grpc/example_mli_factory.py.
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

You can see an example of how to do this in colearn_grpc/mnist_grpc.py.
   
Constraints on the dataloader function:
1. The first parameter should be a mandatory parameter called "train_folder" which stores the location of the dataset.
2. The subsequent parameters should have default arguments.

Contraints on the model function:
1. The first parameter should be a mandatory parameter called "data_loaders". 
   This must have the same type as the return type of the dataloader.
2. The subsequent parameters should have default arguments.

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

