# MLI Factory

The machine learning interface factory are the minimum methods a client needs to implement
to work with the GRPC Server (and become a Learner).

There are two main types of functions:

- Supported Systems (get_models, get_dataloaders, get_compatibilities)
- Get a MachineLearningInterface (get_mli)

When the GRPC server is connected to the Orchestrator, it will query the supported system
functions to know what the MLI Factory can serve.

Later when the Orchestrator wants to run something on this Learner it will call get_mli
with a model_arch_name, a dataloader_name and more parameters for both.
The object returned is then used to run the experiment through the MLI.

### Supported Systems

The supported systems functions get_models and get_dataloaders should return a set of
<name, {default parameters dictionary}> which will be stored (not currently implemented)
in the api database. The idea being that the user can change these values on the
UI while preparing to start/join an experiment.

### ExampleMliFactory

An example MLIFactory that will implement all the tasks in run_demo.
This is the one used by contract_learn.
