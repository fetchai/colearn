# colearn_other package

## Submodules

## colearn_other.fraud_dataset module


### class colearn_other.fraud_dataset.FraudLearner(train_data: numpy.array, train_labels: numpy.array, test_data: numpy.array, test_labels: numpy.array, batch_size: int = 10000, steps_per_round: int = 1)
Bases: `colearn.ml_interface.MachineLearningInterface`


#### mli_accept_weights(weights: colearn.ml_interface.Weights)
Updates the model with the proposed set of weights
:param weights: The new weights


#### mli_get_current_weights()
Returns the current weights of the model


#### mli_propose_weights()
Trains the model. Returns new weights. Does not change the current weights of the model.


#### mli_test_weights(weights: colearn.ml_interface.Weights, eval_config: Optional[dict] = None)
Tests the proposed weights and fills in the rest of the fields


#### set_weights(weights: colearn.ml_interface.Weights)

#### test(data, labels)

### class colearn_other.fraud_dataset.ModelType(value)
Bases: `enum.Enum`

An enumeration.


#### SVM( = 1)

### colearn_other.fraud_dataset.prepare_data_loaders(train_folder: str, train_ratio: float = 0.8, \*\*_kwargs)
Load training data from folders and create train and test arrays


* **Parameters**

    
    * **train_folder** – Path to training dataset


    * **train_ratio** – What portion of train_data should be used as test set


    * **kwargs** – 



* **Returns**

    Tuple of tuples (train_data, train_labels), (test_data, test_loaders)



### colearn_other.fraud_dataset.prepare_learner(model_type: colearn_other.fraud_dataset.ModelType, data_loaders: Tuple[Tuple[numpy.array, numpy.array], Tuple[numpy.array, numpy.array]], \*\*_kwargs)

### colearn_other.fraud_dataset.split_to_folders(data_dir: str, n_learners: int, data_split: Optional[List[float]] = None, shuffle_seed: Optional[int] = None, output_folder: Optional[pathlib.Path] = None, \*\*_kwargs)
## colearn_other.mli_factory module


### class colearn_other.mli_factory.TaskType(value)
Bases: `enum.Enum`

An enumeration.


#### FRAUD( = 5)

#### KERAS_CIFAR10( = 3)

#### KERAS_MNIST( = 2)

#### PYTORCH_COVID_XRAY( = 4)

#### PYTORCH_XRAY( = 1)

### colearn_other.mli_factory.mli_factory(str_task_type: str, train_folder: str, str_model_type: str, test_folder: Optional[str] = None, \*\*learning_kwargs)
## colearn_other.new_demo module


### colearn_other.new_demo.main(str_task_type: str, n_learners: int = 5, n_epochs: int = 20, vote_threshold: float = 0.5, train_data_folder: Optional[str] = None, test_data_folder: Optional[str] = None, str_model_type: Optional[str] = None, \*\*learning_kwargs)
## Module contents
