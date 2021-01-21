# colearn_keras package

## Submodules

## colearn_keras.keras_cifar10 module


### class colearn_keras.keras_cifar10.ModelType(value)
Bases: `enum.Enum`

An enumeration.


#### CONV2D( = 1)

### colearn_keras.keras_cifar10.prepare_data_loaders(train_folder: str, train_ratio: float = 0.9, batch_size: int = 32, \*\*_kwargs)
Load training data from folders and create train and test dataloader


* **Parameters**

    
    * **train_folder** – Path to training dataset


    * **train_ratio** – What portion of train_data should be used as test set


    * **batch_size** – 


    * **kwargs** – 



* **Returns**

    Tuple of train_loader and test_loader



### colearn_keras.keras_cifar10.prepare_learner(model_type: colearn_keras.keras_cifar10.ModelType, data_loaders: Tuple[tensorflow.python.data.ops.dataset_ops.PrefetchDataset, tensorflow.python.data.ops.dataset_ops.PrefetchDataset], steps_per_epoch: int = 100, vote_batches: int = 10, learning_rate: float = 0.001, \*\*_kwargs)

### colearn_keras.keras_cifar10.split_to_folders(n_learners: int, data_split: Optional[List[float]] = None, shuffle_seed: Optional[int] = None, output_folder: Optional[pathlib.Path] = None, \*\*_kwargs)
## colearn_keras.keras_learner module


### class colearn_keras.keras_learner.KerasLearner(model: tensorflow.python.keras.engine.training.Model, train_loader: tensorflow.python.data.ops.dataset_ops.DatasetV2, test_loader: Optional[tensorflow.python.data.ops.dataset_ops.DatasetV2] = None, minimise_criterion: bool = True, criterion: str = 'loss', model_fit_kwargs: Optional[dict] = None, model_evaluate_kwargs: Optional[dict] = None)
Bases: `colearn.ml_interface.MachineLearningInterface`


#### mli_accept_weights(weights: colearn.ml_interface.Weights)
Updates the model with the proposed set of weights
:param weights: The new weights


#### mli_get_current_weights()
Returns the current weights of the model


#### mli_propose_weights()
Trains the model. Returns new weights. Does not change the current weights of the model.


#### mli_test_weights(weights: colearn.ml_interface.Weights)
Tests the proposed weights and fills in the rest of the fields


#### set_weights(weights: colearn.ml_interface.Weights)

#### test(loader: tensorflow.python.data.ops.dataset_ops.DatasetV2)

#### train()

#### vote(new_score)
## colearn_keras.keras_mnist module


### class colearn_keras.keras_mnist.ModelType(value)
Bases: `enum.Enum`

An enumeration.


#### CONV2D( = 1)

### colearn_keras.keras_mnist.prepare_data_loaders(train_folder: str, train_ratio: float = 0.9, batch_size: int = 32, \*\*_kwargs)
Load training data from folders and create train and test dataloader


* **Parameters**

    
    * **train_folder** – Path to training dataset


    * **train_ratio** – What portion of train_data should be used as test set


    * **batch_size** – 


    * **kwargs** – 



* **Returns**

    Tuple of train_loader and test_loader



### colearn_keras.keras_mnist.prepare_learner(model_type: colearn_keras.keras_mnist.ModelType, data_loaders: Tuple[tensorflow.python.data.ops.dataset_ops.PrefetchDataset, tensorflow.python.data.ops.dataset_ops.PrefetchDataset], steps_per_epoch: int = 100, vote_batches: int = 10, learning_rate: float = 0.001, \*\*_kwargs)

### colearn_keras.keras_mnist.split_to_folders(n_learners: int, data_split: Optional[List[float]] = None, shuffle_seed: Optional[int] = None, output_folder: Optional[pathlib.Path] = None, \*\*_kwargs)
## colearn_keras.test_keras_learner module


### colearn_keras.test_keras_learner.get_mock_dataloader()

### colearn_keras.test_keras_learner.get_mock_model()

### colearn_keras.test_keras_learner.nkl()
Returns a Keraslearner


### colearn_keras.test_keras_learner.test_criterion(nkl)

### colearn_keras.test_keras_learner.test_get_current_weights(nkl)

### colearn_keras.test_keras_learner.test_minimise_criterion(nkl)

### colearn_keras.test_keras_learner.test_propose_weights(nkl)

### colearn_keras.test_keras_learner.test_vote(nkl)
## Module contents
