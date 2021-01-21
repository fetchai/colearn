# colearn_pytorch package

## Submodules

## colearn_pytorch.pytorch_covid_xray module


### class colearn_pytorch.pytorch_covid_xray.ModelType(value)
Bases: `enum.Enum`

An enumeration.


#### MULTILAYER_PERCEPTRON( = 1)

### class colearn_pytorch.pytorch_covid_xray.TorchCovidXrayPerceptronModel()
Bases: `torch.nn.modules.module.Module`


#### forward(x)
Defines the computation performed at every call.

Should be overridden by all subclasses.

**NOTE**: Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.


#### training(: bool)

### colearn_pytorch.pytorch_covid_xray.prepare_data_loaders(train_folder: str, train_ratio: float = 0.8, batch_size: int = 8, no_cuda: bool = False, \*\*_kwargs)
Load training data from folders and create train and test dataloader


* **Parameters**

    
    * **train_folder** – Path to training dataset


    * **train_ratio** – What portion of train_data should be used as test set


    * **batch_size** – 


    * **no_cuda** – Disable GPU computing


    * **kwargs** – 



* **Returns**

    Tuple of train_loader and test_loader



### colearn_pytorch.pytorch_covid_xray.prepare_learner(model_type: colearn_pytorch.pytorch_covid_xray.ModelType, data_loaders: Tuple[torch.utils.data.dataloader.DataLoader, torch.utils.data.dataloader.DataLoader], learning_rate: float = 0.001, steps_per_epoch: int = 40, vote_batches: int = 10, no_cuda: bool = False, vote_on_accuracy: bool = True, \*\*_kwargs)

### colearn_pytorch.pytorch_covid_xray.prepare_model(model_type: colearn_pytorch.pytorch_covid_xray.ModelType)

### colearn_pytorch.pytorch_covid_xray.split_to_folders(data_dir: str, n_learners: int, data_split: Optional[List[float]] = None, shuffle_seed: Optional[int] = None, output_folder: Optional[pathlib.Path] = None, \*\*_kwargs)
## colearn_pytorch.pytorch_learner module


### class colearn_pytorch.pytorch_learner.PytorchLearner(model: torch.nn.modules.module.Module, optimizer: torch.optim.optimizer.Optimizer, train_loader: torch.utils.data.dataloader.DataLoader, test_loader: Optional[torch.utils.data.dataloader.DataLoader] = None, device=device(type='cpu'), criterion: Optional[torch.nn.modules.loss._Loss] = None, minimise_criterion=True, vote_criterion: Optional[Callable[[torch.Tensor, torch.Tensor], float]] = None, num_train_batches: Optional[int] = None, num_test_batches: Optional[int] = None, score_name: str = 'score')
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

#### test(loader: torch.utils.data.dataloader.DataLoader)

#### train()

#### vote(new_score)
## colearn_pytorch.pytorch_xray module


### class colearn_pytorch.pytorch_xray.ModelType(value)
Bases: `enum.Enum`

An enumeration.


#### CONV2D( = 1)

### class colearn_pytorch.pytorch_xray.TorchXrayConv2DModel()
Bases: `torch.nn.modules.module.Module`

Total params: 19,265
Trainable params: 19,073
Non-trainable params: 192
_________________________________________________________________


#### forward(x)
Defines the computation performed at every call.

Should be overridden by all subclasses.

**NOTE**: Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.


#### training(: bool)

### class colearn_pytorch.pytorch_xray.XrayDataset(data_dir, transform=None, train=True, train_ratio=0.96, seed=None, width=128, height=128, \*\*_kwargs)
Bases: `Generic`[`torch.utils.data.dataset.T_co`]

X-ray dataset.


#### static to_rgb_normalize_and_resize(filename, width, height)

### colearn_pytorch.pytorch_xray.prepare_data_loaders(train_folder: str, test_folder: Optional[str] = None, train_ratio: float = 0.96, batch_size: int = 8, no_cuda: bool = False, \*\*_kwargs)
Load training data from folders and create train and test dataloader


* **Parameters**

    
    * **train_folder** – Path to training dataset


    * **test_folder** – Path to test dataset


    * **train_ratio** – When test_folder is not specified what portion of train_data should be used as test set


    * **batch_size** – 


    * **no_cuda** – Disable GPU computing


    * **kwargs** – 



* **Returns**

    Tuple of train_loader and test_loader



### colearn_pytorch.pytorch_xray.prepare_learner(model_type: colearn_pytorch.pytorch_xray.ModelType, data_loaders: Tuple[torch.utils.data.dataloader.DataLoader, torch.utils.data.dataloader.DataLoader], learning_rate: float = 0.001, steps_per_epoch: int = 40, vote_batches: int = 10, no_cuda: bool = False, vote_using_auc: bool = True, \*\*_kwargs)

### colearn_pytorch.pytorch_xray.prepare_model(model_type: colearn_pytorch.pytorch_xray.ModelType)

### colearn_pytorch.pytorch_xray.split_to_folders(data_dir: str, n_learners: int, data_split: Optional[List[float]] = None, shuffle_seed: Optional[int] = None, output_folder: Optional[pathlib.Path] = None, train: bool = True, \*\*_kwargs)
## colearn_pytorch.test_pytorch_learner module


### colearn_pytorch.test_pytorch_learner.get_mock_criterion()

### colearn_pytorch.test_pytorch_learner.get_mock_dataloader()

### colearn_pytorch.test_pytorch_learner.get_mock_model()

### colearn_pytorch.test_pytorch_learner.get_mock_optimiser()

### colearn_pytorch.test_pytorch_learner.nkl()
Returns a Pytorchlearner


### colearn_pytorch.test_pytorch_learner.test_accept_weights(nkl)

### colearn_pytorch.test_pytorch_learner.test_get_current_weights(nkl)

### colearn_pytorch.test_pytorch_learner.test_propose_weights(nkl)

### colearn_pytorch.test_pytorch_learner.test_setup(nkl)

### colearn_pytorch.test_pytorch_learner.test_vote(nkl)

### colearn_pytorch.test_pytorch_learner.test_vote_minimise_criterion(nkl)
## colearn_pytorch.utils module


### colearn_pytorch.utils.auc_from_logits(outputs: torch.Tensor, labels: torch.Tensor)
Function to compute area under curve based on model outputs (in logits) and ground truth labels


* **Parameters**

    
    * **outputs** – Tensor of model outputs in logits


    * **labels** – Tensor of ground truth labels



* **Returns**

    AUC score



### colearn_pytorch.utils.binary_accuracy_from_logits(outputs: torch.Tensor, labels: torch.Tensor)
Function to compute binary classification accuracy based on model output (in logits) and ground truth labels


* **Parameters**

    
    * **outputs** – Tensor of model output in logits


    * **labels** – Tensor of ground truth labels



* **Returns**

    Fraction of correct predictions



### colearn_pytorch.utils.categorical_accuracy(outputs: torch.Tensor, labels: torch.Tensor)
Function to compute accuracy based on model prediction and ground truth labels


* **Parameters**

    
    * **outputs** – Tensor of model predictions


    * **labels** – Tensor of ground truth labels



* **Returns**

    Fraction of correct predictions



### colearn_pytorch.utils.prepare_data_split_list(data, n)
Create list of sizes for splitting


* **Parameters**

    
    * **data** – dataset


    * **n** – number of equal parts



* **Returns**

    list of sizes


## Module contents
