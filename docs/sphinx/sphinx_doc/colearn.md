# colearn package

## Subpackages


* colearn.utils package


    * Submodules


    * colearn.utils.data module


    * colearn.utils.plot module


    * colearn.utils.results module


    * Module contents


## Submodules

## colearn.ml_interface module


### class colearn.ml_interface.MachineLearningInterface()
Bases: `abc.ABC`


#### abstract mli_accept_weights(weights: colearn.ml_interface.Weights)
Updates the model with the proposed set of weights
:param weights: The new weights


#### abstract mli_get_current_weights()
Returns the current weights of the model


#### abstract mli_propose_weights()
Trains the model. Returns new weights. Does not change the current weights of the model.


#### abstract mli_test_weights(weights: colearn.ml_interface.Weights)
Tests the proposed weights and fills in the rest of the fields


### class colearn.ml_interface.ProposedWeights(\*, weights: colearn.ml_interface.Weights, vote_score: float, test_score: float, vote: bool, evaluation_results: Dict = None)
Bases: `pydantic.main.BaseModel`


#### evaluation_results(: Optional[Dict])

#### test_score(: float)

#### vote(: bool)

#### vote_score(: float)

#### weights(: colearn.ml_interface.Weights)

### class colearn.ml_interface.Weights(\*, weights: Any = None)
Bases: `pydantic.main.BaseModel`


#### weights(: Any)
## colearn.standalone_driver module


### colearn.standalone_driver.run(n_epochs: int, learners: List[colearn.ml_interface.MachineLearningInterface])

### colearn.standalone_driver.run_one_epoch(epoch_index: int, learners: Sequence[colearn.ml_interface.MachineLearningInterface], vote_threshold=0.5)
## colearn.training module


### colearn.training.collective_learning_round(learners: Sequence[colearn.ml_interface.MachineLearningInterface], vote_threshold, epoch)

### colearn.training.individual_training_round(learners: Sequence[colearn.ml_interface.MachineLearningInterface], epoch)

### colearn.training.initial_result(learners: Sequence[colearn.ml_interface.MachineLearningInterface])

### colearn.training.set_equal_weights(learners: Sequence[colearn.ml_interface.MachineLearningInterface])
## Module contents
