# How collective learning works
A Colearn experiment begins when a group of entities, a group of *learners*, decide on a model architecture and begin learning. Together they will train a single global model. The goal is to train a model that performs better than any of the learners can produce by training on their private data set. 

### How Training Works

Training occurs in rounds; during each round the learners attempt to improve the performance of the global shared model. 
To do so each round an **update** of the global model (for example new set of weights in a neural network) is proposed. 
The learners then **validate** the update and decide if the new model is better than the current global model.  
If enough learners *approve* the update then global model is updated. After an update is approved or rejected a new round begins. 

The detailed steps of a round updating a global model *M* are as follows:

1. One of the learners is selected and proposes a new updated model *M'*
2. The rest of the learners **validate** *M'*
   - If *M'* has better performance than *M* then the learner votes to approve
   - If not the learner votes to reject
3. The total votes are tallied
   - If more than some threshold (typically 50%) of learners approve then *M'* becomes the new global model. If not, *M* continues to be global model
4. A new round begins. 

By using a decentralized ledger (a blockchain) this learning process can be run in a completely decentralized, secure and auditable way. Further security can be provided by using [differential privacy](https://en.wikipedia.org/wiki/Differential_privacy) when generating an update.


## The driver
The driver implements the voting protocol, so it handles selecting a learner to train, 
sending the update out for voting, calculating the vote and accepting or declining the update. 
Here we have a very minimal driver that doesn't use networking or a blockchain. Eventually the driver will be a smart contract. 
This is the code that implements one round of voting:

```python
def run_one_epoch(epoch_index: int, learners: Sequence[MachineLearningInterface],
                  vote_threshold=0.5):
    proposer = epoch_index % len(learners)
    new_weights = learners[proposer].mli_propose_weights()

    prop_weights_list = [ln.mli_test_weights(new_weights) for ln in learners]
    approves = sum(1 if v.vote else 0 for v in prop_weights_list)

    vote = False
    if approves >= len(learners) * vote_threshold:
        vote = True
        for j, learner in enumerate(learners):
            learner.mli_accept_weights(prop_weights_list[j])

    return prop_weights_list, vote
```
The driver has a list of learners, and each round it selects one learner to be the proposer.
The proposer does some training and proposes an updated set of weights.
The driver then sends the proposed weights to each of the learners and they each vote on whether this is an improvement.
If the number of approving votes is greater than the vote threshold the proposed weights are accepted, ad if not they're rejected.


## The MachineLearningInterface
```Python 
{!../colearn/ml_interface.py!} 
```
There are four methods that need to be implemented:

1. `propose_weights` causes the model to do some training and then return a
   new set of weights that are propsed to the other learners. 
   This method shouldn't change the current weights of the model - that
   only happens when `accept_weights` is called.
2. `test_weights` - the models takes some new weights and returns a vote on whether the new weights are an improvement. 
   As in propse_weights, this shouldn't change the current weights of the model - 
   that only happens when `accept_weights` is called.
3. `accept_weights` - the models accepts some weights that have been voted on and approved by the set of learners. 
    The old weighs of the model are discarded and replaced by the new weights.
4. `current_weights` should return the current weights of the model.

For more details about directly implementing the machine learning interface
see the tutorial [here](./intro_tutorial_mli.md)
