# How collective learning works

## The voting protocol
The core idea of collective learning is the voting protocol. One learner is selected to train and produce an updated version of the model parameters,
and then the others vote on the proposed update. The driver implements the voting protocol - see the code below.

## The driver
The driver implements the voting protocol, so it handles selecting a learner to train, 
sending the update out for voting, calculating the vote and accepting or declining the update. 
Here we have a very minimal driver that doesn't use networking or a blockchain. Eventually the driver will be a smart contract. 
This is the code that implements one round of voting:

```python
def run_one_epoch(epoch_index: int, learners: Sequence[MachineLearningInterface],
                  vote_threshold=0.5):
    proposer = epoch_index % len(learners)
    new_weights = learners[proposer].propose_weights()

    prop_weights_list = [ln.test_weights(new_weights) for ln in learners]
    approves = sum(1 if v.vote else 0 for v in prop_weights_list)

    vote = False
    if approves >= len(learners) * vote_threshold:
        vote = True
        for j, learner in enumerate(learners):
            learner.accept_weights(prop_weights_list[j])

    return prop_weights_list, vote
```
The driver has a list of learners, and each round it selects one learner to be the proposer.
The proposer does some training and proposes an updated set of weights.
The driver then sends the proposed weights to each of the learners and they each vote on whether this is an improvement.
If the number of approving votes is greater than the vote threshold the proposed weights are accepted, ad if not they're rejected.



## Implementation details - the ML interface
The ML interface is a set of methods that users need to define to interact with the driver. This is the bit that's customisable! 
Users that want to write new models that the driver can use just need to inherit from the foloowing class and implement the necessary functions.

