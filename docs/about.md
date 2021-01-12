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



## Implementation details - the ML interface
The ML interface is a set of methods that users need to define to interact with the driver. This is the bit that's customisable! 
Users that want to write new models that the driver can use just need to inherit from the foloowing class and implement the necessary functions.

