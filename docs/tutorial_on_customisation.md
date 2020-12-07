# Tutorial

The most flexible way to use the collective learning backends is to implement something that implements
the collective learning interface. The methods that need to be implemented are propose_weights, evaluate_weights and accept_weights. 

But the simpler way is to use one of the classes that we have provided that implement 
standard bits of collective learning for common ML libraries. These learners are SKLearnLearner, 
KerasLearner, and PytorchLearner. These learners implement methods to propose, evaluate and accept weights, 
and the user just needs to implement the _get_model function for the derived class.  

In this tutorial we are going to walk through using the PyTorchLearner.  

