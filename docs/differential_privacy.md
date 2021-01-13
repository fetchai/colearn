# What is differential privacy?
Differential privacy (DP) is a system for publicly sharing information about a dataset by describing the patterns
 of groups within the dataset while withholding information about individuals in the dataset. 
 The idea behind differential privacy is that if the effect of making an arbitrary single substitution in 
 the database is small enough, the query result cannot be used to infer much about any single individual,
  and therefore provides privacy.

# How to use differential privacy with colearn
The opacus and tensorflow-privacy libraries implement DP for pytorch and keras respectively.
To see an example of using them see [dp_pytorch]({{ repo_root }}/colearn_examples_pytorch/mnist_pytorch_dp.py) 
and [dp_keras]({{ repo_root }}/colearn_examples_keras/mnist_keras_dp.py).