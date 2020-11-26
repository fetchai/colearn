# What is differential privacy?
Differential privacy is a system for publicly sharing information about a dataset by describing the patterns
 of groups within the dataset while withholding information about individuals in the dataset. 
 The idea behind differential privacy is that if the effect of making an arbitrary single substitution in 
 the database is small enough, the query result cannot be used to infer much about any single individual,
  and therefore provides privacy.

# How to use differential privacy with colearn
Differential privacy is available for the PyTorchLearner and KerasLearner models. 
To enable DP, ensure that in the model config you have selected a model that is derived from PyTorch or KerasLearner, 
and that `use_dp = True`.
An example ModelConfig for MNIST is shown below:

```python3
class MNISTConfig(ModelConfig):
    def __init__(self, seed=None):
        super().__init__(seed)

        # Training params
        self.optimizer = tf.keras.optimizers.Adam

        # Model params
        self.model_type = MNISTSuperminiLearner
        self.loss = "sparse_categorical_crossentropy"

        ...

        # DP params
        self.use_dp = True
        self.sample_size = 3300
        self.alphas = list(range(2, 32))
        self.noise_multiplier = 1.2
        self.max_grad_norm = 1.0
        self.l2_norm_clip = 1.0
        self.microbatches = self.batch_size
```
