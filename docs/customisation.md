# Writing your own models
This tutorial covers how to write your own models to try out collective learning.

Here's an example of a model:
```python
class MNISTConvLearner(KerasLearner):
    def _get_model(self):
        input_img = tf.keras.Input(
            shape=(self.config.width, self.config.height, 1), name="Input"
        )
        x = tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu", padding="same", name="Conv1_1"
        )(input_img)
        x = tf.keras.layers.BatchNormalization(name="bn1")(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), name="pool1")(x)
        x = tf.keras.layers.Conv2D(
            128, (3, 3), activation="relu", padding="same", name="Conv2_1"
        )(x)
        x = tf.keras.layers.BatchNormalization(name="bn4")(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), name="pool2")(x)
        x = tf.keras.layers.Flatten(name="flatten")(x)
        x = tf.keras.layers.Dense(
            self.config.n_classes, activation="softmax", name="fc1"
        )(x)
        model = tf.keras.Model(inputs=input_img, outputs=x)

        # compile model & add optimiser
        opt = self.config.optimizer(
            lr=self.config.l_rate, decay=self.config.l_rate_decay
        )

        model.compile(loss=self.config.loss, metrics=["accuracy"], optimizer=opt)
        return model
```
As can be seen the model inherits from the
[KerasLearner]({{ repo_root }}/colearn_examples/keras_learner.py) 
which inherits from the BasicLearner which implements the interface. 
Each of these intermediate classes are customization points the library provides to simplify 
the deployment of your own models. For any Keras based model we recommend starting with the KerasLearner. 
In the Example folder there is also a [SKLearnLearner]({{ repo_root }}/colearn_examples/sklearn_learner.py) for Scikit-learn models .

The BasicLearner handles some of the logic required by the interface and hands what is model specific to the subclass. For example BasicLearner implements test_model

```python

def test_model(self, weights=None) -> ProposedWeights:
    """Tests the proposed weights and fills in the rest of the fields"""
    if weights is None:
        weights = self.get_weights()

    proposed_weights = ProposedWeights()
    proposed_weights.weights = weights
    proposed_weights.validation_accuracy = self._test_model(weights,
                                                            validate=True)
    proposed_weights.test_score = self._test_model(weights,
                                                   validate=False)
    proposed_weights.vote = (
            proposed_weights.validation_accuracy >= self.vote_score
    )

    return proposed_weights
```
but get_weights is left unimplemented
```python

def get_weights(self):
    raise NotImplementedError
```
so it is implemented by the KerasLearner
```python
    def get_weights(self):
        return KerasWeights(self._model.get_weights())
```
Bu using and extending the learners provided it is simple to setup your own models.
