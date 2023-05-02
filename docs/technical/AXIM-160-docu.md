
# Axim Issue No 160 - Explore a set of model metrics that are common to most ml models

Table of contents:
1. [Easily includable metrics](#easy_metrics)
2. [More time-consuming metrics](#hard_metrics)
3. [Possible approach](#approach)

## Easily includable metrics <a name="easy_metrics"></a>

List of possible metrics that can be included with ease. Basically all metrics from [keras metrics](https://keras.io/api/metrics/) can be used. Except the ones listed under `Classification metrics based on True/False positives & negatives`.
But in this list, most of the metrics are more relevant for a regression use case and not so much for Scania or Mnist:
* Accuracy 
    * Calculates how often predictions equal labels.
* TopKCategoricalAccuracy
    * Computes how often targets are in the top K predictions.
* MSE 
    * Computes the mean squared error between y_true and y_pred.
* MAE 
    * Computes the mean absolute error between the labels and predictions.
* MAPE 
    * Computes the mean absolute percentage error between y_true and y_pred.
* MSLE 
    * Computes the mean squared logarithmic error between y_true and y_pred.

## More time-consuming metrics <a name="hard_metrics"></a>

This is a list of the more relevant metrics for a classification task such as Scania or Mnist. Therefore the shape of the input data and the loss function needs to be changed.
Here are some of the most common ones:
* Precision
    * Computes the precision of the predictions with respect to the labels. The metric creates two local variables, true_positives and false_positives that are used to compute the precision. This value is ultimately returned as precision, an idempotent operation that simply divides true_positives by the sum of true_positives and false_positives.
* Recall
    * Computes the recall of the predictions with respect to the labels. This metric creates two local variables, true_positives and false_negatives, that are used to compute the recall. This value is ultimately returned as recall, an idempotent operation that simply divides true_positives by the sum of true_positives and false_negatives.
* ROC AUC 
    * Approximates the AUC (Area under the curve) of the ROC curve. The AUC (Area under the curve) of the ROC (Receiver operating characteristic; default) or PR (Precision Recall) curves are quality measures of binary classifiers. Unlike the accuracy, and like cross-entropy losses, ROC-AUC evaluate all the operational points of a model.

The F1 score is currently only available in the nightly build of Tensorflow. So we would need to add it manually to the model. The F1 score definition is the following:
```
The F1 score is defined as the harmonic mean of precision and recall. As a short reminder, the harmonic mean is an alternative metric for the more common arithmetic mean. It is often useful when computing an average rate.
```

Similar to the F1 score any other metrics which is not part of the Keras library can be added as a function manually and then added to the metrics list.

## Possible approach <a name="approach"></a>

Assuming we would want to add metrics from the easier category one would need to:

* Add the metrics to the list where the model is compiled when registering a new model. E.g. in the keras_scania.py file.
* If necessary adapt the input shape and/or the loss function to match the metrics.
* Instead of using one metric as a criterion, we would need to change it to the loss function (tested it on one learner to use loss function and on one use accuracy and they both seemed to find their optimum)
* Therefore we need to rewrite so we minimize the loss function and not maximising the metrics how it is atm.
* This can simply be done by using the default variables of the keras_learner.py
* Rewrite the test function in e.g. keras_learner.py Line 277 to also return all the metrics and not only the criterion (loss)
* Add a new variable in the ProposedWeights which includes all metrics (ml_interface.py) as a dict
* In the grpc_learner_server.py under `TestWeights` included the metrics dict
* Adapt the orchestrator and frontend to forward and display the metric dicts in the frontend

The second approach would be to use the first metric as the deciding score for voting and add the others.
So changing the vote_score variable from and float to a dict or a list.


Remarks:
* Tested it briefly locally on Mnist. Although the loss function seemed to work, it is not clear why in the end it did not approve the other participants suggestions. I belive that the data set used for the  vote score and the graph's test score are different. If that's true it can be confusing for the user and needs to be discussed. 
* Got test_score and vote_score. Which are two different scores on two different data sets.