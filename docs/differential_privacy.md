# What is differential privacy?
To make a machine learning system that protects privacy we first need to have a definition of what privacy is. 
Differential privacy (DP) is one such definition. 
First we need to have three concepts: the _database_ is a collection of data about _individuals_ (for example, their medical records), and we want to make a _query_ about that data (for example "How much does smoking increase someone's risk of cancer?").
DP says that privacy is preserved if the result of the query cannot be used to determine if any particular individual is present in the database.

So if person A has their medical data in a database, and the query that we want to make on that database is 
"How much does smoking increase someone's risk of cancer" then the result of that query shouldn't disclose whether or not person A's details are in the database.

From this comes the idea of _sensitivity_ of a query. 
The _sensitivity_ of a query determines how much the result of the query depends on an individual's data. 
For example, the query "How much does smoking increase the risk of cancer for adults in the UK?" is less sensitive than the query "How much does smoking increase the risk of cancer for men aged 50-55 in Cambridge?" because the second query uses a smaller set of individuals.

## Epsilon-differential privacy
EDP is a scheme for preserving differential privacy. 
In EDP all queries have random noise added to them, so they are no longer deterministic.
So if the query was "What fraction of people in the database are male", and the true result is 0.5 then the results of calling this query three times might be 0.53, 0.49 and 0.51. 
This makes it harder to tell if an individual's data is in the database, because the effect of adding a person can't be distinguished from the effect of the random noise.
Intuitively this is a bit like blurring an image: adding noise obscures personal information.
The amount of personal information that is revealed isn't zero, but it is guaranteed to be below a certain threshold.

The level of privacy that is provided is controlled by the parameter epsilon; the greater epsilon is the more noise is added and the more privacy is preserved.
Queries that are more sensitive have more noise added, because they reveal more information about individuals.
It is important to add as little noise as possible, because adding more noise obscures the patterns that you want to extract from the data.

## Differential privacy when training neural netowrks
Each training step for a neural network can be though of as a complicated query on a database of training data.
Differential privacy mechanisms tell you how much noise you need to add to guarantee a certain level of privacy.
The `opacus` and `tensorflow-privacy` libraries implement epsilon-differential privacy for training neural networks for pytorch and keras respectively.


# How to use differential privacy with colearn
By using `opacus` and `tensorflow-privacy` we can make colleactive learning use differential privacy.
The learner that is proposing weights does so using a DP-enabled optimiser.

To see an example of using this see [dp_pytorch]({{ repo_root }}/examples/pytorch_mnist_diffpriv.py) 
and [dp_keras]({{ repo_root }}/examples/keras_mnist_diffpriv.py).