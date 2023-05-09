
# Axim Issue No 128 - Add Predictive maintenance Demo to Axim

Table of contents:
1. [Installation](#installation)
2. [Data](#data)
3. [Preprocessing](#preprocessing)
4. [Model](#model)

## Installation <a name="installation"></a>

There are currently two ways how to run an experiment with the Scania dataset:
* Via Axim on staging
* Locally with docker containers

### Running an experiment on staging

There is a step-by-step slide show available on [google docs](https://docs.google.com/presentation/d/1HKZ8VjH0SSlF3Mq9N-Hb-fGBjt9ckzZMtqyAtoOiP-w/edit#slide=id.g1fc4d81106e_0_6).

Here you can find the steps as a summary:
* Go to [Axim Staging](https://axim-staging.sandbox-london-b.fetch-ai.com/auth/register) and register two users
* Login on two different browser windows
* Start both nodes and wait until
* On both add and start the new learners with the following image: `gcr.io/fetch-ai-sandbox/colearn:Scania`
* In one browser create a new experiment, where you need to create a new Scania model and dataset (URL for the dataset is: `gs://colearn-public/Scania/0`)
* Invite your second email address and start the experiment
* Switch to the second account and join the existing experiment
* For the dataset use `gs://colearn-public/Scania/1`
* After successfully joining, the experiment should run and you can follow the process

### Running it locally

First, check if you have access to the [colearn](https://github.com/fetchai/colearn) and [axim](https://github.com/fetchai/axim) repo.

Clone both repos and check out the readme in the axim repo, which is under `scripts/experiment`. Get all the images from the list.

Make sure that the settings are configured right in the `.env` file and in the `envs` folder as well.
```
DATASET_PREFIX="gs://colearn-public/Scania"

DOCKER_IMAGES_TAG="validators_demo"
```
If the Colearn image is not up to date, go to the Colearn repo and run this:
```
docker build -t colearn -f ./docker/ml.Dockerfile . && docker tag colearn:latest gcr.io/fetch-ai-colearn-gcr/colearn:validators_demo
```

All docker images are available now. You can now continue with the example in the readme.md. It works the same way for the Scania dataset. You only need to select the right data loader and model, which are both called `KERAS_Scania`.

## How does the data look like <a name="data"></a>

The data used for this use case is called the Scania Trucks data set and is available [here](https://archive.ics.uci.edu/ml/datasets/APS+Failure+at+Scania+Trucks).

Introduction about the data:

>The dataset consists of data collected from heavy Scania
trucks in everyday usage. The system in focus is the
Air Pressure system (APS) which generates pressurised
air that are utilized in various functions in a truck,
such as braking and gear changes. The datasets'
positive class consists of component failures
for a specific component of the APS system.
The negative class consists of trucks with failures
for components not related to the APS. The data consists
of a subset of all available data, selected by experts.

It consists of two csv files, one for training and one for testing. Each row in the file represents one failure that has been documented.

It has been preprocessed, has been split up into eight parts and has been uploaded to the google cloud storage under `gs://colearn-public/Scania/{0-7}`.

Short summary:
* Binary classification problem
* Number of instances: 60k (1k positive class)
* There are in total 171 attributes (features) for each entry
* Those feature names have been anonymized due to proprietary reasons
* It only contains integers, floats and missing values (`na`)

## What steps have been done to the data <a name="preprocessing"></a>

Steps:
* Split train data into train and vote 
* 1k vote, 1k test, 58k train
* Preprocessing 
* Split into eight datasets (shuffled)
* Saving index to X set for being able to identify which rows have been split into which parts

For training set:
* Replace class names ("pos", "neg") with 1 and 0
* Remove constant columns
* Discard features with more than 70% missing values
* Remove rows with missing values for features with less than 5% missing values
* Impute remaining missing values for features with less than 15% missing values (SimpleImputer with median strategy)
* Impute the rest with MICE (multivariate imputation by chained equations) using numpy's IterativeImputer with Ridge estimator
* SMOTE (Synthetic Minority Oversampling Technique for imbalanced data)
```
over = SMOTE(sampling_strategy=0.3)
under = RandomUnderSampler(sampling_strategy=0.5)
```
* Scale the date using Sklearn's MinMaxScaler

For vote and test set:
* Replace class names ("pos", "neg") with 1 and 0
* Use feature selection strategy, median imputer, mice imputer and scaler from training set 

In the end, each of the eight datsets has the following shapes:
* X_test & X_vote: (1000,163)
* X_train: (6229, 163)

## How does the model look like <a name="model"></a>

There are two Keras models available via the Colearn interface, which are defined in the [keras_Scania.py](https://github.com/fetchai/colearn/blob/feat/add-predictive-maintenance-demo/colearn_keras/keras_Scania.py):
* MLP (Multilayer Perceptrons).
* Resnet-50

Both are compiled using the `Adam` optimizer with `SparseCategoricalAccuracy` as a metric.

Of course, the resnet model takes way longer for training and is better suited for an image classification task.

The model receives as input for each entry the 163 anonymized and preprocessed features as a list. And it predicts whether the failure of the truck is due to the APS system (output would be `1` for the positive class) or not (output would be `0` for the negative class).

### MLP model code

```
model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(n_classes, activation='softmax'),
    ])
```
### Resnet model code

```
input_img = tf.keras.Input(
    shape=(rows, cols, channels), name="Input"
)
x = tf.keras.layers.ZeroPadding2D(padding=padding)(input_img)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.RepeatVector(new_channels)(x)
x = tf.keras.layers.Reshape(
    (rows + padding * 2, cols + padding * 2, new_channels))(x)

resnet = ResNet50(include_top=False, input_tensor=x)

x = resnet.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = Dropout(0.7)(x)
x = tf.keras.layers.Dense(n_classes, activation='softmax')(x)

model = tf.keras.Model(inputs=input_img, outputs=x)
```
