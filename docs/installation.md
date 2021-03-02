# Installation
The core package, `colearn`, contains only the [MachineLearningInterface](about.md) and a simple driver that 
implements the Collective Learning Protocol. 

### To install colearn using Pip:

To install only the core package:
```
pip install colearn
```

To make collective learning easier to use we have defined extra packages with helpers
for model development in Keras and Pytorch.

To install with Keras/Pytorch extras:
```
pip install colearn[keras]
pip install colearn[pytorch]
```

To install both the Keras and Pytorch extras use:
```
pip install colearn[all]
```

### To install colearn using Anaconda

1. Download the source code from github:
   ```bash
   git clone https://github.com/fetchai/colearn.git && cd colearn
   ```

2. Create conda environment:
   ```bash
   conda create --name myenv python=3.7.5 conda-build
   conda activate myenv
   ```

3. Install the package from source:
   ```bash
   conda-build colearn
   conda install --use-local colearn 
   ```


### To run stand-alone examples:
   ```bash
   # Download the examples
   wget https://raw.githubusercontent.com/fetchai/colearn/master/examples/keras_cifar.py
   wget https://raw.githubusercontent.com/fetchai/colearn/master/examples/keras_fraud.py
   wget https://raw.githubusercontent.com/fetchai/colearn/master/examples/keras_mnist.py
   wget https://raw.githubusercontent.com/fetchai/colearn/master/examples/keras_mnist_diffpriv.py
   wget https://raw.githubusercontent.com/fetchai/colearn/master/examples/keras_xray.py
   wget https://raw.githubusercontent.com/fetchai/colearn/master/examples/mli_fraud.py
   wget https://raw.githubusercontent.com/fetchai/colearn/master/examples/mli_random_forest_iris.py
   wget https://raw.githubusercontent.com/fetchai/colearn/master/examples/pytorch_cifar.py
   wget https://raw.githubusercontent.com/fetchai/colearn/master/examples/pytorch_covid.py
   wget https://raw.githubusercontent.com/fetchai/colearn/master/examples/pytorch_mnist.py
   wget https://raw.githubusercontent.com/fetchai/colearn/master/examples/pytorch_mnist_diffpriv.py
   wget https://raw.githubusercontent.com/fetchai/colearn/master/examples/pytorch_xray.py
   # Or they can be accessed from colearn/examples by cloning colearn repo

   # for colearn[keras] or colearn[all]
   python3 keras_mnist.py
   # for colearn[pytorch] or colearn[all]
   python3 pytorch_mnist.py
   ``` 

## Installing From Source

Alternatively, to install the latest code from the repo:

1. Download the source code from github:
   ```bash
   git clone https://github.com/fetchai/colearn.git && cd colearn
   ```
1. Create and launch a clean virtual environment with Python 3.7. 
   (This library has currently only been tested with Python 3.7).
   ```bash
   pipenv --python 3.7 && pipenv shell
   ```

2. Install the package from source:
    ```bash
    pip install -e .[all]
    ```
3. Run one of the examples:
    ```bash
    examples/pytorch_mnist.py
    ``` 
   
If you are developing the colearn library then install it in editable mode so that new
changes are effective immediately:
```
pip install -e .[all]
```

## Running the tests
Tests can be run with:
```
tox
```
## Documentation
To run the documentation, first install [mkdocs](https://www.mkdocs.org) and plugins:
```bash
pip install .[docs] 
```

Then run: 
```
mkdocs serve
```

