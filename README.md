# Welcome to the Fetch.ai Collective Learning

Colearn is a library that enables privacy-preserving decentralized machine learning tasks on the FET network.

This blockchain-mediated collective learning system enables multiple stakeholders to build a shared 
machine learning model without needing to rely on a central authority. 
This library is currently in development. 


## Installation
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
   
For more instructions, see the documentation at [fetchai.github.io/colearn/](https://fetchai.github.io/colearn/)


## Build image

To build ML server image and push to google cloud use the following command:
```
cd docker
python3 ./build.py --publish --allow_dirty
# Check this worked correctly
docker images
```


## Documentation
To run the documentation, first install mkdocs and plugins:
```bash
pip install mkdocs==1.1.2 mkdocs-macros-plugin==0.5.0 \
mkdocs-macros-test==0.1.0 mkdocs-material==6.2.3 \
mkdocs-material-extensions==1.0.1 markdown-include==0.6.0
```

Then run: 
```
mkdocs serve
```


### Current Version

We have released *v.0.1* of the Colearn Machine Learning Interface, the first version of an interface that will allow developers to prepare for future releases. 
Together with the interface we provide a simple backend for local experiments. This is the first backend with upcoming blockchain ledger based backends to follow.  
Future releases will use similar interfaces so that learners built with the current system will work on a different backend that integrates a distributed ledger and provides other improvements.
The current framework will then be used mainly for model development and debugging.
We invite all users to experiment with the framework, develop their own models, and provide feedback!



## Quick Overview
The collective learning protocol allows learners to collaborate on training a model without requiring trust between the participants. Learners vote on updates to the model, and only updates which pass the quality threshold are accepted. This makes the system robust to attempts to interfere with the model by providing bad updates. For more details on the collective learning system see [here](docs/about.md)
