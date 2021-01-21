# Installation
The core package, `colearn`, contains only the [MachineLearningInterface](about.md) and a simple driver that 
implements the Collective Learning Protocol. 
To install only the core package:
```
pip install .
```

To make collective learning easier to use we have defined extra packages with helpers
for model development in Keras and Pytorch.

To install with Keras/Pytorch extras:
```
pip install .[keras]
pip install .[pytorch]
```

To install all the extras, including the ones required for the examples, use:
```
pip install .[all]
```

If you are developing the colearn library then install it in editable mode so that new
changes are effective immediately:
```
pip install -e .[all]
```
