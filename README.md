# AIfES-Converter

This is a convert tool to create [AIfES](https://aifes.ai) models for direct use in the Arduino IDE or other IDEs. 
It can read Feed Forward Neural Networks (FFNN) models from Keras and PyTorch and converts them to AIfES models, which are exported in header files. Those header 
files can than be added to your Project in any IDE and can be used there. 

## Quick Start
Install the converter:
````commandline
pip install AIfES-Converter
````

IMPORTANT: For a detailed description of the installation see the [documentation](#documentation)

Convert a Keras model, e.g.:
````python
from aifes import keras2aifes

keras2aifes.convert_to_fnn_f32_express(model, 'path/to/location')
````

Convert a PyTorch model, e.g.:
````python
from aifes import pytorch2aifes

pytorch2aifes.convert_to_fnn_f32_express(model, 'path/to/location')
````

## Documentation

For a detailed documentation see [here]([https://fraunhofer-ims.github.io/AIfES-Converter/]).
