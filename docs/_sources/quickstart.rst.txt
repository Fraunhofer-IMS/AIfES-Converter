Quickstart
===========

Installation
------------

This tool can be installed with pip by using:

.. code-block::

    python -m pip install AIfES-Converter


It has the following dependencies:

* Numpy >= 1.19
* packaging


Additionally, depending on your used python framework one or both of:

* Tensorflow >= 2.4
* PyTorch >= 1.8

.. warning::
    Those are not given as default dependencies, as only one of them is needed. Either make sure, that you have your desired framework already installed or
    use on of the following commands for installation of the framework with AIfES-Converter together.

    .. dropdown:: Installation of Framework with AIfES-Converter
        :open:
        :animate: fade-in-slide-down

        If you want to install the framework with AIfES-Converter together you can use one of the following commands:

        .. code-block::

            # For Tensorflow
            python -m pip install AIfES-Converter[tensorflow]

            # For PyTorch
            python -m pip install AIfES-Converter[pytorch]

            # For Tensorflow and PyTorch
            python -m pip install AIfES-Converter[both]


Quick Example
---------------

.. code-block::

    from tensorflow import keras
    import torch
    from aifes import keras2aifes, pytorch2aifes

    #------------------------------------------------------------------------
    # Create your model with Keras or PyTorch
    #------------------------------------------------------------------------
    # You can create or load your model here. See the Usage section for more details.
    # Here we are loading a presaved Keras Model
    keras_model = keras.models.load_model('path/to/location/keras')

    # For PyTorch you can do the same with
    pytorch_model = torch.load('path/to/location/pytorch')

    #------------------------------------------------------------------------
    # Convert the model to an executable AIfES model
    #------------------------------------------------------------------------
    # For a detailed description of the available functions see the section 'Available Functions'
    # Converts the Keras/PyTorch model to an express AIfES model and saves
    # the header files in path/to/location/{framework}/output
    keras2aifes.convert_to_fnn_f32_express(keras_model, 'path/to/location/keras/output')
    pytorch2aifes.convert_to_fnn_f32_express(pytorch_model, 'path/to/location/pytorch/output')
