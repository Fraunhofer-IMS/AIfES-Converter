.. _available-functions:

Available Functions
####################
The following functions are available for a conversion from Keras and PyTorch to AIfES. They are sorted by the used interface.

AIfES Express
--------------

AIfES Express is an interface that allows easy and fast setup of FFNNs. It offers an intuitive interface and allows for changes of the model during runtime.
We also offer the interface AIfES, see the `AIfES`_ description further down the page.
The following type specific converter functions exist:

Float 32
^^^^^^^^^^
.. tab-set::

    .. tab-item:: Keras
        :sync: key-keras

        .. autofunction:: aifes.keras2aifes.convert_to_fnn_f32_express

    .. tab-item:: PyTorch
        :sync: key-pytorch

        .. autofunction:: aifes.pytorch2aifes.convert_to_fnn_f32_express




Quantized Q7
^^^^^^^^^^^^^^
.. _q7_aifes_e:

.. tab-set::

    .. tab-item:: Keras
        :sync: key-keras

        .. autofunction:: aifes.keras2aifes.convert_to_fnn_q7_express

    .. tab-item:: PyTorch
        :sync: key-pytorch

        .. autofunction:: aifes.pytorch2aifes.convert_to_fnn_q7_express



AIfES
------

AIfES offers an interface that is similar to other frameworks. With the following converter functions the FFNNs can be converted to the normal interface. It allows more
control over the FFNNs by enabling optimized layer implementations (e.g., CMSIS implementations for ARM Cortex M controller) and finer control of layers.
The following type specific converter functions exist:

Float 32
^^^^^^^^

Default Implementation
"""""""""""""""""""""""

For conversion to the default implementation of the layers use the following function:

.. tab-set::

    .. tab-item:: Keras
        :sync: key-keras

        .. autofunction:: aifes.keras2aifes.convert_to_fnn_f32

    .. tab-item:: PyTorch
        :sync: key-pytorch

        .. autofunction:: aifes.pytorch2aifes.convert_to_fnn_f32



CMSIS Implementation
"""""""""""""""""""""
.. _cmsis_f32_implementation:

For the optimized implementation of the layers using CMSIS use the following function:

.. tab-set::

    .. tab-item:: Keras
        :sync: key-keras

        .. autofunction:: aifes.keras2aifes.convert_to_fnn_f32_cmsis

    .. tab-item:: PyTorch
        :sync: key-pytorch

        .. autofunction:: aifes.pytorch2aifes.convert_to_fnn_f32_cmsis




Quantized Q7
^^^^^^^^^^^^
.. _q7_aifes:


Default Implementation
"""""""""""""""""""""""

For conversion to the default implementation of the layers and automatic quantization of your Model use the following function:

.. tab-set::

    .. tab-item:: Keras
        :sync: key-keras

        .. autofunction:: aifes.keras2aifes.convert_to_fnn_q7

    .. tab-item:: PyTorch
        :sync: key-pytorch

        .. autofunction:: aifes.pytorch2aifes.convert_to_fnn_q7



CMSIS Implementation
"""""""""""""""""""""
.. _cmsis_q7_implementation:

For the optimized implementation of the layers using CMSIS and automatic quantization of your Model use the following function:

.. tab-set::

    .. tab-item:: Keras
        :sync: key-keras

        .. autofunction:: aifes.keras2aifes.convert_to_fnn_q7_cmsis

    .. tab-item:: PyTorch
        :sync: key-pytorch

        .. autofunction:: aifes.pytorch2aifes.convert_to_fnn_q7_cmsis


