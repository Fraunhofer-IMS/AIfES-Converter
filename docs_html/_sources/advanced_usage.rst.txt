Advanced Usage
==============

This converter tool allows for advanced configuration options during conversion. You can enable the usage of the CMSIS
for ARM based controller to accelerate the inference of your model. Also, this convert tool can quantize your model to
Q7 .

Quantization of your FFNN
--------------------------

With this converter you can quantize your model to Q7. This converts the floating point precision FFNN to a fixed point precision with 8 bit resolution.
This reduces the storage size of your model and can improve the performance of FFNNs on microcontrollers without
Floating Point Unit (e.g. Arduino Uno). For more details about the quantization used in AIfES see |quant_git_aifes|.


To convert your model the convert needs a representative data set of the input data. Make sure, that the range of all inputs
is representable with 8 bits of resolution. You will receive an error if this is not possible. To prevent this you can scale
your inputs to zero mean and unit variance. Ensure that your representative data set contains the highest and lowest expected values for your inputs, so
that the neural net is capable of representing those values as well. This helps to avoid saturation of your inputs and therefore
maybe false results. In the following we will show some examples.

See the documentation for :ref:`AIfES Express<q7_aifes_e>` and :ref:`AIfES<q7_aifes>` for detailed description of the available configuration parameters.
In the following table are some common configurations for the Arduino Boards (respectively their microcontrollers):

+---------------------------------+-----------------------------+-----------+
| Board / Microcontroller         | target_alignment            | byteorder |
+=================================+=============================+===========+
| Arduino Uno                     | 2 (16 Bit),                 | little    |
| (AVR ATmega328P)                | Storage organized as 16 Bit |           |
+---------------------------------+-----------------------------+-----------+
| Arduino Mega                    | 2 (16 Bit),                 | little    |
| (AVR ATmega2560)                | Storage organized as 16 Bit |           |
+---------------------------------+-----------------------------+-----------+
| Arduino Micro                   | 2 (16 Bit),                 | little    |
| (AVR ATmega32U4)                | Storage organized as 16 Bit |           |
+---------------------------------+-----------------------------+-----------+
| Arduino Nano 33 BLE /           | 4 (32 Bit),                 | little    |
| Sense (nRF52840, ARM Cortex-M4) | Storage organized as 32 Bit |           |
+---------------------------------+-----------------------------+-----------+

.. |quant_git_aifes| raw:: html

            <a href="https://fraunhofer-ims.github.io/AIfES_for_Arduino/_tutorial_inference_q7.html#autotoc_md33" target="_blank">Q7 documentation</a>


.. dropdown:: Example
    :open:
    :animate: fade-in-slide-down

    The following examples show how to convert your model. For this you need to load or setup your Keras or PyTorch model
    as described in the :doc:`usage` section.

    .. tab-set::

        .. tab-item:: Keras
            :sync: key-keras

            .. tab-set::

                .. tab-item:: AIfES Express Example
                    :sync: key-aifes_e

                    .. code-block::

                        from aifes import keras2aifes

                        representative_dataset = np.array([[0, 0],
                                                           [0, 1],
                                                           [1, 0],
                                                           [1, 1]])

                        # Example for 32 bit storage alignment (4 Byte) and little endian representation
                        keras2aifes.convert_to_fnn_q7_express(model, 'path/to/location/keras/output', representative_data=representative_dataset, target_alignment=4, byteorder="little")

                .. tab-item:: AIfES Example
                    :sync: key-aifes

                    .. code-block::

                        from aifes import keras2aifes

                        representative_dataset = np.array([[0, 0],
                                                           [0, 1],
                                                           [1, 0],
                                                           [1, 1]])

                        # Example for 32 bit storage alignment (4 Byte) and little endian representation
                        keras2aifes.convert_to_fnn_q7(model, 'path/to/location/keras/output', representative_data=representative_dataset, target_alignment=4, byteorder="little")

        .. tab-item:: Pytorch
            :sync: key-pytorch

            .. tab-set::

                .. tab-item:: AIfES Express Example
                    :sync: key-aifes_e

                    .. code-block::

                        from aifes import pytorch2aifes

                        representative_dataset = np.array([[0, 0],
                                                           [0, 1],
                                                           [1, 0],
                                                           [1, 1]])

                        # Example for 32 bit storage alignment (4 Byte) and little endian representation
                        pytorch2aifes.convert_to_fnn_q7_express(model, 'path/to/location/pytorch/output', representative_data=representative_dataset, target_alignment=4, byteorder="little")

                .. tab-item:: AIfES Example
                    :sync: key-aifes

                    .. code-block::

                        from aifes import pytorch2aifes

                        representative_dataset = np.array([[0, 0],
                                                           [0, 1],
                                                           [1, 0],
                                                           [1, 1]])

                        # Example for 32 bit storage alignment (4 Byte) and little endian representation
                        pytorch2aifes.convert_to_fnn_q7(model, 'path/to/location/pytorch/output', representative_data=representative_dataset, target_alignment=4, byteorder="little")


Use CMSIS for ARM-based microcontrollers
-----------------------------------------

To use CMSIS in your model you can use the intended converter functions (for Float see :ref:`CMSIS F32<cmsis_f32_implementation>`
and for Q7 see :ref:`CMSIS Q7<cmsis_q7_implementation>`). To be able to use these functions you need to have CMSIS available
in your IDE. For the Arduino IDE follow the instructions on our |github_cmsis|. For other IDEs follow the instructions of
your IDE.

.. |github_cmsis| raw:: html

            <a href="https://github.com/Fraunhofer-IMS/AIfES_for_Arduino#arm-cmsis" target="_blank">GitHub Repository</a>

.. tab-set::

        .. tab-item:: Keras
            :sync: key-keras

            .. code-block::

                from aifes import keras2aifes

                # Example for F32, also as Q7 possible
                keras2aifes.convert_to_fnn_f32_cmsis(model, 'path/to/location/keras/output')

        .. tab-item:: Pytorch
            :sync: key-pytorch

            .. code-block::

                from aifes import pytorch2aifes

                # Example for F32, also as Q7 possible
                pytorch2aifes.convert_to_fnn_f32_cmsis(model, 'path/to/location/pytorch/output')
