Usage
============

AIfES-Converter is designed to allow an easy conversion of your model from Keras and PyTorch to AIfES. Different types of
Feed Forward Neural Networks (FFNNs) are supported with different data types. In the following section we show how to export the model to
AIfES using this tool.

Export of model
---------------

Only Keras and PyTorch are supported directly by the AIfES-Converter converter. But you can use any other framework where the
FFNNs can be imported by Keras or PyTorch. We first show how to use this tool. Then a brief overview is given, how to
use this tool with other frameworks.

Keras and PyTorch
^^^^^^^^^^^^^^^^^^

In the following section we provide a detailed description of the necessary steps.


1. Create the model
"""""""""""""""""""""

.. tab-set::

    .. tab-item:: Keras
        :sync: key-keras

        Create a model in Python by using Keras. Keep an eye on the supported features of AIfES in the
        |documentation| as only those functions can be
        converted. You can also load a stored model and convert it to AIfES.

        .. |documentation| raw:: html

            <a href="https://fraunhofer-ims.github.io/AIfES_for_Arduino/#OverviewFeatures" target="_blank">documentation</a>


        .. dropdown:: Example
            :open:
            :animate: fade-in-slide-down

            In this example different options for creation and loading of Keras models are shown.

            .. tab-set::

                .. tab-item:: Create model & train it or set parameters manually

                    Here we show you how you can create your model with Keras. You can choose to either train your model
                    or to set the weights and biases manually. For this you need to create the model first:

                    .. code-block::

                        import tensorflow
                        from tensorflow import keras

                        model = keras.Sequential()
                        model.add(keras.layers.Input(shape=(2,)))
                        model.add(keras.layers.Dense(3, activation='sigmoid'))
                        model.add(keras.layers.Dense(1, activation='sigmoid'))

                    After the creation of your model, you can now either train it or set the weights and biases.

                    .. tab-set::

                        .. tab-item:: Train the model

                            .. code-block::

                                # Train your model
                                optimizer = keras.optimizers.Adam(learning_rate=0.1)
                                model.compile(optimizer=optimizer, loss="binary_crossentropy")

                                model.summary()

                                X = np.array([[0., 0.],
                                              [1., 1.],
                                              [1., 0.],
                                              [0., 1.]])

                                T = np.array([[0.],
                                              [0.],
                                              [1.],
                                              [1.]])

                                model.fit(X, T, batch_size=4, epochs=200)

                        .. tab-item:: Set the weights and biases of the model

                            .. code-block::

                                # You may set the weights manually instead of training the model.
                                w1 = np.array([ 0.9368746 , -0.29152113,  0.29609978,
                                                -0.14289689, -0.56332463, -0.567354  ]).reshape(2, 3)
                                b1 = np.array([0.72655, 2.67281, -0.21291])

                                w2 = np.array([1.0711929, 0.34211,
                                               0.0844624]).reshape(3, 1)
                                b2 = np.array([0.14391])

                                weights = [w1, b1, w2, b2]
                                model.set_weights(weights)

                            Then you can export the model. During export an executable header file is created.

                .. tab-item:: Load model

                    You can load stored models and convert them in the next step. To load the model follow the next steps:

                    .. code-block::

                        from tensorflow import keras

                        # If you stored your model with:
                        # model.save('path/to/location')
                        # Then you can load it with:
                        model = keras.models.load_model('path/to/location')

                        # If your model is stored as a *.h5 file you can load it with:
                        # model = keras.models.load_model("my_h5_model.h5")

    .. tab-item:: Pytorch
        :sync: key-pytorch

        Create a model in Python by using PyTorch. Keep an eye on the supported features of AIfES in the
        |documentation| as only those functions can be
        converted. You can also load a stored model and convert it to AIfES.


        .. dropdown:: Example
            :open:
            :animate: fade-in-slide-down

            In this example different options for creation and loading of PyTorch models are shown.

            .. tab-set::

                .. tab-item:: Create model

                    .. code-block::

                        import numpy as np
                        import torch
                        from torch import nn

                        class Net(nn.Module):

                        def __init__(self):
                            super(Net, self).__init__()

                            self.dense_layer_1 = nn.Linear(2, 3)
                            self.leaky_relu_layer = nn.LeakyReLU(0.01)
                            self.dense_layer_2 = nn.Linear(3, 1)
                            self.sigmoid_layer = nn.Sigmoid()

                        def forward(self, x):
                            x = self.dense_layer_1(x)
                            x = self.leaky_relu_layer(x)
                            x = self.dense_layer_2(x)
                            x = self.sigmoid_layer(x)
                            return x


                        X = np.array([[0., 0.],
                                      [1., 1.],
                                      [1., 0.],
                                      [0., 1.]])

                        Y = np.array([[0.],
                                      [0.],
                                      [1.],
                                      [1.]])


                        X_tensor = torch.FloatTensor(X)
                        Y_tensor = torch.FloatTensor(Y)

                        model = Net()
                        criterion = nn.BCELoss()
                        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

                        epochs = 200
                        model.train()
                        for epoch in range(epochs):
                            optimizer.zero_grad()
                            pred = model(X_tensor)
                            loss = criterion(pred, Y_tensor)
                            loss.backward()
                            optimizer.step()



                .. tab-item:: Load model

                    You can also load a presaved model by following the next steps. Other methods are also possible,
                    but not listed here. See the documentation from PyTorch for more details.

                    .. code-block::

                        import torch

                        # Saving model:
                        # torch.save(model, 'path/to/location')

                        # Load model:
                        model = torch.load('path/to/location')





.. _export-model:

2. Export the model using AIfES-Converter
"""""""""""""""""""""""""""""""""""""""""""

Choose the desired AIfES FFNN implementation from the available ones :ref:`here<available-functions>`. Call the 
converter tool with the chosen implementation. The converter 
tool will create the necessary header files for usage in your IDE in the destination file path `path/to/location`.

.. note::
	If the output file path doesn't exist, the corresponding folder will be created.

The converter tool allows you also to only export the structure of your model without any weights and biases. For this you need to set the option ``with_weights=False`` during conversion, if supported by the corresponding
function (currently only supported for float32 based implementation, see :ref:`available functions<available-functions>`, as Q7 needs a representative dataset and
quantization parameters per layer). This creates a header file, which is prepared to include your own weights and biases.

.. attention::
    The created header file of the conversion without weights is not executable. Please see the comments in the header file regarding the weights and biases in the function
    ``aifes_f32_fnn_create_model()``/``aifes_e_f32_fnn_inference()``.

.. dropdown:: Example
    :open:
    :animate: fade-in-slide-down

    In this example we want to convert the model to an FFNN with floating point precision and the normal AIfES interface:

    .. tab-set::

        .. tab-item:: Keras
            :sync: key-keras

            .. code-block::

                from aifes import keras2aifes

                keras2aifes.convert_to_fnn_f32(model, 'path/to/location')

        .. tab-item:: PyTorch
            :sync: key-pytorch

            .. code-block::

                from aifes import pytorch2aifes

                pytorch2aifes.convert_to_fnn_f32(model, 'path/to/location')




3. Import the model into your project
""""""""""""""""""""""""""""""""""""""

Import the files in your IDE and use the prepared function calls on top of the header file to call the FFNN in AIfES.
For this, copy the header files from the ``OUTPUT_PATH`` and add them to your project in your desired IDE, as you would add
any other header file. Make sure, that you have installed the AIfES Library, either via the Library Manager in the 
Arduino IDE or by following the instructions in our |GitHub_Repository| for other IDEs. Then import the header files into your
project.

.. |GitHub_Repository| raw:: html

    <a href="https://github.com/Fraunhofer-IMS/AIfES_for_Arduino#installation-guides-for-various-ides" target="_blank">GitHub Repository</a>

.. dropdown:: Example Arduino IDE
    :open:
    :animate: fade-in-slide-down

    To import the header files into an Arduino Project you need a saved project. Open the folder, where your project is stored (``*.ino`` file). If you dont have a project yet, create one and save it. After this open the folder.
    Copy the created header files from the AIfES-Converter into the folder with the ``*.ino`` file. Close the project in the Arduino IDE and then open it again. The header files should be now included in the project and be visible as separate tabs.
    Import the header files into your ``*.ino`` file by adding the following two lines to your file:
	
    .. tab-set::

        .. tab-item:: AIfES Express Example
            :sync: key-aifes_e
			
            .. code-block:: C

                // For F32 based implementation
                #import "aifes_e_f32_fnn.h"
                #import "aifes_e_f32_weights.h"

                // For Q7 based implementation
                #import "aifes_e_q7_fnn.h"
                #import "aifes_e_q7_weights.h"


        .. tab-item:: AIfES Example
            :sync: key-aifes
			
            .. code-block:: C

                // For F32 based implementation
                #import "aifes_f32_fnn.h"
                #import "aifes_f32_weights.h"

                // For Q7 based implementation
                #import "aifes_q7_fnn.h"
                #import "aifes_q7_weights.h"

				
4. Call the prepared functions
"""""""""""""""""""""""""""""""

Finally, the prepared functions need to be called by your application. The necessary function calls are in the comments at the top
of the header file for the fnn ``aifes_e_{dtype}_fnn.h`` respectively ``aifes_{dtype}_fnn.h``.

.. dropdown:: Example
    :open:
    :animate: fade-in-slide-down

    .. tab-set::

        .. tab-item:: AIfES Express Example
            :sync: key-aifes_e
			
            .. code-block:: C

                // Define of input and output array
                float input_data[2]; // AIfES input data
                float output_data[1]; // AIfES output data

                // Set input array
                input_data[0] = 0;
                input_data[1] = 1;

                // Run inference
                aifes_e_f32_fnn_inference((float*)input_data,(float*)output_data);

                // Print results
                printf("Results: %.5f", output_data[0]);


        .. tab-item:: AIfES Example
            :sync: key-aifes

            .. code-block:: C

                // You first need to initialize the model once with:
                // ---------------------------------------------------------------------------
                uint8_t error;
                error = aifes_f32_fnn_create_model();
                if (error == 1)
                {
                    //do something for error handling
                    //e.g. while(true)
                }
                // ---------------------------------------------------------------------------
                // Please check the error. It is 1 if something inside the initialization failed. Check the output.

                // Paste the following code into your project and insert your inputs/features:
                // ---------------------------------------------------------------------------
                float input_data[2]; // AIfES input data
                float output_data[1]; // AIfES output data

                 // Set input array
                input_data[0] = 0;
                input_data[1] = 1;

                // Run inference
                aifes_f32_fnn_inference((float*) input_data, (float*) output_data);

                // Print results
                printf("Results: %.5f", output_data[0]);
                // ---------------------------------------------------------------------------


    .. note::
        The input and output dimensions need to be updated to your FFNN.
        A customized function call is provided in the header file. There, the input and output dimensions are automatically updated
        to your specific model. Therefore, use the function calls from the header file.

Other Frameworks
^^^^^^^^^^^^^^^^^^

You can convert models from other frameworks to AIfES as well, as long as a conversion of those models to Keras or PyTorch
and then to AIfES is possible. Otherwise, you can recreate the model using Keras and set the weights and biases manually and then
export the model to AIfES. For ONNX you can use PyTorch, as it supports ONNX out of the box.
For more details about that see |pytorch_link|.

.. |pytorch_link| raw:: html

    <a href="https://pytorch.org/docs/stable/onnx.html" target="_blank">PyTorchs ONNX Documentation</a>
