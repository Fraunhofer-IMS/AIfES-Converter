"""
Copyright (C) 2022  Fraunhofer Institute for Microelectronic Circuits and Systems.
All rights reserved.
AIfES-Converter is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.
You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import warnings
from ..support.aifes_model import *

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.layers import ELU, LeakyReLU, ReLU, Softmax
    from tensorflow.keras.activations import sigmoid, softsign, tanh
except ImportError as err:
    raise ImportError("Tensorflow is not installed. Please make sure that you install Tensorflow in the right version "
                      "(>= 2.4) to convert your model from Keras to AIfES.")

from packaging import version
try:
    assert version.parse(tf.version.VERSION) >= version.parse('2.4.0')
except AssertionError as err:
    raise ImportError("Tensorflow is not installed in the required version. Please install version 2.4 and above.")



class KerasExtractor:
    """Keras Extractor Class. Provides interface functions for the AifesCodeGenerator to extract values from a
    Keras model"""

    # Activation Functions available in Dense Layer
    ACT_FUNCTIONS = ['elu', 'leakyrelu', 'leaky_relu', 'relu', 'softsign', 'softmax', 'sigmoid', 'tanh']
    # Separate Activation Functions as Keras.Layers
    ACT_LAYER = [ELU, LeakyReLU, ReLU, softsign, Softmax, sigmoid, tanh]

    def __init__(self, model: keras.Model, use_transposed_layers=False):
        """
        Initialize the KerasExtractor
        :param model: Keras Model which should be converted
        :param use_transposed_layers: If transposed layers should be used for the dense layers
        """

        self._model = model
        self._aifes_model = None
        self._use_transposed_layers = use_transposed_layers
        self._has_bias = True

    def extractor_structure(self) -> AifesModel:
        """
        Extracts the Keras model and saves it as an AIfES Model representation
        :return: Returns a representation of the Keras model as AIfES Model
        """

        # Local variables
        # Contains the AIfES structure after extraction
        aifes_fnn_structure = []
        # Get layer count
        layer_count = len(self._model.layers)
        aifes_layer_count = layer_count + 1

        # Go through each layer and extract values from it
        for x in range(0, layer_count, 1):
            curr_layer = self._model.layers[x]

            # Check if current layer is a dense layer
            if self._is_dense_layer(curr_layer):
                # Check if first layer, then we need to add an input layer
                if x == 0:
                    aifes_fnn_structure.append(AifesLayer_Input(self._model.layers[x].input_shape[1],
                                                                self._model.layers[x].input_shape[1]))

                # Add corresponding dense layer depending on if transposed layers should be used
                if not self._use_transposed_layers:
                    aifes_fnn_structure.append(AifesLayer_Dense(self._model.layers[x].units,
                                                                self._model.layers[x].units))
                else:
                    aifes_fnn_structure.append(AifesLayer_DenseTranspose(self._model.layers[x].units,
                                                                         self._model.layers[x].units))

                # Check if dense layer contains activation, if not, no activation is added
                if self._is_dense_layer_with_activation(curr_layer):
                    aifes_fnn_structure.append(self._get_activation_function(curr_layer))
                else:
                    if self._is_unsupported_activation_function(curr_layer):
                        raise ValueError(f"Unsupported activation function in layer {x}. See "
                                         f"https://fraunhofer-ims.github.io/AIfES_for_Arduino/#OverviewFeatures "
                                         f"for available activation functions.")



            # Check if current layer is an activation layer and is after the first layer
            elif self._is_activation_layer(curr_layer) and x > 0:
                # Add activation layer to AIfES model
                aifes_fnn_structure.append(self._get_activation_layer(curr_layer))

            # Layer is neither a dense nor activation layer, raise error
            else:
                if x == 0:
                    raise ValueError(f"First layer needs to be a dense layer. Got '{curr_layer}' instead.")
                else:
                    raise ValueError(f"Unsupported layer chosen. Got '{curr_layer}', but must be one of "
                                 "Dense, ELU, LeakyReLU, linear, relu, sigmoid, softmax, softsign or "
                                 "tanh")

        # Create AIfES Model and return it
        self._aifes_model = AifesModel(aifes_fnn_structure, aifes_layer_count, None)
        return self._aifes_model

    def extractor_values(self):
        """
        Extracts the values of a Keras model and returns them
        :return: Extracted weights
        """
        if not self._has_bias:
            raise ValueError("Your model needs dense layer with bias for a conversion to AIfES with weights. Please "
                             "ensure that your layers have bias.")

        weights = self._model.get_weights()
        return weights

    def get_transpose_status(self) -> bool:
        """
        Returns status, if transposed layers should be used
        :return: Bool, True if transposed layers are used, otherwise False
        """
        return self._use_transposed_layers

    def _is_dense_layer(self, curr_layer) -> bool:
        """
        Checks if current layer is a correctly formated dense layer
        :param curr_layer: Layer of the model, which should be checked
        :return: True, if current layer is dense layer, otherwise False
        """
        if curr_layer.__class__.__name__ == 'Dense':
            if self._is_correctly_configured_dense_layer(curr_layer):
                return True
            else:
                return False
        else:
            return False

    def _is_dense_layer_with_activation(self, curr_layer) -> bool:
        """
        Checks is activation function is part of self.ACT_FUNCTIONS, and has therefore an activation function. Linear activation function is default and therefore not considered as activation function.
        :param curr_layer: Current layer, which should be checked
        :return: True, if activation function is set and supported, otherwise False
        """
        # Get activation function
        layer_config = curr_layer.get_config()
        acti = layer_config["activation"]

        # When configurable activation function, acti is of type dict. We need only the name, so we extract it here
        if type(acti) is dict:
            acti = acti['class_name'].lower()

        # Check if acti is part of ACT_FUNCTIONS
        if acti in self.ACT_FUNCTIONS:
            return True
        else:
            return False

    def _get_activation_function(self, curr_layer) -> AifesLayer:
        """
        Returns the activation layer for AIfES of the curr_layer. Extracts the value from a dense layer with set activation function.
        :param curr_layer: Current layer, from which the activation function should be extracted
        :return: AifesLayer with the initialized AIfES activation layer
        """
        # Get activation function
        layer_config = curr_layer.get_config()
        acti = layer_config["activation"]

        # When configurable activation function, acti is of type dict. We need only the name, so we extract it here
        if type(acti) is dict:
            acti = acti['class_name'].lower()

        # Return corresponding activation layer
        if acti == 'elu':
            if type(layer_config["activation"]) is dict:
                return AifesLayer_Elu(layer_config["activation"]["config"]["alpha"])
            else:
                warnings.warn("Elu layer was not customized. The default alpha value of 1.0 is used. ")
                return AifesLayer_Elu(alpha_value=1.0)
        elif acti == 'leakyrelu':
            if type(layer_config["activation"]) is dict:
                return AifesLayer_Leaky_ReLU(layer_config["activation"]["config"]["alpha"])
            else:
                warnings.warn("LeakyRelu was not customized. The default alpha value of 0.3 is used. ")
                return AifesLayer_Leaky_ReLU(alpha_value=0.3)
        elif acti == 'leaky_relu':
            warnings.warn("LeakyRelu was not customized. The default alpha value of 0.3 is used. ")
            return AifesLayer_Leaky_ReLU(alpha_value=0.3)
        elif acti == 'linear':
            return AifesLayer_Linear()
        elif acti == 'relu':
            return AifesLayer_ReLU()
        elif acti == 'sigmoid':
            return AifesLayer_Sigmoid()
        elif acti == 'softmax':
            return AifesLayer_Softmax()
        elif acti == 'softsign':
            return AifesLayer_Softsign()
        elif acti == 'tanh':
            return AifesLayer_Tanh()
        else:
            raise ValueError(
                "Unsupported activation in layer. Got " + str(acti) + ", but must be part of"
                "ELU, LeakyReLU, linear, relu, sigmoid, softmax, softsign or tanh")

    def _is_activation_layer(self, curr_layer) -> bool:
        """
        Check if current layer is an activation layer. Checks self.ACT_LAYER if curr_layer is included.
        :param curr_layer: Current layer
        :return: True, if current layer is activation layer, otherwise False
        """
        if type(curr_layer) in self.ACT_LAYER:
            return True
        else:
            return False

    def _get_activation_layer(self, curr_layer) -> AifesLayer:
        """
        Returns the activation layer for AIfES of the curr_layer. Checks the type of the curr_layer. (Independent activation function, not set with Dense layer)
        :param curr_layer: Current layer
        :return: AifesLayer with the initialized AIfES activation layer
        """
        layer_type = type(curr_layer)
        if layer_type == ELU:
            return AifesLayer_Elu(curr_layer.alpha)
        elif layer_type == LeakyReLU:
            return AifesLayer_Leaky_ReLU(curr_layer.alpha)
        elif layer_type == ReLU:
            return AifesLayer_ReLU()
        elif layer_type == sigmoid:
            return AifesLayer_Sigmoid()
        elif layer_type == Softmax:
            return AifesLayer_Softmax()
        elif layer_type == softsign:
            return AifesLayer_Softsign()
        elif layer_type == tanh:
            return AifesLayer_Tanh()
        else:
            raise ValueError("Unsupported activation layer " + str(layer_type) + ". Activation Layer needs to be of"
                             " type ELU, LeakyReLU, ReLU, Sigmoid, Softmax, Softsign or Tanh")

    def _is_unsupported_activation_function(self, curr_layer):
        # Get activation function
        layer_config = curr_layer.get_config()
        acti = layer_config["activation"]

        # When configurable activation function, acti is of type dict. We need only the name, so we extract it here
        if type(acti) is dict:
            acti = acti['class_name'].lower()

        if acti == 'linear':
            return False
        else:
            return True

    def _is_correctly_configured_dense_layer(self, curr_layer):
        if str(curr_layer.dtype) != 'float32':
            raise ValueError(f"A dense layer has not the data type 'float32', but instead {curr_layer.dtype}. Please "
                             f"change it to 'float32'")
        if str(curr_layer.use_bias) != 'True':
            self._has_bias = False

        return True




