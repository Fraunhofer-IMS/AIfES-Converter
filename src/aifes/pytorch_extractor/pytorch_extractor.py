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
from .pytorch_extractor_utils import get_layer_list
from ..support.aifes_model import *

try:
    import torch
    from torch.nn import Module
    from torch.nn import Linear
    from torch.nn import ELU, LeakyReLU, ReLU, Sigmoid, Softmax, Softsign, Tanh
except ImportError as err:
    raise ImportError("PyTorch is not installed. Please make sure that you install PyTorch in the right version "
                      "(>= 1.8) to convert your model from PyTorch to AIfES")

from packaging import version
try:
    assert version.parse(torch.version.__version__) >= version.parse('1.8.0')
except AssertionError as err:
    raise ImportError("PyTorch is no installed in the required version. Please install version 1.8 and above.")


class PytorchExtractor:
    """PyTorch Extractor Class. Provides interface functions for the AifesCodeGenerator to extract values from a PyTorch model"""

    # Activation Layers available in AIfES as Type from torch.nn
    ACT_FUNCTIONS = [ELU, LeakyReLU, ReLU, Softsign, Softmax, Sigmoid, Tanh]

    def __init__(self, model: Module, use_transposed_layers=False):
        """
        Initialize the PyTorchExtractor
        :param model: PyTorch Model which should be converted
        :param use_transposed_layers: If transposed layers should be used for the dense layers
        """
        self._model = model
        self._aifes_model = None
        self._use_transposed_layers = use_transposed_layers
        self._has_bias = True

    def extractor_structure(self) -> AifesModel:
        """
        Extracts the PyTorch model and saves it as an AIfES Model representation
        :return: Returns a representation of the PyTorch model as AIfES Model
        """

        # Local variables
        # Contains the PyTorch layer model as a list
        fnn_structure = get_layer_list(self._model)
        # Contains the AIfES structure after extraction
        aifes_fnn_structure = []
        # Get layer count
        layer_count = len(fnn_structure)
        aifes_layer_count = self._get_layer_cnt(fnn_structure) + 1

        # Go through each layer and extract values from it
        for x in range(0, layer_count, 1):
            curr_layer = fnn_structure[x]

            # If first layer, we need to add an input layer
            if x == 0:
                if self._is_dense_layer(curr_layer):
                    aifes_fnn_structure.append(AifesLayer_Input(curr_layer.in_features, curr_layer.in_features))
                else:
                    raise ValueError("First layer of the model needs to be a 'linear' layer. Got " + str(type(curr_layer))
                                 + " instead.")
            # Check if dense layer
            if self._is_dense_layer(curr_layer):
                # Add corresponding dense layer depending on if transposed layers should be used
                if not self._use_transposed_layers:
                    aifes_fnn_structure.append(AifesLayer_Dense(curr_layer.out_features, curr_layer.out_features))
                else:
                    aifes_fnn_structure.append(AifesLayer_DenseTranspose(curr_layer.out_features, curr_layer.out_features))

            # Check if activation layer
            elif self._is_activation_layer(curr_layer):
                aifes_fnn_structure.append(self._get_activation_layer(curr_layer))

            # Layer is neither a dense nor a supported activation layer, raise error
            else:
                raise ValueError("Unsupported layer in layer " + str(x) + ". Got " + str(curr_layer) + ", but must be part of"
                                 " ELU, LeakyReLU, Linear, ReLU, Sigmoid, Softmax, Softsign or Tanh")

        # Export AIfES model and return it
        self._aifes_model = AifesModel(aifes_fnn_structure, aifes_layer_count, None)
        return self._aifes_model

    def extractor_values(self):
        """
        Extracts the values of the PyTorch model and returns them
        :return: Extracted weights
        """
        if not self._has_bias:
            raise ValueError("Your model needs linear layer with bias for a conversion to AIfES with weights. Please "
                             "ensure that your layers have bias.")

        weights = [param.detach().numpy().T for param in self._model.parameters()]
        return weights

    def get_transpose_status(self) -> bool:
        """
       Returns status, if transposed layers should be used
       :return: Bool, True if transposed layers are used, otherwise False
       """
        return self._use_transposed_layers

    def _is_dense_layer(self, curr_layer) -> bool:
        """
        Checks if current layer is a dense layer
        :param curr_layer: Layer of the model, which should be checked
        :return: True, if current layer is dense layer, otherwise False
        """
        if type(curr_layer) is Linear:
            if self._is_correctly_configured_dense_layer(curr_layer):
                return True
            else:
                return False
        else:
            return False

    def _get_layer_cnt(self, model) -> int:
        """
        Count the number of fnn (Linear) in the PyTorch net
        :return:    Number of layer
        """
        layer_cnt = 0
        for layer in model:
            if type(layer) is Linear:
                layer_cnt += 1

        return layer_cnt

    def _is_activation_layer(self, curr_layer) -> bool:
        """
        Check if current layer is an activation layer
        :param curr_layer: Current layer from model
        :return: True/False depending on layer type
        """
        if type(curr_layer) in self.ACT_FUNCTIONS:
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
            return AifesLayer_Leaky_ReLU(curr_layer.negative_slope)
        elif layer_type == ReLU:
            return AifesLayer_ReLU()
        elif layer_type == Sigmoid:
            return AifesLayer_Sigmoid()
        elif layer_type == Softmax:
            return AifesLayer_Softmax()
        elif layer_type == Softsign:
            return AifesLayer_Softsign()
        elif layer_type == Tanh:
            return AifesLayer_Tanh()
        else:
            raise ValueError("Unsupported activation layer " + str(layer_type) + ". Activation Layer needs to be of type "
                             "ELU, LeakyReLU, ReLU, Sigmoid, Softmax, Softsign or Tanh")

    def _is_correctly_configured_dense_layer(self, curr_layer):
        if curr_layer.bias is None:
            self._has_bias = False

        return True
