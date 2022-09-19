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
from typing import List
import numpy as np
from enum import Enum


# Enum of the different types of layers
class Layer(Enum):
    DENSE = 1
    DENSE_WT = 11  # Dense layer with transposed weights matrix
    ELU = 2
    INPUT = 3
    LEAKY_RELU = 4
    LINEAR = 5
    RELU = 6
    SIGMOID = 7
    SOFTMAX = 8
    SOFTSIGN = 9
    TANH = 10


# Definition of activation layers
act_layer = [Layer.ELU, Layer.LINEAR, Layer.LEAKY_RELU, Layer.SOFTSIGN, Layer.SOFTMAX, Layer.SIGMOID, Layer.RELU,
             Layer.TANH]
# Definition of configurable activation layers
configurable_act_layer = [Layer.ELU, Layer.LEAKY_RELU]


# Enum of different data types
class Dtype(Enum):
    FLOAT32 = 1
    Q31 = 2
    Q7 = 3

# Enum of different AIfES Frontends
class AifesType(Enum):
    EXPRESS = 1
    NORMAL = 2

# Dictionary to convert from Dtype to AIfES specific names
dtype_to_aifes = {Dtype.FLOAT32: 'f32', Dtype.Q31: 'q31', Dtype.Q7: 'q7'}


# Super class for AifesLayer containing common variables
class AifesLayer:
    def __init__(self, layer_type: Layer, layer_name: str, input_shape: np.ndarray, output_shape: np.ndarray):
        self.layer_type = layer_type
        # Layer name
        self.layer_name = layer_name
        # Input Shape
        self.input_shape = input_shape
        # Output Shape
        self.output_shape = output_shape
        self.init_macro = None

    # Add print options for easier debugging
    def __str__(self):
        output_str = "Layer Type: "
        output_str += str(self.layer_type) + ", " + self.layer_name
        output_str += "; Input Shape: " + str(self.input_shape)
        output_str += "; Output Shape: " + str(self.output_shape)
        return output_str


# Type dependent class for each layer type with corresponding init_macro
class AifesLayer_Dense(AifesLayer):
    def __init__(self, input_shape: np.ndarray, output_shape: np.ndarray, layer_name='dense'):
        super().__init__(Layer.DENSE, layer_name, input_shape, output_shape)
        # Init macro
        self.init_macro = "AILAYER_DENSE_{DTYPE_C}_A(" + str(input_shape) + ");"


class AifesLayer_DenseTranspose(AifesLayer):
    def __init__(self, input_shape: np.ndarray, output_shape: np.ndarray, layer_name='dense'):
        super().__init__(Layer.DENSE_WT, layer_name, input_shape, output_shape)
        # Init macro
        self.init_macro = "AILAYER_DENSE_{DTYPE_C}_A(" + str(input_shape) + ");"


class AifesLayer_Elu(AifesLayer):
    def __init__(self,  alpha_value: float, input_shape=None, output_shape=None, layer_name='elu' ):
        super().__init__(Layer.ELU, layer_name, input_shape, output_shape)
        # Alpha Value
        self.alpha_value = alpha_value
        # Init macro
        self.init_macro = "AILAYER_ELU_{DTYPE_C}_A({Q_START_INIT}" + str(alpha_value) + "{Q_STOP_INIT});"


class AifesLayer_Input(AifesLayer):
    def __init__(self, input_shape=None, output_shape=None, layer_name='input'):
        super().__init__(Layer.INPUT, layer_name, input_shape, output_shape)
        # Init macro
        self.init_macro = "AILAYER_INPUT_{DTYPE_C}_A(" + "2" + ", input_layer_shape);"


class AifesLayer_Linear(AifesLayer):
    def __init__(self, input_shape=None, output_shape=None, layer_name='linear'):
        super().__init__(Layer.LINEAR, layer_name, input_shape, output_shape)
        # Init macro
        self.init_macro = None


class AifesLayer_Leaky_ReLU(AifesLayer):
    def __init__(self, alpha_value: float, input_shape=None, output_shape=None, layer_name='leaky_relu'):
        super().__init__(Layer.LEAKY_RELU, layer_name, input_shape, output_shape)
        # Alpha Value
        self.alpha_value = alpha_value
        # Init macro
        self.init_macro = "AILAYER_LEAKY_RELU_{DTYPE_C}_A({Q_START_INIT}" + str(alpha_value) + "{Q_STOP_INIT});"


class AifesLayer_ReLU(AifesLayer):
    def __init__(self, input_shape=None, output_shape=None, layer_name='relu'):
        super().__init__(Layer.RELU, layer_name, input_shape, output_shape)
        # Init macro
        self.init_macro = "AILAYER_RELU_{DTYPE_C}_A();"


class AifesLayer_Sigmoid(AifesLayer):
    def __init__(self, input_shape=None, output_shape=None, layer_name='sigmoid'):
        super().__init__(Layer.SIGMOID, layer_name, input_shape, output_shape)
        # Init macro
        self.init_macro = "AILAYER_SIGMOID_{DTYPE_C}_A();"


class AifesLayer_Softmax(AifesLayer):
    def __init__(self, input_shape=None, output_shape=None, layer_name='softmax'):
        super().__init__(Layer.SOFTMAX, layer_name, input_shape, output_shape)
        # Init macro
        self.init_macro = "AILAYER_SOFTMAX_{DTYPE_C}_A();"


class AifesLayer_Softsign(AifesLayer):
    def __init__(self, input_shape=None, output_shape=None, layer_name='softsign'):
        super().__init__(Layer.SOFTSIGN, layer_name, input_shape, output_shape)
        # Init macro
        self.init_macro = "AILAYER_SOFTSIGN_{DTYPE_C}_A();"


class AifesLayer_Tanh(AifesLayer):
    def __init__(self, input_shape=None, output_shape=None, layer_name='tanh'):
        super().__init__(Layer.TANH, layer_name, input_shape, output_shape)
        # Init macro
        self.init_macro = "AILAYER_TANH_{DTYPE_C}_A();"


# Class for AIfES Model, contains the extracted structure of the Keras Model
class AifesModel:
    def __init__(self, aifes_fnn_structure: List[AifesLayer], aifes_layer_count: int, flatten_aifes_weights: list):
        self.aifes_fnn_structure = aifes_fnn_structure
        self.aifes_layer_count = aifes_layer_count
        self.flatten_aifes_weights = flatten_aifes_weights

    # Define str function for easier debug
    def __str__(self):
        output_str = "####AIfES Model####\n"
        for el in self.aifes_fnn_structure:
            output_str += str(el) + "\n"
        output_str += "Layer count: " + str(self.aifes_layer_count) + "\n"
        output_str += "Layer Weights: "
        output_str += str(self.flatten_aifes_weights)
        return output_str
