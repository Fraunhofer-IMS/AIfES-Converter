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
from .aifes_code_generator.aifes_code_creator import AifesCodeGenerator
from .keras_extractor.keras_extractor import KerasExtractor
from .support.aifes_model import AifesType
import numpy as np
from tensorflow import keras


def convert_to_fnn_f32_express(keras_model: keras.Model, output_path: str, with_weights=True):
    """
    Converts the given Keras model to AIfES Express F32. Creates one header file with the model in the output_path that can be included to any AIfES project. If weights are extracted as well, a second header file is created, which contains the flattened weights of the Keras model.

    :param keras_model: Keras model, which should be converted
    :param output_path: File path, where the converted model should be stored. If the folder doesn't exist, it will be created by this function.
    :param with_weights: Extracts the weights and bias from the given model and creates a separate header file with flatten weights.

    """
    keras_extractor = KerasExtractor(keras_model)
    aifestype = AifesType.EXPRESS

    aifescode = AifesCodeGenerator(keras_extractor, 'default', aifestype, output_path, with_weights)
    aifescode.run_f32()


def convert_to_fnn_q7_express(keras_model: keras.Model, output_path: str, representative_data: np.ndarray,
                              target_alignment: int, byteorder: str):
    """
    Converts the given Keras model to AIfES Express Q7. Creates one header file with the model in the output_path that can be included to any AIfES project. A second header file is created, which contains the flattened weights of the Keras model. This function converts the given Keras model to Q7. It needs representative data to calculate the quantization parameters.

    :param keras_model: Keras model, which should be converted
    :param output_path: File path, where the converted model should be stored. If the folder doesn't exist, it will be created by this function.
    :param representative_data: Representative data of the input data of the given Keras model. The data is needed to calculate the quantization parameters for the hidden layers.
    :param target_alignment: Alignment of the created flatbuffer depending on target architecture (1, 2, or 4 Bytes). E.g., for ARM Cortex M4 it is 4, which corresponds to 4 Bytes as it has a 32 Bit storage, for AVR Arduino it is 2, as the memory is organized as 16 Bit (2 Bytes)
    :param byteorder: Byte order of target system, i.e., 'little' for little endian or 'big' for big endian.

    """

    keras_extractor = KerasExtractor(keras_model, use_transposed_layers=False)
    aifestype = AifesType.EXPRESS

    aifescode = AifesCodeGenerator(keras_extractor, 'default', aifestype, output_path, with_weights=True)
    aifescode.run_q7(representative_data, target_alignment, byteorder)


def convert_to_fnn_f32(keras_model: keras.Model, output_path: str, with_weights=True):
    """
    Converts the given Keras model to AIfES F32 (non-express version). Creates one header file with the model in the output_path that can be included to any AIfES project. If weights are extracted as well, a second header file is created, which contains the flattened weights of the Keras model.

    :param keras_model: Keras model, which should be converted
    :param output_path: File path, where the converted model should be stored. If the folder doesn't exist, it will be created by this function.
    :param with_weights: Extracts the weights and bias from the given model and creates a separate header file with flattened weights.

    """
    keras_extractor = KerasExtractor(keras_model, use_transposed_layers=False)
    aifestype = AifesType.NORMAL

    aifescode = AifesCodeGenerator(keras_extractor, 'default', aifestype, output_path, with_weights)
    aifescode.run_f32()


def convert_to_fnn_f32_cmsis(keras_model: keras.Model, output_path: str, with_weights=True):
    """
    Converts the given Keras model to AIfES F32 CMSIS implementation (non-express version). Creates one header file with the model in the output_path that can be included to any AIfES project. If weights are extracted as well, a second header file is created, which contains the flattened weights of the Keras model.

    :param keras_model: Keras model, which should be converted
    :param output_path: File path, where the converted model should be stored. If the folder doesn't exist, it will be created by this function.
    :param with_weights: Extracts the weights and bias from the given model and creates separate header file wih flatten weights.

    """
    keras_extractor = KerasExtractor(keras_model)
    aifestype = AifesType.NORMAL

    aifescode = AifesCodeGenerator(keras_extractor, 'cmsis', aifestype, output_path, with_weights)
    aifescode.run_f32()


def convert_to_fnn_q7(keras_model: keras.Model, output_path: str, representative_data: np.ndarray,
                      target_alignment: int, byteorder: str, transpose=True):
    """
    Converts the given Keras model to AIfES Q7 implementation (non-express version). Creates one header file with the model in the output_path that can be included to any AIfES project. A second header file is created, which contains the flattened weights of the Keras model. This function converts the given Keras model to Q7. It needs representative data to calculate the quantization parameters.

    :param keras_model: Keras model, which should be converted
    :param output_path: File path, where the converted model should be stored. If the folder doesn't exist, it will be created by this function.
    :param representative_data: Representative data of the input data of the given Keras model. The data is needed to calculate the quantization parameters for the hidden layers.
    :param target_alignment: Alignment of the created flatbuffer depending on target architecture (1, 2, or 4 Bytes). E.g., for ARM Cortex M4 it is 4, which corresponds to 4 Bytes as it has a 32 Bit storage, for AVR Arduino it is 2, as the memory is organized as 16 Bit (2 Bytes)
    :param byteorder: Byte order of target system, i.e., 'little' for little endian or 'big' for big endian.
    :param transpose: When transpose=True the weights of the layers are transposed, so that the weights for each neuron are next to each other in memory. This can improve the performance of the ANN. Default is therefore 'True'.

    """
    keras_extractor = KerasExtractor(keras_model, transpose)
    aifestype = AifesType.NORMAL

    aifescode = AifesCodeGenerator(keras_extractor, 'default', aifestype, output_path, with_weights=True)
    aifescode.run_q7(representative_data, target_alignment, byteorder)


def convert_to_fnn_q7_cmsis(keras_model: keras.Model, output_path: str, representative_data: np.ndarray,
                            target_alignment: int, byteorder: str):
    """
    Converts the given Keras model to AIfES Q7 implementation with CMSIS support (non-express version). Creates one header file with the model in the output_path that can be included to any AIfES project. A second header file is created, which contains the flattened weights of the Keras model. This function converts the given Keras model to Q7. It needs representative data to calculate the quantization parameters.

    :param keras_model: Keras model, which should be converted
    :param output_path: File path, where the converted model should be stored. If the folder doesn't exist, it will be created by this function.
    :param representative_data: Representative data of the input data of the given Keras model. Is needed to calculate the quantization parameters for the hidden layers.
    :param target_alignment: Alignment of the created flatbuffer depending on target architecture (1, 2, or 4 Bytes). E.g., for ARM Cortex M4 it is 4, which corresponds to 4 Bytes as it has a 32 Bit storage, for AVR Arduino it is 2, as the memory is organized as 16 Bit (2 Bytes)
    :param byteorder: Byte order of target system, i.e., 'little' for little endian or 'big' for big endian.

    """
    keras_extractor = KerasExtractor(keras_model, use_transposed_layers=True)
    aifestype = AifesType.NORMAL

    aifescode = AifesCodeGenerator(keras_extractor, 'cmsis', aifestype, output_path, with_weights=True)
    aifescode.run_q7(representative_data, target_alignment, byteorder)
