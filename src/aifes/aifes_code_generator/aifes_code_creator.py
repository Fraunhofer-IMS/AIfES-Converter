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
import os
from pkg_resources import resource_filename
from ..support.aifes_model import AifesType
from ..support.support_functions import flatten_weights, create_c_array_str
from ..model_converter.support_model_conversion_q7 import str_flatbuffer_c_style
from .support_aifes_code_creator import *
from ..model_converter.model_conversion import ModelConversion
import numpy as np


class AifesCodeGenerator:
    """Aifes Code Generator Class. Uses an AifesModel to create header files that represent this model. Header files can be used in any IDE."""

    # Available implementations of dense layers
    IMPLEMENTATIONS = ['default', 'cmsis']
    # Available byteorder for Q7 quantization
    BYTEORDER = ["little", "big"]

    def __init__(self, extractor, implementation: str, aifestype: AifesType, destination_path: str, with_weights: bool):
        """
        Initializes the AifesCodeGenerator class with common variables
        :param extractor: Extractor class used for extraction of the AIfES Model from the corresponding framework
        :param implementation: Which implementation should be used
        :param aifestype: Which type of AIfES frontend (express, normal) should be used, is from type AifesType
        :param destination_path: Destination path for the header files, is automatically created, if not already existent
        :param with_weights: If weights should be extracted from the source model
        """

        self._extractor = extractor
        self._implementation = implementation
        self._aifestype = aifestype
        self._destination_path = destination_path
        self._aifes_model = None
        self._with_weights = with_weights

        # Check destination path, if it doesn't exist, create it
        if not os.path.isdir(self._destination_path):
            os.mkdir(self._destination_path)

        # Check parameters
        if self._implementation not in self.IMPLEMENTATIONS:
            raise ValueError("Unsupported implementation type. Got {implementation} but should be one of 'default' or"
                             " 'cmsis'".format(implementation=self._implementation))

    def run_f32(self):
        """
        Creates the header files for a f32 based implementation
        """
        # Extract the AIfES structure
        self._aifes_model = self._extractor.extractor_structure()

        # Check which type of frontend should be used and create corresponding header file
        if self._aifestype == AifesType.EXPRESS:
            self._create_aifes_express_header_fnn_f32()
        elif self._aifestype == AifesType.NORMAL:
            self._create_aifes_header_fnn_f32()
        else:
            raise ValueError("Unsupported AIfES Frontend. Should be either 'EXPRESS' or 'NORMAL'")

        # Check if weights should be extracted
        if self._with_weights:

            # Flatten the weights
            self._aifes_model.flatten_aifes_weights = flatten_weights(self._extractor.extractor_values(),
                                                                      self._extractor.get_transpose_status())

            # Write the flatten weights into the right header file depending on AIfES Frontend
            if self._aifestype == AifesType.EXPRESS:
                self._create_aifes_express_weights_fnn_f32()
            elif self._aifestype == AifesType.NORMAL:
                self._create_aifes_weights_fnn_f32()
            else:
                raise ValueError("Unsupported AIfES Frontend. Should be either 'EXPRESS' or 'NORMAL'")

    def run_q7(self, representative_data: np.ndarray, target_alignment: int, byteorder: str):
        """
        Creates the header files for a Q7 implementation. For this the weights of the given model is converted to Q7.
        """
        # Check parameter
        if byteorder not in self.BYTEORDER:
            raise ValueError("Byteorder must be either 'little' or 'big. Got {byteorder}".format(byteorder=byteorder))

        # Extract the AIfES structure
        self._aifes_model = self._extractor.extractor_structure()

        # Check dimension of representative data set and model input
        num_input = representative_data.shape[1]
        model_input = self._aifes_model.aifes_fnn_structure[0].input_shape

        if num_input != model_input:
            raise ValueError("The input dimension of the example data ({num_input}) doesn't match the input number of "
                             "the ANN ({model_input}).".format(num_input=num_input, model_input=model_input))

        # Extract weights and bias of model
        weights = self._extractor.extractor_values()

        # Convert model to quantized model
        model_converter = ModelConversion(self._aifes_model, representative_data, target_alignment, byteorder)
        q_params_layers, q_params_weights_bias = model_converter.convert_to_q7(weights)

        # Create header files depending on the chosen AIfES frontend
        if self._aifestype == AifesType.EXPRESS:
            self._create_aifes_express_header_fnn_q7()
            self._create_aifes_express_weights_fnn_q7(target_alignment, byteorder)
        elif self._aifestype == AifesType.NORMAL:
            self._create_aifes_header_fnn_q7(q_params_layers)
            self._create_aifes_weights_fnn_q7(target_alignment, byteorder)
        else:
            raise ValueError("Unsupported AIfES Frontend. Should be either 'EXPRESS' or 'NORMAL'")

    def _create_aifes_express_header_fnn_f32(self):
        """
        Creates the header file with the aifes model as aifes express function. Uses the template file aifes_e_f32_fnn.h.
        Checks the init values of alpha from Leaky ReLU and ELU for compatibility with AIfES Express.
        """
        # Create aifes structure and activation list
        aifes_fnn_structure, aifes_fnn_activations = aifes_express_create_model_structure(
            self._aifes_model.aifes_layer_count, self._aifes_model.aifes_fnn_structure)

        if self._with_weights:
            name_weights = '(void*)aifes_e_f32_flat_weights;'

        else:
            name_weights = '// Place your flattened layer weights here or give a pointer to them like: ' \
                           '(void*)aifes_e_f32_flat_weights'

        # Edit the template config file with the current net config
        checkWords = ("PLACEHOLDER_INPUTS", "PLACEHOLDER_OUTPUTS", "PLACEHOLDER_LAYERS", "PLACEHOLDER_STRUCTURE",
                      "PLACEHOLDER_ACTIVATIONS", "PLACEHOLDER_WEIGHTS")
        repWords = (str(self._aifes_model.aifes_fnn_structure[0].input_shape),
                    str(self._aifes_model.aifes_fnn_structure[-2].output_shape),
                    str(self._aifes_model.aifes_layer_count), ', '.join(map(str, aifes_fnn_structure)),
                    ', '.join(map(str, aifes_fnn_activations)), name_weights)

        f_template = open(resource_filename(__name__, "templates/aifes_express/aifes_e_f32_fnn.h"), 'r')
        f_destination = open(self._destination_path + "/aifes_e_f32_fnn.h", 'w')

        for line in f_template:
            for check, rep in zip(checkWords, repWords):
                line = line.replace(check, rep)
            f_destination.write(line)

        f_template.close()
        f_destination.close()

    def _create_aifes_express_weights_fnn_f32(self):
        """
        Creates a header file with flattened weights and bias for usage with aifes express.
        Uses template aifes_e_f32_weights
        """
        # Create Weights as string
        weights = create_c_array_str(self._aifes_model.flatten_aifes_weights)

        # Edit the template config file
        checkWords = "PLACEHOLDER_WEIGHTS"
        repWords = weights

        f_template = open(resource_filename(__name__, "templates/aifes_express/aifes_e_f32_weights.h"), 'r')
        f_destination = open(self._destination_path + "/aifes_e_f32_weights.h", 'w')

        for line in f_template:
            line = line.replace(checkWords, repWords)
            f_destination.write(line)
        f_template.close()
        f_destination.close()

    def _create_aifes_express_header_fnn_q7(self):
        """
        Creates the header file with the aifes model as aifes express function. Uses the template file aifes_e_q7_fnn.h.
        Checks the init values of alpha from Leaky ReLU and ELU for compatibility with AIfES Express.
        """
        # Create aifes structure and activation list
        aifes_fnn_structure, aifes_fnn_activations = aifes_express_create_model_structure(
            self._aifes_model.aifes_layer_count, self._aifes_model.aifes_fnn_structure)

        # Edit the template config file with the current net config
        checkWords = ("PLACEHOLDER_INPUTS", "PLACEHOLDER_OUTPUTS", "PLACEHOLDER_LAYERS", "PLACEHOLDER_STRUCTURE",
                      "PLACEHOLDER_ACTIVATIONS")

        repWords = (str(self._aifes_model.aifes_fnn_structure[0].input_shape),
                    str(self._aifes_model.aifes_fnn_structure[-2].output_shape),
                    str(self._aifes_model.aifes_layer_count), ', '.join(map(str, aifes_fnn_structure)),
                    ', '.join(map(str, aifes_fnn_activations)))

        f_template = open(resource_filename(__name__, "templates/aifes_express/aifes_e_q7_fnn.h"), 'r')
        f_destination = open(self._destination_path + "/aifes_e_q7_fnn.h", 'w')

        for line in f_template:
            for check, rep in zip(checkWords, repWords):
                line = line.replace(check, rep)
            f_destination.write(line)

        f_template.close()
        f_destination.close()

    def _create_aifes_express_weights_fnn_q7(self, alignment: int, byteorder: str):
        """
        Writes flattened weights and biases to header file. Uses template aifes_q7_weights.h.
        Expects converted q7 values!
        """

        # Create Weights as string
        weights = str_flatbuffer_c_style(self._aifes_model.flatten_aifes_weights, target_alignment=alignment,
                                         byteorder=byteorder, mutable=False)

        # Edit the template config file
        checkWords = "PLACEHOLDER_WEIGHTS"
        repWords = weights

        f_template = open(resource_filename(__name__, "templates/aifes_express/aifes_e_q7_weights.h"), 'r')
        f_destination = open(self._destination_path + "/aifes_e_q7_weights.h", 'w')

        for line in f_template:
            line = line.replace(checkWords, repWords)
            f_destination.write(line)

        f_template.close()
        f_destination.close()

    def _create_aifes_header_fnn_f32(self):
        """
        Creates header file for AIfES (non-express version). Datatype can be selected and cmsis or default
        implementation are available. Uses template aifes_f32_fnn.h. Uses dtype f32
        :param implementation: Used implementation, i.e. default or cmsis
        """
        # Create Variables for AIfES Header file
        dtype = Dtype.FLOAT32
        layer_structure = self._aifes_model.aifes_fnn_structure

        # Create aifes layer definition and init text parts
        aifes_fnn_layer_def, aifes_fnn_layer_init = aifes_create_model_structure(layer_structure, dtype, self._implementation)

        if self._with_weights:
            weights_name = 'uint32_t parameter_memory_size = aialgo_sizeof_parameter_memory(&model);\n\n  ' \
                           'aialgo_distribute_parameter_memory(&model, (void*) aifes_f32_flat_weights, parameter_memory_size);'

        else:
            weights_name = '/* Initialize your AIfES model here. You can either use a pointer to your flatten array ' \
                           'like so: \n  uint32_t parameter_memory_size = aialgo_sizeof_parameter_memory(&model);\n\n  ' \
                           'aialgo_distribute_parameter_memory(&model, (void*) aifes_f32_flat_weights, parameter_memory_size);\n\n' \
                           '  Or you define the weights and biases per layer. For this you can update the layer initialization, e.g.:\n\n' \
                           '  // Use constant data only for inference. For training remove the const qualifier!!\n  ' \
                           'const float weights_data_dense[] = {-10.1164f, 7.297f, -8.4212f, -7.6482f, 5.4396f, ' \
                           '-9.0155f};\n  const float bias_data_dense[] = {-2.9653f,  2.3677f, -1.5968f};\n  ' \
                           'ailayer_dense_f32_t dense_layer_1 = AILAYER_DENSE_F32_M(3, weights_data_dense, bias_data_dense);\n\n' \
                           '  Alternatively you can set the weights and biases of the layers directly.' \
                           '\n  see https://create.arduino.cc/projecthub/aifes_team/aifes-inference-tutorial-f44d96 for more details*/'

        if self._implementation == 'cmsis':
            cmsis_include = '\n#include <aifes_cmsis.h>'

        elif self._implementation == 'default':
            cmsis_include = ''
        else:
            raise ValueError("Undefined implementation of {}. Must be either 'cmsis' or default".format(implementation))

        # Edit the template config file with the current net config
        checkWords = ("PLACEHOLDER_INPUTS", "PLACEHOLDER_OUTPUTS", "PLACEHOLDER_LAYER_DEF", "PLACEHOLDER_LAYER_INIT",
                      "PLACEHOLDER_WEIGHTS", "PLACEHOLDER_CMSIS_INCLUDE")
        repWords = (str(self._aifes_model.aifes_fnn_structure[0].input_shape),
                    str(self._aifes_model.aifes_fnn_structure[-2].output_shape),
                    aifes_fnn_layer_def,
                    aifes_fnn_layer_init,
                    weights_name,
                    cmsis_include)

        f_template = open(resource_filename(__name__, "templates/aifes/aifes_f32_fnn.h"), 'r')
        f_destination = open(self._destination_path + "/aifes_f32_fnn.h", 'w')

        for line in f_template:
            for check, rep in zip(checkWords, repWords):
                line = line.replace(check, rep)
            f_destination.write(line)

        f_template.close()
        f_destination.close()

    def _create_aifes_weights_fnn_f32(self):
        """
        Writes flattened weights and biases to header file. Uses template aifes_f32_weights.h
        """

        # Create Weights as string
        weights = create_c_array_str(self._aifes_model.flatten_aifes_weights)

        # Edit the template config file
        checkWords = "PLACEHOLDER_WEIGHTS"
        repWords = weights

        f_template = open(resource_filename(__name__, "templates/aifes/aifes_f32_weights.h"), 'r')
        f_destination = open(self._destination_path + "/aifes_f32_weights.h", 'w')

        for line in f_template:
            line = line.replace(checkWords, repWords)
            f_destination.write(line)

        f_template.close()
        f_destination.close()


    def _create_aifes_header_fnn_q7(self, q_params_layers: list):
        """
        Creates header file for AIfES (non-express version). Uses template aifes_q7_fnn.h.
        Uses dtype Q7.
        :param dtype: Datatype of model
        :param implementation: Used implementation, i.e. default or cmsis
        """
        # Create Variables for AIfES Header file
        layer_structure = self._aifes_model.aifes_fnn_structure
        dtype = Dtype.Q7
        dtype_str = dtype_to_aifes[dtype]

        # Create aifes layer definition and init text parts
        aifes_fnn_layer_def, aifes_fnn_layer_init = aifes_create_model_structure(layer_structure, dtype, self._implementation)

        if self._implementation == 'cmsis':
            cmsis_include = '\n#include <aifes_cmsis.h>'

        elif self._implementation == 'default':
            cmsis_include = ''
        else:
            raise ValueError("Undefined implementation of {}. Must be either 'cmsis' or default".format(self._implementation))

        # Edit the template config file with the current net config
        checkWords = ("PLACEHOLDER_INPUTS", "PLACEHOLDER_OUTPUTS", "PLACEHOLDER_LAYER_DEF", "PLACEHOLDER_LAYER_INIT",
                      "PLACEHOLDER_INPUT_SHIFT", "PLACEHOLDER_INPUT_ZERO", "PLACEHOLDER_OUTPUT_SHIFT",
                      "PLACEHOLDER_OUTPUT_ZERO", "PLACEHOLDER_CMSIS_INCLUDE")
        repWords = (str(self._aifes_model.aifes_fnn_structure[0].input_shape),
                    str(self._aifes_model.aifes_fnn_structure[-2].output_shape),
                    aifes_fnn_layer_def,
                    aifes_fnn_layer_init,
                    str(q_params_layers[0][0]), str(q_params_layers[0][1]),
                    str(q_params_layers[-1][0]), str(q_params_layers[-1][1]),
                    cmsis_include)

        f_template = open(resource_filename(__name__, "templates/aifes/aifes_q7_fnn.h"), 'r')
        f_destination = open(self._destination_path + "/aifes_q7_fnn.h".format(DTYPE=dtype_str), 'w')

        for line in f_template:
            for check, rep in zip(checkWords, repWords):
                line = line.replace(check, rep)
            f_destination.write(line)

        f_template.close()
        f_destination.close()

    def _create_aifes_weights_fnn_q7(self, alignment: int, byteorder: str):
        """
        Writes flattened weights and biases to header file. Uses template aifes_q7_weights.h
        Uses dtype Q7. Needs converted weights to q7!
        """

        # Create Weights as string
        weights = str_flatbuffer_c_style(self._aifes_model.flatten_aifes_weights, target_alignment=alignment,
                                         byteorder=byteorder, mutable=False)

        # Edit the template config file
        checkWords = "PLACEHOLDER_WEIGHTS"
        repWords = weights

        f_template = open(resource_filename(__name__, "templates/aifes/aifes_q7_weights.h"), 'r')
        f_destination = open(self._destination_path + "/aifes_q7_weights.h", 'w')

        for line in f_template:
            line = line.replace(checkWords, repWords)
            f_destination.write(line)

        f_template.close()
        f_destination.close()
