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
import math
from ..support.aifes_model import AifesLayer, Layer, Dtype, dtype_to_aifes, act_layer, configurable_act_layer
from ..model_converter.support_model_conversion_q7 import calc_q_params_q7


def aifes_create_model_structure(layer_structure: List[AifesLayer], dtype: Dtype, implementation: str):
    """
    Creates the model structure for non-express AIfES versions. Creates the init and definition text blocks for
    the placeholder 'PLACEHOLDER_INIT' and 'PLACEHOLDER_DEF'
    :param layer_structure: AIfES Layer structure of the current model; is part of AifesModel.aifes_fnn_structure class
    and filled by the extractor
    :param dtype: Current Dtype type, e.g. Dtype.FLOAT32, Dtype.Q7. Is used for init and definition in template files
    :param implementation: Choosen implementation for dense layers, i.e. 'default' or 'cmsis'
    :returns: Returns the textblocks for 'PLACEHODER_DEF' (aifes_fnn_layer_def) and 'PLACEHOLDER_INIT'
    (aifes_fnn_layer_init)
    """
    # Define local variables
    # Set dytpe to corret string
    dtype_str = dtype_to_aifes[dtype]

    # Define input layer
    aifes_fnn_layer_def = ("ailayer_input_{dtype}_t" + "\t\t" +
                          "input_layer\t= " + layer_structure[0].init_macro + "\n").format(dtype=dtype_str,
                                                                                           DTYPE_C=dtype_str.upper())
    # Define first dense layer after input layer
    aifes_fnn_layer_def += ("ailayer_dense_{dtype}_t" + "\t\t" + "dense_layer_1\t= " +
                            layer_structure[1].init_macro + "\n").format(dtype=dtype_str, DTYPE_C=dtype_str.upper())

    # Init input layer
    aifes_fnn_layer_init = f"model.input_layer = ailayer_input_{dtype_str}_default(&input_layer);\n"
    # Init first dense layer after input layer
    # Extract layer name, i.e. Dense
    layer_init = layer_structure[1].layer_name
    # If layer ist transposed, corresponding string needs to be added
    if layer_structure[1].layer_type == Layer.DENSE_WT:
        layer_init += "_wt"
    # Init first dense layer
    aifes_fnn_layer_init += (f"  x = ailayer_{layer_init}_{dtype_str}_{implementation}(&dense_layer_1" +
                             ", model.input_layer);\n")

    # Create model structure for definition and init for remaining layers
    counter = 1  # Counter to name the dense layers and activation layers according to their position in the ANN
    prev_layer_dense = False
    for i in range(2, len(layer_structure)):
        el = layer_structure[i]
        if el.layer_type == Layer.INPUT:
            # Should never happen
            raise ValueError("Input layer after initial layer. Something has gone wrong.")
        if el.layer_type == Layer.LINEAR:
            # Linear layer is pass through of values, therefore no additional layer is needed
            continue

        # Check if previous layer and current layer are dense, then we need to increase the counter, as linear
        # activation layer was in between
        if prev_layer_dense and el.layer_type not in act_layer:
            counter += 1

        # Create layer name
        layer_name = "{name}_layer_{num}".format(name=el.layer_name, num=str(counter))
        # Create definition of layer
        aifes_fnn_layer_def += create_definition_of_layer(dtype, el, layer_name)
        # Create init of layer depending on type and implementation, i.e. default or cmsis
        if el.layer_type in act_layer:
            implementation_loc = "default"
            counter += 1
            prev_layer_dense = False
        else:
            implementation_loc = implementation
            prev_layer_dense = True

        # Set layer name to init layer
        layer_init = el.layer_name
        # If transposed corresponding string needs to be added
        if el.layer_type == Layer.DENSE_WT:
            layer_init += "_wt"
        # Set init of layer
        aifes_fnn_layer_init += (f"  x = ailayer_{layer_init}_{dtype_str}_{implementation_loc}(&" + layer_name + ", x);"
                                                                                                                 "\n")

    # Set output of model
    aifes_fnn_layer_init += "  model.output_layer = x;"

    return aifes_fnn_layer_def, aifes_fnn_layer_init


def create_definition_of_layer(dtype: Dtype, curr_layer: AifesLayer, layer_name: str):
    """
    Creates definition string of given layer for use in template file
    :param dtype: Current Dtype, e.g. Dtype.FLOAT32, Dtype.Q7
    :param curr_layer: Given Layer for which the definition should be created
    :param layer_name: Layer name of given layer in ANN
    :return: Definition string of given layer
    """
    # Create dtype string
    dtype_str = dtype_to_aifes[dtype]

    # If current layer is configurable, the macro for definition needs to be set to that value if quantization is active
    if curr_layer.layer_type in configurable_act_layer:
        # When quantization is active, the alpha value needs to be converted
        if dtype == Dtype.Q7:
            alpha = curr_layer.alpha_value
            if alpha > 0:
                max = alpha
                min = 0
            else:
                max = 0
                min = alpha
            try:
                (shift, zero_point) = calc_q_params_q7(min, max)
            except:
                raise ValueError(f"During quantization of the alpha value {alpha} for the activation function "
                                 f"{curr_layer.layer_name} an error occurred. Please adjust the alpha value so it can "
                                 f"fit within Q7")

            # Initialize place holder strings for init macro
            q_start_init = "AISCALAR_Q7("
            q_stop_init = "," + str(shift) + "," + str(zero_point) + ")"
        else:
            # No quantization, no place holder needed
            q_start_init = ""
            q_stop_init = ""

        aifes_fnn_layer_def = ("ailayer_{name}_{dtype}_t" + "\t\t" + layer_name + "\t= " + curr_layer.init_macro +
                                "\n").format(name=curr_layer.layer_name, dtype=dtype_str, DTYPE_C=dtype_str.upper(),
                                             Q_START_INIT=q_start_init, Q_STOP_INIT=q_stop_init)
    else:
        # No configurable activation layer, so no need to set parameters
        aifes_fnn_layer_def = ("ailayer_{name}_{dtype}_t" + "\t\t" + layer_name + "\t= " + curr_layer.init_macro +
                                "\n").format(name=curr_layer.layer_name, dtype=dtype_str, DTYPE_C=dtype_str.upper())

    return aifes_fnn_layer_def


def aifes_express_create_model_structure(aifes_layer_count: int, layer_structure: List[AifesLayer]):
    """
    Creates the model structure for AIfES express as a text block for use in template file
    :param aifes_layer_count: Number of AIfES Layers, automatically calculated by keras_extractor_structure
    :param layer_structure: Layer structure of current ANN, automatically calculated by keras_extractor_structure
    :returns: Needed text blocks for use in template, replaces 'PLACEHOLDER_STRUCTURE' (aifes_fnn_structure) and
    'PLACEHOLDER_ACTIVATIONS' (aifes_fnn_activations)
    """
    # Create Variables for AIfES Express Header file
    aifes_fnn_structure = [0] * aifes_layer_count
    aifes_fnn_activations = ['AIfES_E_'] * (aifes_layer_count - 1)

    # Set Input layer shape
    aifes_fnn_structure[0] = layer_structure[0].input_shape

    # Set the following layers to correct shape and activation, going through the layers with corresponding dense and
    # activation layer
    for i in range(1, aifes_layer_count):
        # Get corresponding dense layer
        if layer_structure[2 * i - 1].layer_type in [Layer.DENSE, Layer.DENSE_WT]:
            aifes_fnn_structure[i] = layer_structure[2 * i - 1].input_shape
        else:
            raise ValueError("Layer " + str(i) + " contains no valid dense layer")
        # Get the corresponding actiavtion layer
        if layer_structure[2 * i].layer_type in act_layer:
            # Check if activation layer is leaky ReLU and check the corresponding alpha value
            if layer_structure[2 * i].layer_type == Layer.LEAKY_RELU:
                if not math.isclose(layer_structure[2 * i].alpha_value, 0.01, rel_tol=0.01):
                    raise ValueError("Alpha value of layer {i} for Leaky Relu isn't default value of 0.01 but "
                                     "{alpha}. Please change the value to 0.01!".
                                     format(i=i, alpha=layer_structure[2 * i].alpha_value))
            # Check if activation layer is ELU and check the corresponding alpha value
            if layer_structure[2 * i].layer_type == Layer.ELU:
                if not math.isclose(layer_structure[2 * i].alpha_value, 1.0, rel_tol=0.01):
                    raise ValueError(
                        "Alpha value of layer {i} for ELU isn't default value of 1.0 but {alpha}. Please change "
                        "the value to 1.0!".format(i=i, alpha=layer_structure[2 * i].alpha_value))
            # If everything is alright, add activation layer to activation structure
            aifes_fnn_activations[i - 1] += layer_structure[2 * i].layer_name
        else:
            raise ValueError("Layer " + str(i) + " contains no valid activation function")

    return aifes_fnn_structure, aifes_fnn_activations
