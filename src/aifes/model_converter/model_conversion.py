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
from ..support.aifes_model import AifesModel, configurable_act_layer
from .support_model_conversion_q7 import *


class ModelConversion:
    def __init__(self, aifes_model: AifesModel, representative_dataset: np.ndarray,
                 alignment: int, byteorder: str):
        """
        Initializes the class with common variables.
        :param aifes_model: AIfES model which should be converted
        :param representative_dataset: representative dataset for conversion of the ANN
        :param alignment: Alignment of target architecture in bytes, i.e. 2 for AVR Arduino (16 bit MCU),
        4 for ARM Cortex (32 Bit MCU)der: Byteorder of target a
        :param byteorrchitecture, i.e. 'little' for little endian and 'big' for big endian
        """
        self._aifes_model = aifes_model
        self._representative_dataset = representative_dataset
        self._alignment = alignment
        self._byteorder = byteorder

    def convert_to_q7(self, weights):
        """
        Converts the given weights to a flatbuffer in q7 style. Uses AIfES style of Q7 implementation.
        :param self:
        :param weights: Weights of the model, which should be converted. Can be extrated from Keras model by using keras_extractor_values
        :returns: Returns resulting Q parameters for layers and weights/bias
        """
        # Representation of the model for AIfES pytools
        layers = []
        act_params = []  # Append additional parameters

        for el in self._aifes_model.aifes_fnn_structure:
            layers.append(el.layer_type)
            if el.layer_type in configurable_act_layer:
                act_params.append(el.alpha_value)

        result_q_params, weights_q_params, weights_q7 = quantize_model_q7(layers, weights, self._representative_dataset,
                                                                          act_params=act_params)

        flatbuffer_q7 = create_flatbuffer_q7(result_q_params, weights_q_params, weights_q7,
                                             target_alignment=self._alignment, byteorder=self._byteorder)

        self._aifes_model.flatten_aifes_weights = flatbuffer_q7

        return result_q_params, weights_q_params
