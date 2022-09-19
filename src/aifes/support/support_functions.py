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
from collections.abc import Iterable
import numpy as np


def flatten_weights(weights: np.ndarray, transpose=False):
    """Flatten weights from array to flat weights"""

    if transpose:
        for i in range(len(weights)):
            weights[i] = weights[i].transpose()

    w = []
    for l in weights:
        if isinstance(l, Iterable):
            w = w + flatten_weights(l)
        else:
            w = w + [l]
    return w


def create_c_array_str(input_list: list, num_elements_line=5) -> str:

    num_parameters = len(input_list)
    output_str = ""

    for i in range(num_parameters):
        output_str += "{}f, ".format(input_list[i])
        if i%num_elements_line == 0 and i > 0:
            output_str += "\n\t\t"

    return output_str
