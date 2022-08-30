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

def get_layer_list(model):
    """
    This function returns a list of all layers present in a model (Submodules will be expanded).
    :param model: The model of which the layers should be extracted.
    :return: A list containing the model layers.
    """
    children = [child for child in model.children()]
    layer_list = []
    if not children:
        # if model has no children the model itself is the last child
        return model
    else:
        # Recursively look for children from children until the last child is reached
        for child in children:
            try:
                layer_list.extend(get_layer_list(child))
            except TypeError:
                layer_list.append(get_layer_list(child))
    return layer_list
