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
import numpy as np
from ..support.aifes_model import Layer


def calc_q_params_q7(min_value, max_value):
    """Calculate quantization parameters for values of the given range

    """
    
    if max_value - min_value < 0:
        raise Exception('min_value has to be smaller than max_value.')
    elif max_value == 0 and min_value == 0:
        shift = 0
        zero_point = 0
        return (shift, zero_point)
    
    # An interval that does not include the zero has to be handled special
    if min_value > 0 and max_value > 0:
        min_value = 0
    elif min_value < 0 and max_value < 0:
        max_value = 0
        

    min_target = -128
    target_interval_bitlen = 8
    interval_old = max_value - min_value

    value_interval_bitlen = -24
    while (2 ** value_interval_bitlen) <= interval_old:
        value_interval_bitlen += 1
        
    interval_new = 2 ** value_interval_bitlen
    
    min_new = min_value - (interval_new - interval_old) / 2.0
    
    if target_interval_bitlen - value_interval_bitlen < 0:
        raise Exception('One or more values are too big to quantize them to a 8 bit integer.')
    
    shift = int(target_interval_bitlen - value_interval_bitlen)
    zero_point = int(np.round(-min_new * (1 << shift)) + min_target)
    
    return (shift, zero_point)
    
def quantize_tensor_q7(tensor, q_params):
    """Quantize the tensor to Q7 representation with the given quantization parameters (shift, zero_point)

    """
    
    tensor_q7 = np.round(tensor * (1 << q_params[0]) + q_params[1]).astype(np.int32)
    
    return tensor_q7

def quantize_model_q7(layers, weights, representative_dataset, act_params=[]):
    """Quantize the model to the Q7 representation

    Arguments:
    layers -- List of the layers (type Layer Enum)
    weights -- F32 weights
    representative_dataset -- A dataset that is representative for the whole training dataset (To calculate min and max values)
    act_params -- A list containing parameters of the activation functions (e.g. the Leaky ReLU)
    """
    
    intermediate_q_params = [calc_q_params_q7(1.1*np.min(representative_dataset), 1.1*np.max(representative_dataset))]
    result_q_params = [intermediate_q_params[-1]] # The quantization params that are in the parameter memory (rest is in inference / training memory, because it is not configurable by the user)
    weights_q7 = []
    weights_q_params = []
    
    dense_idx = 0
    act_param_idx = 0
    x = representative_dataset
    for layer in layers:
        if layer == Layer.DENSE or layer == Layer.DENSE_WT:
            y = x @ weights[2*dense_idx] + weights[2*dense_idx+1]
            
            w_q_params = calc_q_params_q7(-np.max(np.abs(weights[2*dense_idx])), np.max(np.abs(weights[2*dense_idx])))
            b_q_params = (w_q_params[0] + intermediate_q_params[-1][0], 0)
            
            intermediate_q_params.append(calc_q_params_q7(1.1*np.min(y), 1.1*np.max(y)))
            result_q_params.append(intermediate_q_params[-1])
            
            weights_q7.append(quantize_tensor_q7((weights[2*dense_idx].T if layer == Layer.DENSE_WT else weights[2*dense_idx]), w_q_params).astype(np.int8))
            weights_q_params.append(w_q_params)
            
            weights_q7.append(quantize_tensor_q7(weights[2*dense_idx+1], b_q_params).astype(np.int32))
            weights_q_params.append(b_q_params)
            
            dense_idx += 1
        elif layer == Layer.SIGMOID:
            y = 1.0 / (1.0 + np.exp(-x))
            intermediate_q_params.append((8, -2**7))
        elif layer == Layer.SOFTMAX:
            y = np.exp(y-np.max(y)) / np.sum(np.exp(y-np.max(y)))
            intermediate_q_params.append((8, -2**7))
        elif layer == Layer.TANH:
            y = np.tanh(x)
            intermediate_q_params.append((7, 0))
        elif layer == Layer.SOFTSIGN:
            y = x / (1 + np.abs(x))
            intermediate_q_params.append((7, 0))
        elif layer == Layer.LEAKY_RELU:
            alpha = act_params[act_param_idx]
            y = np.where(x > 0, x, x * alpha)
            intermediate_q_params.append(intermediate_q_params[-1])
            act_param_idx += 1
        elif layer == Layer.RELU:
            y = np.where(x > 0, x, 0.0)   
            intermediate_q_params.append(intermediate_q_params[-1])
        elif layer == Layer.ELU:
            alpha = act_params[act_param_idx]
            y = np.where(x > 0, x, alpha * (np.exp(x) - 1.0))
            intermediate_q_params.append(intermediate_q_params[-1])
            act_param_idx += 1
        elif layer == Layer.INPUT:
            y = x
        x = y
        
    return result_q_params, weights_q_params, weights_q7


def pad_buffer(buffer, target_alignment):
    """Add zeros to the buffer according to the target alignment

    """
    
    while len(buffer) % target_alignment != 0:
        buffer += b'\x00'
    return buffer


def q_params_q7_to_bytes(q_params, target_alignment, byteorder='little'):
    """Converts the given Q7 quantization parameters to a byte array

    Arguments:
    q_params -- (shift, zero_point)
    target_alignment -- The alignment of the structs on the target achitecture. Must be the same as configured in the AIFES_MEMORY_ALIGNMENT macro in aifes.
    byteorder -- "little" for little endian; "big" for big endian. Has to match to the byte order of the target architecture.
    """
    
    buffer = b''
    buffer += q_params[0].to_bytes(2, byteorder)              # shift uint16
    buffer += q_params[1].to_bytes(1, byteorder, signed=True) # zero_point int8
    buffer = pad_buffer(buffer, target_alignment)             # padding
    return buffer


def q_params_q31_to_bytes(q_params, target_alignment, byteorder='little'):
    """Converts the given Q31 quantization parameters to a byte array

    Arguments:
    q_params -- (shift, zero_point)
    target_alignment -- The alignment of the structs on the target achitecture. Must be the same as configured in the AIFES_MEMORY_ALIGNMENT macro in aifes.
    byteorder -- "little" for little endian; "big" for big endian. Has to match to the byte order of the target architecture.
    """
    buffer = b''
    buffer += q_params[0].to_bytes(2, byteorder)              # shift uint16
    buffer += q_params[1].to_bytes(4, byteorder, signed=True) # zero_point int32
    buffer = pad_buffer(buffer, target_alignment)             # padding
    return buffer


def create_flatbuffer_q7(result_q_params, weights_q_params, weights_q7, target_alignment, byteorder='little'):
    """Creats a byte array, containing all given model parameters like quantization parameters and quantized weights.

    Arguments:
    result_q_params -- Quantization parameter tuples for the layer results
    weights_q_params -- Quantization parameter tuples for the weights and biases
    weights_q7 -- Weights and biases as a list of numpy arrays
    target_alignment -- The alignment of the arrays and structs on the target achitecture. Must be the same as configured in the AIFES_MEMORY_ALIGNMENT macro in aifes.
    byteorder -- "little" for little endian; "big" for big endian. Has to match to the byte order of the target architecture.
    """
    
    flatbuffer = b''
    
    for res_params in result_q_params:
        flatbuffer += q_params_q7_to_bytes(res_params, target_alignment, byteorder)
    
    for w_q_params, w in zip(weights_q_params, weights_q7):
        if w.dtype == np.int8:
            flatbuffer += q_params_q7_to_bytes(w_q_params, target_alignment, byteorder)
        elif w.dtype == np.int32:
            flatbuffer += q_params_q31_to_bytes(w_q_params, target_alignment, byteorder)
        if byteorder == 'big':
            # Switch to big endian stype
            flatbuffer += w.byteswap().tobytes()
        else:
            flatbuffer += w.tobytes()
        flatbuffer = pad_buffer(flatbuffer, target_alignment)     # padding to match alignment
    
    return flatbuffer
    
    
def create_flatbuffer_f32(weights_f32):
    """Creats a byte array for F32 models, containing all given weights.

    Arguments:
    weights_f32 -- Weights and biases as a list of numpy arrays
    """
    
    flatbuffer = b''
    for w in weights_f32:
        flatbuffer += w.tobytes()
    
    return flatbuffer


def str_flatbuffer_c_style(flatbuffer, target_alignment=4, mutable=True, elements_per_line=-1, byteorder='little') -> str:
    """Print the given flatbuffer to the console for easy copy into your code.

    Arguments:
    flatbuffer -- A byte array containing the model parameters
    target_alignment -- The alignment of the structs on the target achitecture. Must be the same as configured in the AIFES_MEMORY_ALIGNMENT macro in aifes.
    mutable -- False if the parameters will not be changed afterwards. (For example if you want to do only inferences.)
    elements_per_line -- Number of array elements that are printed per line
    byteorder -- "little" for little endian; "big" for big endian. Has to match to the byte order of the target architecture.
    """

    if elements_per_line == -1:
        elements_per_line = int(16 / target_alignment) + 4
    
    pad_buffer(flatbuffer, target_alignment)
    
    # Turn byte order for little or big endian
    flatbuffer_turned = []
    if byteorder == 'little':
        for i in range(int(len(flatbuffer)/target_alignment)):
            buffer = []
            for j in range(target_alignment):
                buffer.append(int(flatbuffer[(i+1)*target_alignment - j - 1]))
            flatbuffer_turned.extend(buffer)
    else:
        for byte in flatbuffer:
            flatbuffer_turned.append(int(byte))
    
    out_string = "const uint32_t parameter_memory_size = {};\n".format(len(flatbuffer_turned))
    
    if not mutable:
        out_string += "const "
    
    count = 0
    if target_alignment == 1:
        out_string += "uint8_t"
    elif target_alignment == 2:
        out_string += "uint16_t"
    elif target_alignment == 4:
        out_string += "uint32_t"
    else:
        raise Exception('Only a target_alignment of 1, 2 or 4 is supported.')
        
    out_string += " model_parameters[" + str(int(len(flatbuffer_turned) / target_alignment)) + "] = {\n    "
    out_string += "0x"
    for byte in flatbuffer_turned:
        if count != 0 and count % target_alignment == 0:
            out_string += ", "
            if int(count / target_alignment) % elements_per_line == 0:
                out_string += "\n"
                out_string += "    0x"
            else:
                out_string += "0x"
        out_string += "{:02X}".format(byte)
        count += 1
    out_string += "\n};\n"

    return out_string