/*
  www.aifes.ai
  https://github.com/Fraunhofer-IMS/AIfES_for_Arduino

  You can find more AIfES tutorials here:
  https://create.arduino.cc/projecthub/aifes_team

  AIfES: Configuration file automatically generated from AIfES-Converter
  https://github.com/Fraunhofer-IMS/AIfES-Converter

  You first need to initialize the model once with:
  ---------------------------------------------------------------------------
    uint8_t error;
    error = aifes_q7_fnn_create_model();
    if (error == 1)
    {
        //do something for error handling
        //e.g. while(true)
    }
  ---------------------------------------------------------------------------
  Please check the error. It is 1 if something inside the initialization failed. Check the output

  Paste the following code into your project and insert your inputs/features:
  ---------------------------------------------------------------------------
    float input_data[PLACEHOLDER_INPUTS]; // AIfES input data
    float output_data[PLACEHOLDER_OUTPUTS]; // AIfES output data

    aifes_q7_fnn_inference((float*) input_data, (float*) output_data);
  ---------------------------------------------------------------------------

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
*/

#include <aifes.h>PLACEHOLDER_CMSIS_INCLUDE
#include "aifes_q7_weights.h"

#ifndef AIFES_Q7_FNN
#define AIFES_Q7_FNN

#define DATASETS 1
#define INPUTS  PLACEHOLDER_INPUTS
#define OUTPUTS PLACEHOLDER_OUTPUTS

aimodel_t model;

// Layer definition
uint16_t input_layer_shape[] = {DATASETS, INPUTS};

PLACEHOLDER_LAYER_DEF

uint16_t output_layer_shape[] = {DATASETS, OUTPUTS};

uint8_t aifes_q7_fnn_create_model()
{
  ailayer_t *x;

  PLACEHOLDER_LAYER_INIT

  aialgo_compile_model(&model); // Compile the AIfES model


  aialgo_distribute_parameter_memory(&model, (void*) model_parameters, parameter_memory_size);

  uint32_t memory_size = aialgo_sizeof_inference_memory(&model);

  // Comment the following lines after the first run of AIfES, when you use the option with an array instead of malloc
  void *memory_ptr = malloc(memory_size);

  if(memory_ptr == NULL)
  {
    aiprint("Not enough memory available for inference of the neural network.");
    return 1;
  }

  aiprint("AIfES needs an inference memory of:");
  aiprint_uint("%u", memory_size);
  aiprint("\nYou can use this size and create an array to avoid using malloc.\n");

  // Uncomment the following line and add the output of the previous printf into SIZE_OF_PRINTF. Then comment the lines
  // above
  // uint8_t memory_array[SIZE_OF_PRINTF];
  // uint8_t *memory_ptr= memory_array


  aialgo_schedule_inference_memory(&model, memory_ptr, memory_size);
  return 0;
}

void aifes_q7_fnn_inference(float* input_data, float* output_data)
{

  // F32 tensor that has to be quantized
  aitensor_t in_f32 = AITENSOR_2D_F32(input_layer_shape, input_data);

  // Automatically quantized Q7 tensor to feed into the neural network
  aimath_q7_params_t in_q_params = { .shift = PLACEHOLDER_INPUT_SHIFT, .zero_point = PLACEHOLDER_INPUT_ZERO};
  int8_t in_data[PLACEHOLDER_INPUTS];
  aitensor_t in = AITENSOR_2D_Q7(input_layer_shape, &in_q_params, in_data);

  aimath_q7_quantize_tensor_from_f32(&in_f32, &in); // Quantize the F32 tensor and write the results to the Q7 tensor

  // Definition of the output shape and tensor
  aimath_q7_params_t out_q_params;
  int8_t out_data[PLACEHOLDER_OUTPUTS];
  aitensor_t output_tensor = AITENSOR_2D_Q7(output_layer_shape, &out_q_params, out_data);

  aialgo_inference_model(&model, &in, &output_tensor);

  // Convert output tensor of Q7 to float
  for (uint32_t i=0; i < PLACEHOLDER_OUTPUTS; i++)
  {
    output_data[i] = Q7_TO_FLOAT(out_data[i], out_q_params.shift, out_q_params.zero_point);
  }


}

#endif // AIFES_F32_FNN