/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <TensorFlowLite.h>

#include "constants.h"
#include "main_functions.h"
#include "model.h"
#include "output_handler.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

constexpr int kTensorArenaSize = 100 * 1024;
// Keep aligned to 16 bytes for CMSIS
alignas(16) uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {
  tflite::InitializeTarget();
  Serial.begin(9600);               // initialize serial communication at 9600 bits per second:
  Serial1.begin(9600);

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::AllOpsResolver resolver;

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);
}

void model_inference(float input_data[], int input_size) {
  for (int i=0; i<input_size; i++) {
    input->data.f[i] = input_data[i];
  }
  
  // Run inference, and report any error
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    // MicroPrintf("Invoke failed on x: %f\n", static_cast<double>(x));
    MicroPrintf("Invoke failed on x: \n");
    return;
  }

  float y0 = output->data.f[0]; 
  // float y1 = output->data.f[1];
  // float y2 = output->data.f[2];

  transmit_output(y0);
}

void transmit_output(float y) {
  if (y < 0.5) {
    Serial.print(String("0 "));
    Serial1.println('0');
    Serial.println(String(y));
  }
  else {
    Serial.print(String("1 "));
    Serial1.println('1');
    Serial.println(String(y));
  }
}

// The name of this function is important for Arduino compatibility.
void loop() {
  unsigned int input_size = 32;

  float t_signal[] = {-0.02745005488395691, -0.07137013971805573, -0.08235016465187073, 0.796051561832428, 
                      0.043920088559389114, 0.021960044279694557, -0.06039011850953102, -0.18117035925388336,
                      -0.15921030938625336, -0.18117035925388336, -0.19215038418769836, -0.07137013971805573,
                      -0.05490010976791382, -0.11529023200273514, -0.13176026940345764, -0.04941009730100632,
                      -0.043920088559389114, 1.0815321207046509, -0.11529023200273514, -0.19764038920402527,
                      -0.18666036427021027, -0.12078023701906204, -0.06039011850953102, -0.08235016465187073,
                      0.06039011850953102, 0.11529023200273514, -0.005490011069923639, -0.03843007609248161,
                      -0.04941009730100632, -0.06588013470172882, 0.005490011069923639, -0.09882019460201263};

  // float s_signal[] = {-0.009880007, 0.059280045, 0.024700018, -0.0049400036, -0.029640023, -0.044460032, -0.034580026,
  //                     -0.034580026, -0.049400035, -0.049400035, -0.09880007, 1.0571607, 0.16302012, -0.029640023,
  //                     -0.074100055, -0.08398006, -0.088920064, -0.009880007, 0.07904006, 0.09880007, 0.15314011,
  //                     0.18278013, 0.10868008, 0.049400035, 0.029640023, 0.0, 0.014820011, 0.009880007, -0.014820011,
  //                     -0.08398006, -0.034580026, -0.009880007};

  float n_signal[] = {-0.06422005, -0.029640023, -0.049400035, -0.019760014, -0.044460032, -0.044460032, -0.074100055, 
                      -0.088920064, -0.05434004, -0.03952003, -0.044460032, -0.03952003, -0.20254014, -0.7311205, 
                      -0.15808012, -0.05434004, -0.014820011, -0.009880007, 0.049400035, 0.1284401, 0.2766402, 0.29146022,
                      0.15808012, -0.059280045, -0.088920064, -0.024700018, 0.0049400036, 0.034580026, 0.024700018, 0.014820011, 0.9534207, -0.6471405};


  model_inference(n_signal, input_size);
  delay(1000);
  // model_inference(s_signal, input_size);
  // delay(1000);
  model_inference(t_signal, input_size);
  delay(1000);
}
