/* Copyright 2019 Inspur Corporation. All Rights Reserved.

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

// Top level OpenCL code

#pragma OPENCL EXTENSION cl_intel_channels : enable

#include "cycle.cl"

#include "sequencer.cl"

#include "input_reader.cl"

#include "filter_reader.cl"

#include "retriever.cl"

#include "pe.cl"

#include "relu.cl"

#include "pool.cl"

#include "full_size_pool.cl"

#include "pool_tail.cl"

#include "feature_writer.cl"
