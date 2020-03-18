
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
#include "tf2_auto_param.h"

int main(int argc, char **argv)
{
  std::string filename = argv[1];
  std::string netname = argv[2];

  // module_framework_data module_framework;

  module_framework_data* module_framework= (module_framework_data*)calloc(1,sizeof(module_framework_data));
  
  // module_framework = FrameParseStore(filename, netname);

  FrameParseStore(filename, netname, module_framework);

  ParamGeneration(module_framework, netname);

  free(module_framework);

  return true;
}