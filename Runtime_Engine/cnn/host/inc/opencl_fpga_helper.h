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

#ifndef __COMMON_H__
#define __COMMON_H__

#include "includes.h"

// Name of the compiled binary
#if defined RESNET50
#define COMPILED_BINARY "resnet50/cnn"
#elif defined RESNET50_PRUNED
#define COMPILED_BINARY "resnet50_pruned/cnn"
#else
#define COMPILED_BINARY "googlenet/cnn"
#endif

#define STRING_BUFFER_LEN 1024

void DeviceInfoUlong(cl_device_id device, cl_device_info param, const char* name);
void DeviceInfoUint(cl_device_id device, cl_device_info param, const char* name);
void DeviceInfoBool(cl_device_id device, cl_device_info param, const char* name);
void DeviceInfoString(cl_device_id device, cl_device_info param, const char* name);
void DisplayDeviceInfo(cl_device_id device);

#endif
