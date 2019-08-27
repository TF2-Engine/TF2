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

#ifndef __CNN_H__
#define __CNN_H__

#include "archs.h"
#include "defines.h"
#include "types.h"

#if defined RESNET50
#include "cnn_resnet50.h"
#elif defined RESNET50_PRUNED
#include "cnn_resnet50_pruned.h"
#else
#include "cnn_googlenet.h"
#endif

#endif // __CNN_H__
