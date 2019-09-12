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

#ifndef __DEBUG_H__
#define __DEBUG_H__

#include <stdio.h>

#define PRINTF(fmt, ...) printf("%s:%d: " fmt, __FILE__, __LINE__, ##__VA_ARGS__);

extern int common_printf(FILE * fp, const int level, const char * file, const int line, 
                         const char * func, const char * fmt, ...);

#define PRINT_LEVEL 4

#define LEVEL_DEBUG 4
#define LEVEL_INFO 3
#define LEVEL_WARN 2
#define LEVEL_ERROR 1

#if (PRINT_LEVEL >= LEVEL_DEBUG)
#define DEBUG(fmt, ...) \
    common_printf(stdout, LEVEL_DEBUG, __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__);
#else
#define DEBUG(fmt, ...) 
#endif

#if (PRINT_LEVEL >= LEVEL_INFO)
#define INFO(fmt, ...) \
    common_printf(stdout, LEVEL_INFO, __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__);
#else
#define INFO(fmt, ...) 
#endif

#if (PRINT_LEVEL >= LEVEL_WARN)
#define WARN(fmt, ...) \
    common_printf(stdout, LEVEL_WARN, __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__);
#else
#define WARN(fmt, ...) 
#endif

#if (PRINT_LEVEL >= LEVEL_ERROR)
#define ERROR(fmt, ...) \
    common_printf(stdout, LEVEL_ERROR, __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__);
#else
#define ERROR(fmt, ...) 
#endif

#endif
