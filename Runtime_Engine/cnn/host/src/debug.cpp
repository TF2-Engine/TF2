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

#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <stdarg.h>
#include "debug.h"

#define ANSI_COLOR_RED "\x1b[1;31m" 
#define ANSI_COLOR_GREEN "\x1b[1;32m"
#define ANSI_COLOR_YELLOW "\x1b[1;33m"
#define ANSI_COLOR_PURPLE      "\x1b[1;35m"
#define ANSI_COLOR_CYAN      "\x1b[29m"
#define ANSI_COLOR_CYAN_LESS "\x1b[29m"
#define ANSI_COLOR_RESET      "\x1b[0m" 

#define PRINT_DEBUG          "DEBUG"
#define PRINT_INFO               "INFO"
#define PRINT_WARN               "WARN"
#define PRINT_ERROR          "ERROR"

#define PRINT_LEN 10240

static void _common_printf(FILE * fp, const char * time_cl, const char * type_cl, 
                                  const char * func_cl, const char * content_cl,
                                   const char * level, const char * file,
                                   const int line, const char * func, const char * fmt)
{ 
    struct timeval tv; 
    gettimeofday(&tv, NULL);
    char datestr[20]; 
    struct tm tm; 
    time_t timesec = tv.tv_sec; 
    localtime_r(&timesec, &tm); 
    strftime(datestr, sizeof(datestr), "%Y-%m-%d %H:%M:%S", &tm); 
    fprintf(fp, "%s" "%s.%06ld" ANSI_COLOR_RESET 
                //"%s" "[%-05s]" ANSI_COLOR_RESET 
                "%s" "[%s]" ANSI_COLOR_RESET
                //"%s" "[%s, %4d][%-10s] " ANSI_COLOR_RESET 
                "%s" "[%s, %d][%s] " ANSI_COLOR_RESET
                "%s" "%s" ANSI_COLOR_RESET, 
                time_cl, datestr, tv.tv_usec, type_cl,
                level, func_cl, file, line, func, content_cl, fmt);
    fflush(fp); 
}

int common_printf(FILE * fp, const int level, const char * file,
                         const int line, const char * func, const char * fmt, ...)
{
    int i;
    char buf[PRINT_LEN];
    
    va_list args;    
    va_start(args, fmt);
    i = vsnprintf(buf, PRINT_LEN, fmt, args);
    va_end(args);

    switch (level) {
        case LEVEL_DEBUG:
                _common_printf(fp, ANSI_COLOR_CYAN_LESS, ANSI_COLOR_GREEN,
                           ANSI_COLOR_CYAN, ANSI_COLOR_GREEN,
                           PRINT_DEBUG, file, line, func, buf);
                break;                           
                          
        case LEVEL_INFO:
                _common_printf(fp, ANSI_COLOR_CYAN_LESS, ANSI_COLOR_YELLOW,
                           ANSI_COLOR_CYAN, ANSI_COLOR_YELLOW,
                           PRINT_INFO, file, line, func, buf);
                break;
                
        case LEVEL_WARN:
                _common_printf(fp, ANSI_COLOR_CYAN_LESS, ANSI_COLOR_PURPLE,
                           ANSI_COLOR_CYAN, ANSI_COLOR_PURPLE,
                           PRINT_WARN, file, line, func, buf);
                break;           
        case LEVEL_ERROR:
                _common_printf(fp, ANSI_COLOR_CYAN_LESS, ANSI_COLOR_RED,
                           ANSI_COLOR_CYAN, ANSI_COLOR_RED,
                           PRINT_ERROR, file, line, func, buf);        
                break;
        default:
                break;
    }    

    return i;
}
