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

#ifndef __DEFINES_H__
#define __DEFINES_H__

//------------------------------------------------------------------------------------//
// defines.h                                                                          //
// Scope: Used by device code                                                         //
// Function: Defines all the macros and miscellaneous functions.                      //
//------------------------------------------------------------------------------------//

#ifdef OPENCL
#define STATIC
#define CONSTANT constant
#else
#define STATIC static
#define CONSTANT static const
#endif

#define NUM_IMAGES 1

#ifndef OPENCL
#include <cmath>
#include <cstdio>

typedef unsigned int uint;
typedef unsigned short ushort;
typedef unsigned char uchar;

#endif

#define NEXT_DIVISIBLE(X, Y) ( ( X % Y ) == 0 ? X : ( X + Y - ( X % Y ) ) )
#define CEIL(X, Y) ( ( X - 1 ) / (Y) +1 )
#define X_OR_SHIFT_Y(X,Y) ( ( X ) | ( ( X ) >> ( Y ) ) )
#define NEXT_POWER_OF_2(X) ( X_OR_SHIFT_Y(X_OR_SHIFT_Y(X_OR_SHIFT_Y(X_OR_SHIFT_Y(X_OR_SHIFT_Y(X-1, 1), 2), 4), 8), 16) + 1 )
#define MYMAX2(X, Y) ( X >= Y ? X : Y )

#define BIT_MASK(num_bits) ((1ULL << (num_bits))-1)
#define BIT_MASK_RANGE(start_bit, end_bit) (BIT_MASK((start_bit)+1) - BIT_MASK(end_bit))
#define BIT_SEL(value, start_bit, end_bit) ((value & BIT_MASK_RANGE(start_bit, end_bit)) >> end_bit)
#define BIT_IS_SET(value, bit_num) (((value) & (1ULL << (bit_num))) != 0)

#define LOG2_2(x)  ((x) & 0x2        ? 1                       : 0)
#define LOG2_4(x)  ((x) & 0xC        ? 2  + LOG2_2((x)  >>  2) : LOG2_2(x))
#define LOG2_8(x)  ((x) & 0xF0       ? 4  + LOG2_4((x)  >>  4) : LOG2_4(x))
#define LOG2_16(x) ((x) & 0xFF00     ? 8  + LOG2_8((x)  >>  8) : LOG2_8(x))
#define LOG2(x)    ((x) & 0xFFFF0000 ? 16 + LOG2_16((x) >> 16) : LOG2_16(x))
#define CLOG2(x)   (LOG2((x)-1)+1)

// how many different banks are connected to each load from input cache
// formula: ( W_VECTOR / GCD( GCD(FW_VECTOR, OW_VECTOR), W_VECTOR) )
// if FW_VECTOR is equal to all S's , i.e. ss loop always executes one time, then FW_VECTOR can be removed from equation
// because input address always increments by OW_VECTOR
//#define INPUT_CACHE_BANK_PER_RDDATA ( W_VECTOR / GCD( OW_VECTOR, W_VECTOR ) )
#define INPUT_CACHE_BANK_PER_RDDATA (W_VECTOR / GCD( GCD(FW_VECTOR, OW_VECTOR), W_VECTOR))

#define DOUBLE_BUFFER_DIM 2

#define DDR_BANDWIDTH_IN_BYTES ( 64 )
#define DDR_BANDWIDTH_IN_FLOATS ( DDR_BANDWIDTH_IN_BYTES / 4 )

#define POOL_OFFSET_P (POOL_WINDOW_MAX-1)
#define POOL_OFFSET_Q (POOL_WINDOW_MAX-1)

//#define W_VECTOR W_VECTOR
//#define W_VECTOR W_VECTOR

#define GCD(X, Y) ((X % Y) == 0 ? Y : (Y % X) == 0 ? X : ((X % 2) == 0 && (Y % 2) == 0) ? 2 : 1)

// Initializes counter.
#define INIT_COUNTER(name) \
  int  name          = 0; \
  int  name##__count = 0; \
  bool name##__done  = false; \
  bool name##__init  = true;

// Configures counter parameters.
#define SET_COUNTER(name, capacity, start, end, step) \
  int name##__width = CLOG2(capacity); \
  int name##__start = (start); \
  int name##__end   = (end); \
  int name##__step  = (step); \
  int name##__reset_value = (name##__end - name##__start - name##__step - 1) & \
      BIT_MASK(name##__width+1); \
  if (name##__init) { \
    RESET_COUNTER(name); \
    name##__init = false; \
  }

// Resets a counter back to its initial value.
#define RESET_COUNTER(name) do { \
  name          = name##__start & BIT_MASK(name##__width); \
  name##__count = name##__reset_value; \
  name##__done  = false; \
} while (0)

// Increments a counter value by the amount specified by "step".
#define INCREASE_COUNTER(name) do { \
  name##__done  = BIT_IS_SET(name##__count, name##__width); \
  name          = (name + name##__step) & BIT_MASK(name##__width); \
  name##__count = (name##__count - name##__step) & BIT_MASK(name##__width+1); \
} while (0)

// Returns true if the counter is set to its initial value.
#define COUNTER_FIRST(name) \
  ((name##__count & BIT_MASK(name##__width+1)) == name##__reset_value)

// Returns true if the counter is set to the last valid value before being finished.
#define COUNTER_LAST(name) BIT_IS_SET(name##__count, name##__width)

// Returns true if the counter has been incrmented past its last valid value and it now finished.
#define COUNTER_DONE(name) name##__done

#define MAX_COUNTER_CAPACITY (1<<30)

#if defined(USE_AUTORUN_KERNELS) && !defined(EMULATOR)
#define AUTORUN __attribute__((autorun))
#else
#define AUTORUN
#endif

#define TASK __attribute__((max_global_work_dim(0)))

#endif // __DEFINES_H__
