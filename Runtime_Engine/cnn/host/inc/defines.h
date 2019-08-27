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
// -------------------------------------------------------------------------- //
// defines.h:
//
// Define all the macros.
// -------------------------------------------------------------------------- //

#define STATIC static
#define CONSTANT static const

STATIC int next_power_of_2(int x) {
  x--;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return x + 1;
}


#define NEXT_DIVISIBLE(X, Y) ( ( X % Y ) == 0 ? X : ( X + Y - ( X % Y ) ) )
#define CEIL(X, Y) ( ( X - 1 ) / (Y) +1 )
#define X_OR_SHIFT_Y(X,Y) ( ( X ) | ( ( X ) >> ( Y ) ) )
#define NEXT_POWER_OF_2(X) ( X_OR_SHIFT_Y(X_OR_SHIFT_Y(X_OR_SHIFT_Y(X_OR_SHIFT_Y(X_OR_SHIFT_Y(X-1, 1), 2), 4), 8), 16) + 1 )

#endif // __DEFINES_H__
