# Copyright 2019 Inspur Corporation. All Rights Reserved.
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
#    
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
#==============================================================================*/
cmake_minimum_required (VERSION 2.6)
project (TF2_AUTO_CONFIG)

include_directories(inc)
set (TF2_AUTO_CONFIG_SRC src/tf2_auto_config.cpp src/tf2_auto_param.cpp src/cycles_computation.cpp src/fpganetworkinterface.cpp)

#set (CMAKE_BUILD_TYPE "Debug")                                     #debug option
#set (CMAKE_CXX_FLAG_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
set (CMAKE_CXX_FLAG_RELEASE "$ENV{CXXFLAGS} -03 -Wall")

add_executable (tf2_auto_config ${TF2_AUTO_CONFIG_SRC})

