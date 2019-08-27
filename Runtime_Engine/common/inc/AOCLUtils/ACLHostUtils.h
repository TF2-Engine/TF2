//===- Altera OpenCL Host Utilities -===//
// vim: set ts=2 sw=2 expandtab:
//
// Copyright (c) 2011 Altera Corporation.
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// Contains helper functions that support runtime environment creation and
// management by the host program.  Also supports creation of GPU targets
// without their proprietary helper libraries.
//
//===----------------------------------------------------------------------===//

#ifndef ACL_HOST_UTILS_H
#define ACL_HOST_UTILS_H

#include <CL/opencl.h>

// Error callback routine - passed to the CL library when creating a
// context.  On GPU this is the only way to see runtime device errors.
void acl_errorCallback(const char* errinfo, const void *private_info, size_t cb, void *user_data);

// Display error string and returned status ID
void acl_dump_error(const char *str, cl_int status);

// Query the available OpenCL platforms.  Currently only handles
// a single platform in the system.
int acl_getPlatform( cl_platform_id *platform );

// Get a valid OpenCL device.  If building for Altera target, looks for an
// accelerator.  Else assumes that targeting GPU - if more than one GPU,
// grabs Tesla board.  If more than one FPGA device, errors out (see
// acl_getFirstDevice())
int acl_getDevice( cl_device_id *device, const cl_platform_id &platform );

// Same as acl_getDevice, except if multiple devices, grabs the first one
// instead of erroring out
int acl_getFirstDevice( cl_device_id *device, const cl_platform_id &platform );

// Underlying function to acquire a device
int acl_getDevice_internal( cl_device_id *device, const cl_platform_id &platform, const bool get_first_device );

// Query and display information about an OpenCL platform
int acl_dumpPlatformInfo( cl_platform_id &platform );

// Create a context with the passed in platform and device
int acl_getContext( cl_context *context, const cl_platform_id &platform, const cl_device_id &device );

// Create a command queue on the device in the passed in context
int acl_getQueue( cl_command_queue *queue, const cl_device_id &device, const cl_context &context );

// Load a program from source CL file
int acl_createProgramFromSource(cl_program *program, const cl_context &context, const char *f_source_name);

// Load a program from source, and append a header file to the beginning of
// the text.  #includes don't work in cl files (at least on the GPU).
int acl_createProgramFromSource_withHeader(cl_program *program, const cl_context &context, const char *f_source_name, const char *f_header_name);

// Allocate and free memory aligned to value that's good for
// Altera OpenCL performance.
void *aligned_malloc (size_t size);
void  aligned_free (void *ptr);

// OS-independent random number generator
void acl_srand(unsigned int seed);
unsigned int acl_rand(void);
unsigned int acl_rand_max(void);

#endif // ACL_HOST_UTILS_H
