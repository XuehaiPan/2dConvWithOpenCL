#ifndef CONVOLUTION_CONVOLUTION_H
#define CONVOLUTION_CONVOLUTION_H


#include <fstream>
#include <string>
#include <OpenCL/cl.h>


using std::ifstream;
using std::string;
using std::memset;


char *load_program_source(const string &source_file)
{
	ifstream fin;
	fin.open(source_file, ifstream::in);
	if (fin.is_open())
	{
		fin.seekg(0, std::fstream::end);
		size_t file_size = (size_t) fin.tellg();
		fin.seekg(0, std::fstream::beg);
		char *source = new char[file_size + 1];
		fin.read(source, file_size);
		fin.close();
		source[file_size] = '\0';
		return source;
	}
	else
	{
		printf("Error opening source_file %s\n", source_file.data());
		return NULL;
	}
}


float *gpu_compute(const string &source_file, const string &kernel_name, size_t global_work_size[],
                   const float input[], const float filter[],
                   int in_height, int in_width, int in_channels,
                   int out_height, int out_width, int out_channels,
                   int filter_size, int stride)
{
	int total_in_size = in_height * in_width * in_channels;
	int total_out_size = out_height * out_width * out_channels;
	int total_filter_size = filter_size * filter_size * in_channels * out_channels;
	float *output = new float[total_out_size]{};
	
	char *source;                                // kernel source
	int error_code;                              // error_code returned from api calls
	cl_device_id device_id;                      // compute device id
	cl_context context;                          // compute context
	cl_command_queue execute_queue, io_queue;    // compute command queue
	cl_program program;                          // compute program
	cl_kernel kernel;                            // compute kernel
	cl_mem input_on_device;                      // device memory used for the input array
	cl_mem output_on_device;                     // device memory used for the output array
	cl_mem filter_on_device;                     // device memory used for the filter array
	cl_event write_input_to_device, write_filter_to_device, execute_kernel;
	cl_event write_data_to_device[2];
	
	
	//init cl
	error_code = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
	if (error_code != CL_SUCCESS)
	{
		printf("Error: Failed to create a device group!\n");
		goto failure;
	}
	
	// create a compute context
	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &error_code);
	if (!context)
	{
		printf("Error: Failed to create a compute context!\n");
		goto failure;
	}
	
	// create a command queue
	io_queue = clCreateCommandQueue(context, device_id, 0, &error_code);
	execute_queue = clCreateCommandQueue(context, device_id, 0, &error_code);
	if (!execute_queue)
	{
		printf("Error: Failed to create a command queue!\n");
		goto failure;
	}
	
	// load kernel source
	source = load_program_source(source_file);
	if (!source)
	{
		printf("Error: Failed to load kernel source!\n");
		goto failure;
	}
	
	// create compute program
	program = clCreateProgramWithSource(context, 1, (const char **) &source, NULL, &error_code);
	if (!program || error_code != CL_SUCCESS)
	{
		printf("Error: Failed to create compute program!\n");
		goto failure;
	}
	
	// build program executable
	error_code = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	if (error_code != CL_SUCCESS)
	{
		size_t len;
		char buffer[4096];
		printf("Error: Failed to build program executable!\n");
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		printf("%s\n", buffer);
		goto failure;
	}
	
	// create compute kernel
	kernel = clCreateKernel(program, kernel_name.data(), &error_code);
	if (!kernel || error_code != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		goto failure;
	}
	
	// allocate device memory
	input_on_device = clCreateBuffer(context, CL_MEM_READ_ONLY, total_in_size * sizeof(float), NULL, &error_code);
	output_on_device = clCreateBuffer(context, CL_MEM_WRITE_ONLY, total_out_size * sizeof(float), NULL, &error_code);
	filter_on_device = clCreateBuffer(context, CL_MEM_READ_ONLY, total_filter_size * sizeof(float), NULL, &error_code);
	if (!input_on_device || !output_on_device || !filter_on_device || error_code != CL_SUCCESS)
	{
		printf("Error: Failed to allocate device memory!\n");
		goto failure;
	}
	
	// write data to device memory
	error_code = clEnqueueWriteBuffer(io_queue, input_on_device, CL_FALSE,
	                                  0, total_in_size * sizeof(float), input,
	                                  0, NULL, &write_input_to_device);
	error_code |= clEnqueueWriteBuffer(io_queue, filter_on_device, CL_FALSE,
	                                   0, total_filter_size * sizeof(float), filter,
	                                   0, NULL, &write_filter_to_device);
	write_data_to_device[0] = write_input_to_device;
	write_data_to_device[1] = write_filter_to_device;
	if (error_code != CL_SUCCESS)
	{
		printf("Error: Failed to write data to device memory!\n");
		goto failure;
	}
	
	// set kernel arguments
	error_code = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_on_device);
	error_code |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_on_device);
	error_code |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &filter_on_device);
	error_code |= clSetKernelArg(kernel, 3, sizeof(int), &in_height);
	error_code |= clSetKernelArg(kernel, 4, sizeof(int), &in_width);
	error_code |= clSetKernelArg(kernel, 5, sizeof(int), &in_channels);
	error_code |= clSetKernelArg(kernel, 6, sizeof(int), &out_height);
	error_code |= clSetKernelArg(kernel, 7, sizeof(int), &out_width);
	error_code |= clSetKernelArg(kernel, 8, sizeof(int), &out_channels);
	error_code |= clSetKernelArg(kernel, 9, sizeof(int), &filter_size);
	error_code |= clSetKernelArg(kernel, 10, sizeof(int), &stride);
	if (error_code != CL_SUCCESS)
	{
		printf("Error: Failed to set kernel arguments! %d\n", error_code);
		goto failure;
	}
	
	// execute kernel
	error_code = clEnqueueNDRangeKernel(execute_queue, kernel, 2, NULL, global_work_size, NULL,
	                                    2, write_data_to_device, &execute_kernel);
	if (error_code)
	{
		printf("Error: Failed to execute kernel!\n");
		goto failure;
	}
	
	// Read back the results from the device to verify the output
	error_code = clEnqueueReadBuffer(io_queue, output_on_device, CL_TRUE, 0, total_out_size * sizeof(float), output,
	                                 1, &execute_kernel, NULL);
	if (error_code != CL_SUCCESS)
	{
		printf("Error: Failed to read data from device! %d\n", error_code);
		goto failure;
	}
	
	// wait for the command queues to get serviced before reading back results
	clFinish(execute_queue);
	clFinish(io_queue);
	
	goto success;

failure:
	delete[] output;
	output = NULL;

success:
	// shutdown and cleanup
	delete[] source;
	clReleaseMemObject(input_on_device);
	clReleaseMemObject(output_on_device);
	clReleaseMemObject(filter_on_device);
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(execute_queue);
	clReleaseCommandQueue(io_queue);
	clReleaseContext(context);
	clReleaseDevice(device_id);
	clReleaseEvent(write_input_to_device);
	clReleaseEvent(write_filter_to_device);
	clReleaseEvent(execute_kernel);
	
	return output;
}


float *conv_gpu(const float *input, const float *filter,
                int in_height, int in_width, int in_channels, int out_channels, int filter_size, int stride)
{
	int out_height = (in_height - 1) / stride + 1,
		out_width = (in_width - 1) / stride + 1;
	size_t global_work_size[] = {(size_t) out_width, (size_t) out_height};
	return gpu_compute("conv.cl", "conv", global_work_size,
	                   input, filter,
	                   in_height, in_width, in_channels,
	                   out_height, out_width, out_channels,
	                   filter_size, stride);
}


float *deconv_gpu(const float *input, const float *filter,
                  int in_height, int in_width, int in_channels, int out_channels, int filter_size, int stride)
{
	int out_height = (in_height - 1) * stride + 1,
		out_width = (in_width - 1) * stride + 1;
	size_t global_work_size[] = {(size_t) in_width, (size_t) in_height};
	return gpu_compute("deconv.cl", "deconv", global_work_size,
	                   input, filter,
	                   in_height, in_width, in_channels,
	                   out_height, out_width, out_channels,
	                   filter_size, stride);
}


float *conv_cpu(const float *input, const float *filter,
                int in_height, int in_width, int in_channels, int out_channels, int filter_size, int stride)
{
	const int out_height = (in_height - 1) / stride + 1;
	const int out_width = (in_width - 1) / stride + 1;
	const int in_size = in_height * in_width;
	const int out_size = out_width * out_height;
	const int half_filter_size = filter_size >> 1;
	const int single_filter_size = filter_size * filter_size;
	const int channel_filter_size = in_channels * single_filter_size;
	
	float *output = new float[out_height * out_width * out_channels]{};
	
	float res, synapse;
	int filter_offset, synapse_offset, in_offset, out_offset;
	
	for (int out_row = 0, in_row = 0; out_row < out_height; ++out_row, in_row += stride)
	{
		for (int out_col = 0, in_col = 0; out_col < out_width; ++out_col, in_col += stride)
		{
			
			filter_offset = half_filter_size * filter_size + half_filter_size;
			out_offset = out_row * out_width + out_col;
			for (int out_ch = 0; out_ch < out_channels;
			     ++out_ch, out_offset += out_size, filter_offset += channel_filter_size)
			{
				res = 0.0F;
				synapse_offset = filter_offset;
				in_offset = in_row * in_width + in_col;
				for (int in_ch = 0; in_ch < in_channels;
				     ++in_ch, in_offset += in_size, synapse_offset += single_filter_size)
				{
					for (int i = -half_filter_size; i <= half_filter_size; ++i)
					{
						for (int j = -half_filter_size; j <= half_filter_size; ++j)
						{
							if (in_row + i >= 0 && in_row + i < in_height && in_col + j >= 0 && in_col + j < in_width)
							{
								if ((synapse = filter[synapse_offset + i * filter_size + j]) != 0)
									res += input[in_offset + i * in_width + j] * synapse;
							}
						}
					}
				}
				output[out_offset] = res;
			}
		}
	}
	return output;
}


float *deconv_cpu(const float *input, const float *filter,
                  int in_height, int in_width, int in_channels, int out_channels, int filter_size, int stride)
{
	int out_height = (in_height - 1) * stride + 1,
		out_width = (in_width - 1) * stride + 1;
	const int in_size = in_height * in_width;
	const int out_size = out_width * out_height;
	const int half_filter_size = filter_size >> 1;
	const int single_filter_size = filter_size * filter_size;
	const int channel_filter_size = in_channels * single_filter_size;
	
	float *output = new float[out_height * out_width * out_channels]{};
	
	float synapse, in_neuron;
	int filter_offset, synapse_offset, in_offset, out_offset;
	
	for (int in_row = 0, out_row = 0; in_row < in_height; ++in_row, out_row += stride)
	{
		for (int in_col = 0, out_col = 0; in_col < in_width; ++in_col, out_col += stride)
		{
			
			filter_offset = half_filter_size * filter_size + filter_size;
			out_offset = out_row * out_width + out_col;
			for (int out_ch = 0; out_ch < out_channels;
			     ++out_ch, out_offset += out_size, filter_offset += channel_filter_size)
			{
				for (int i = 0, out_row_offset = 0; i < stride; ++i, out_row_offset += out_width)
				{
					for (int j = 0; j < stride; ++j)
					{
						if (out_row + i < out_height && out_col + j < out_width)
							output[out_offset + out_row_offset + j] = 0.0F;
					}
				}
				synapse_offset = filter_offset;
				in_offset = in_row * in_width + in_col;
				for (int in_ch = 0; in_ch < in_channels;
				     ++in_ch, in_offset += in_size, synapse_offset += single_filter_size)
				{
					in_neuron = input[in_offset];
					if (in_neuron == 0.0F)
						continue;
					for (int i = -half_filter_size; i <= half_filter_size; ++i)
					{
						for (int j = -half_filter_size; j <= half_filter_size; ++j)
						{
							if (out_row + i >= 0 && out_row + i < out_height && out_col + j >= 0
							    && out_col + j < out_width)
							{
								if ((synapse = filter[synapse_offset + i * filter_size + j]) != 0)
									output[out_offset + i * out_width + j] += in_neuron * synapse;
							}
						}
					}
				}
			}
		}
	}
	return output;
}


#endif //CONVOLUTION_CONVOLUTION_H