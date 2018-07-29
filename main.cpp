#include <cstdio>
#include <cstdlib>
#include <random>
#include <opencv2/opencv.hpp>
#include "convolution.h"


using namespace std;

int in_width, in_height, in_channels,
	out_height, out_width, out_channels,
	filter_size, stride,
	total_in_size, total_out_size, total_filter_size;

float *input, *filter, *output;


void get_random_array(float *&array, const int size)
{
	default_random_engine e(0);
	normal_distribution<float> distribution(0, 1);
	delete[] array;
	array = new float[size];
	for (int i = 0; i < size; ++i)
		array[i] = distribution(e);
}


void read_input_array_from_file(const string &file)
{
	in_height = 128;
	in_width = 128;
	in_channels = 1;
	delete[] input;
	int in_size = in_height * in_width;
	input = new float[in_size * in_channels];
	
	ifstream fin;
	fin.open(file, ifstream::in);
	if (fin.is_open())
	{
		for (int c = 0; c < in_channels; ++c)
		{
			for (int i = 0; i < in_height; ++i)
			{
				for (int j = 0; j < in_width; ++j)
				{
					fin >> input[c * in_size + i * in_width + j];
				}
			}
		}
		fin.close();
	}
}


void write_output_array_to_file(const string &file)
{
	int out_size = out_height * out_width;
	printf("%d %d %d\n", out_height, out_width, out_size);
	ofstream fout;
	fout.open(file, ofstream::out);
	if (fout.is_open())
	{
		float val = 0;
		for (int c = 0; c < out_channels; ++c)
		{
			for (int i = 0; i < out_height; ++i)
			{
				for (int j = 0; j < out_width; ++j)
				{
					fout << output[c * out_size + i * out_width + j];
					if (j + 1 < out_width)
						fout << ' ';
					else
						fout << endl;
				}
			}
		}
	}
}


void read_input_array_from_image(const string &file)
{
	cv::Mat image = cv::imread(file, cv::IMREAD_COLOR | cv::IMREAD_UNCHANGED);
	in_height = image.rows;
	in_width = image.cols;
	in_channels = image.channels();
	delete[] input;
	int in_size = in_height * in_width;
	input = new float[in_size * in_channels];
	
	for (int c = 0; c < in_channels; ++c)
	{
		for (int i = 0; i < in_height; ++i)
		{
			for (int j = 0; j < in_width; ++j)
			{
				input[c * in_size + i * in_width + j] = image.ptr(i, j)[c];
			}
		}
	}
}


void write_output_array_to_image(const string &file)
{
	cv::Mat image = cv::Mat::zeros(out_height, out_width, CV_8UC(out_channels));
	int out_size = out_height * out_width;
	float val = 0.0F;
	for (int c = 0; c < out_channels; ++c)
	{
		for (int i = 0; i < out_height; ++i)
		{
			for (int j = 0; j < out_width; ++j)
			{
				val = fabs(output[c * out_size + i * out_width + j]);
				if (val > 255)
					val = 255;
				image.ptr(i, j)[c] = (unsigned char) val;
			}
		}
	}
	cv::imwrite(file, image);
}


int main(int argc, char *argv[])
{
	if (argc != 10)
	{
		filter_size = 7;
		stride = 1;
		in_height = 1920;
		in_width = 1080;
		in_channels = 3;
		out_channels = 1;
	}
	else
	{
		// fout = argv[2];
		// filter_file = argv[3];
		filter_size = atoi(argv[4]);
		stride = atoi(argv[5]);
		in_height = atoi(argv[6]);
		in_width = atoi(argv[7]);
		in_channels = atoi(argv[8]);
		out_channels = atoi(argv[9]);
	}
	
	read_input_array_from_image("in.jpg");
	
	// 锐化边缘
	{
		// filter = new float[81]{ 1, 1, 1, 1, 1, 1, 1, 1, 1,
		//                        0, 0, 0, 0, 0, 0, 0, 0, 0,
		//                        0, 0, 0, 0, 0, 0, 0, 0, 0,
		//
		//                        0, 0, 0, 0, 0, 0, 0, 0, 0,
		//                         0, 0, 0, 0, 1, 0, 0, 0, 0,
		//                        0, 0, 0, 0, 0, 0, 0, 0, 0,
		//
		//                        0, 0, 0, 0, 0, 0, 0, 0, 0,
		//                        0, 0, 0, 0, 0, 0, 0, 0, 0,
		//                         0, 0, 0, 0, 1, 0, 0, 0, 0,
		// };
		// filter_size = 3;
		// out_channels = 3;
		// stride = 2;
	}
	
	// 锐化边缘
	{
		// filter = new float[81]{1, 1, 1,
		//                        1, -7, 1,
		//                        1, 1, 1,
		//                        1, 1, 1, 1, -7, 1, 1, 1, 1,
		//                        1, 1, 1, 1, -7, 1, 1, 1, 1,
		//
		//                        1, 1, 1, 1, -7, 1, 1, 1, 1,
		//                        1, 1, 1, 1, -7, 1, 1, 1, 1,
		//                        1, 1, 1, 1, -7, 1, 1, 1, 1,
		//
		//                        1, 1, 1, 1, -7, 1, 1, 1, 1,
		//                        1, 1, 1, 1, -7, 1, 1, 1, 1,
		//                        1, 1, 1, 1, -7, 1, 1, 1, 1
		// };
		// filter_size = 3;
		// out_channels = 3;
		// stride = 1;
	}
	
	// 动感模糊
	{
		filter_size = 25;
		int single_filter_size = filter_size * filter_size;
		filter = new float[3 * 3 * single_filter_size]{};
		
		for (int c = 0; c < 3; ++c)
		{
			for (int i = 0; i < single_filter_size; ++i)
			{
				filter[c * 4 * single_filter_size + i] = 1.0F / single_filter_size;
			}
		}
		out_channels = 3;
		stride = 1;
	}
	
	clock_t start_time, end_time;
	out_height = (in_height - 1) / stride + 1;
	out_width = (in_width - 1) / stride + 1;
	total_in_size = in_height * in_width * in_channels;
	total_out_size = out_height * out_width * out_channels;
	total_filter_size = filter_size * filter_size * in_channels * out_channels;
	
	// get_random_array(input, total_in_size);
	// get_random_array(filter, total_filter_size);
	
	printf("convolution:\n");
	printf("[in_width, in_height, in_channels] = [%d, %d, %d]\n", in_width, in_height, in_channels);
	printf("[out_height, out_width, out_channels] = [%d, %d, %d]\n", out_width, out_height, out_channels);
	printf("[filter_size, stride] = [%d, %d]\n", filter_size, stride);
	
	start_time = clock();
	output = conv_cpu(input, filter, in_height, in_width, in_channels, out_channels, filter_size, stride);
	end_time = clock();
	delete[] output;
	printf("execution time of convolution on CPU: %.2lf ms\n", 1000 * double(end_time - start_time) / CLOCKS_PER_SEC);
	
	start_time = clock();
	output = conv_gpu(input, filter, in_height, in_width, in_channels, out_channels, filter_size, stride);
	end_time = clock();
	write_output_array_to_image("out.jpg");
	delete[] output;
	printf("execution time of convolution on GPU: %.2lf ms\n", 1000 * double(end_time - start_time) / CLOCKS_PER_SEC);
	
	delete[] input, output, filter;
	return 0;
}
