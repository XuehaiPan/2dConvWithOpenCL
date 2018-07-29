__kernel void deconv(__global float *input,
                     __global float *output,
                     __global float *filter,
                     const int in_height,
                     const int in_width,
                     const int in_channels,
                     const int out_height,
                     const int out_width,
                     const int out_channels,
                     const int filter_size,
                     const int stride)
{
	const int in_row = get_global_id(1);
	const int in_col = get_global_id(0);
	const int out_row = in_row * stride;
	const int out_col = in_col * stride;
	const int in_size = in_height * in_width;
	const int out_size = out_width * out_height;
	const int half_filter_size = filter_size >> 1;
	const int single_filter_size = filter_size * filter_size;
	const int channel_filter_size = in_channels * single_filter_size;

	float sum, synapse, in_neuron;
	int filter_offset, synapse_offset, in_offset, out_offset;

	filter_offset = half_filter_size * filter_size + filter_size;
	out_offset = out_row * out_width + out_col;
	for (int out_ch = 0; out_ch < out_channels; ++out_ch, out_offset += out_size, filter_offset += channel_filter_size)
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
		for (int in_ch = 0; in_ch < in_channels; ++in_ch, in_offset += in_size, synapse_offset += single_filter_size)
		{
			in_neuron = input[in_offset];
			if (in_neuron == 0.0F)
				continue;
			for (int i = -half_filter_size; i <= half_filter_size; ++i)
			{
				for (int j = -half_filter_size; j <= half_filter_size; ++j)
				{
					if (out_row + i >= 0 && out_row + i < out_height && out_col + j >= 0 && out_col + j < out_width)
					{
						if ((synapse = filter[synapse_offset + i * filter_size + j]) != 0)
							output[out_offset + i * out_width + j] += in_neuron * synapse;
					}
				}
			}
		}
	}
}