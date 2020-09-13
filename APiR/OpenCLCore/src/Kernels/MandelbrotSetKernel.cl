constant uchar color_map[17 * 3] =
	{
	0, 0, 0,
	66, 45, 15,
	25, 7, 25,
	10, 0, 45,
	5, 5, 73,
	0, 7, 99,
	12, 43, 137,
	22, 81, 175,
	56, 124, 209,
	132, 181, 229,
	209, 234, 247,
	239, 232, 191,
	247, 201, 94,
	255, 170, 0,
	204, 127, 0,
	153, 86, 0,
	104, 51, 2,
	};

float get_iterations(int i_x, int i_y, int i_width, int i_height, int i_max_iterations)
	{
	const float cx = 3.5 * i_x / i_width - 2.5;
	const float cy = 2.0 * i_y / i_height - 1.0;
	float zx = 0;
	float zy = 0;
	int iter = 0;
	while (iter < i_max_iterations)
		{
		const float tempzx = zx * zx - zy * zy + cx;
		zy = 2 * zx * zy + cy;
		zx = tempzx;
    if(zx * zx + zy * zy > 4)
      break;
		++iter;
		}
	return iter;
	}

int get_index(int i_x, int i_y, int i_width)
	{
	return i_y * i_width * 4 + i_x * 4;
	}

kernel void mandelbrot_set(int i_max_iterations, global uchar* o_picture)
	{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	const int width = get_global_size(0);
	const int height = get_global_size(1);
	const int pixel_coords = get_index(x,y,width);
	const int iter = get_iterations(x,y,width,height,i_max_iterations);
  int id = iter;
  //if(iter != i_max_iterations)
  //  id = (iter * 100 / i_max_iterations) % 17;
	o_picture[pixel_coords + 0] = color_map[id * 3 + 0];
	o_picture[pixel_coords + 1] = color_map[id * 3 + 1];
	o_picture[pixel_coords + 2] = color_map[id * 3 + 2];
	o_picture[pixel_coords + 3] = 255;
	}