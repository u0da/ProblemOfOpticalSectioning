#define M_PI 3.1415927f

float2 multiply_complex (const float2  a, const float2 b)
{
	 return (float2)(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

float2 divide_complex(const float2 a, const float2 b)
{
	const float div = dot(b, b);

	return (float2)(dot(a, b) / div, (a.y * b.x - a.x * b.y) / div);
}

int M (float x, float y) 
{
	if ((pown(x, 2) + pown(y, 2)) < pown(M_PI*0.5f, 2))
		return (1);
	else
		return (0);
}

float p_s (float x, float y, float delta_z) 
{
	// float d_1 = 57.4f * 10e-3;
	// float d_0 = 37.0f * 10e-3;
	// float r_0 = 4.5f * 10e-3;
	// float lamba = 0.55f * 10e-6;
	// float w = 2.34f * 10e-5;

	// return (-M_PI * lamba * (d_1*d_1)*delta_z * (pown(x, 2) + pown(y, 2)))/pown(d_0 + w, 2);
	return (0.375f * fabs(delta_z) * M_PI * (pown(x, 2) + pown(y, 2)));
}

float p (float x, float y) {
	return (M_PI*0.5f * (pown(x, 2) + pown(y, 2)));
}


__kernel void h_init_kernel (const float delta_z, __global float *h_real,  __global float *h_imag)
{
	int i = get_global_id(0);
	int j = get_global_id(1);
	
	int sizex = get_global_size(0);
	int sizey = get_global_size(1);
	
	// float d_1 = 57.4f * 1e-3;
	// float d_0 = 37.0f * 1e-3;
	// float r_0 = 4.5f * 1e-3;
	// float lamba = 0.55f * 1e-6;
	// float w = 2.34f * 1e-4;
	// float a = 4.0f * r_0/(lamba * d_1);

	int index = i * sizey + j;
	float x = 0, y = 0;

	x = (M_PI / sizex) * (i - sizex/2);
	y = (M_PI / sizey) * (j - sizey/2);
	// float m_result = M(x*lamba*d_1, y*lamba*d_1);
	float m_result = M(x, y);
	float p_result = p(x, y);
	float p_s_result = p_s(x, y, delta_z);
	
	float cos_result = 0.0f;
	float sin_result = sincos(p_result + p_s_result, &cos_result);
	
	h_real[index] = m_result * cos_result;
	h_imag[index] = m_result * sin_result;
}

__kernel void h_squared_abs_kernel (__global float *h_real, __global float *h_imag)
{
	const int i = get_global_id(0);
	const float2 h_complex = (float2)(h_real[i], h_imag[i]);

	h_real[i] = dot(h_complex, h_complex);
	h_imag[i] = 0.0f; 
}

__kernel void fft_shift_row_kernel(__global float *array, const int num_col, const int offset)
{
	int i = get_global_id(0);

	int half_num_col = num_col/2;
	int row_start_index = i * num_col + offset;
	
	for (int j = 0; j < half_num_col; j++)
	{
		float tmp = array[row_start_index + j];
		array[row_start_index + j] = array[row_start_index + half_num_col + j];
		array[row_start_index + half_num_col + j] = tmp;
	}
	
}

__kernel void fft_shift_col_kernel (__global float *array, const int num_row, const int offset)
{
	int j = get_global_id(0);
	int num_col = get_global_size(0);

	// sizex = num_col
	// sizey = num_row

	int half_num_row = num_row/2;
	__global float *array_local = array + offset;

	for (int i = 0; i < half_num_row; i++)
	{
		float tmp = array_local[i * num_col + j];
		array_local[i * num_col + j] = array_local[(i + half_num_row) * num_col + j];
		array_local[(i + half_num_row) * num_col + j] = tmp;
	}
}

void multiply_matrix_by_vector (__global const float2 *A, 
								__global const float2 *vec,
								__global float2 *result)
{
	for (int i = 0; i < AMOUNT_OF_PICS; i++)
	{
		float2 res = (float2)(0, 0);

		const int row_start = i * AMOUNT_OF_PICS;
		for (int j = 0; j < AMOUNT_OF_PICS; j++)
			res += multiply_complex(A[row_start + j], vec[j]);

		result[i] = res;
	}     
}

// Размер матрицы А = (AMOUNT_OF_PICS, AMOUNT_OF_PICS)
void inverse_matrix (__global float2 *LU)
{
	
	// LU:
	for (int i = 1; i < AMOUNT_OF_PICS; i++)
	{
		for (int j = 0; j < AMOUNT_OF_PICS; j++)
		{
			float2 tmp_sum_of_multiplies_l_u = (float2)(0, 0);
			int l = min(i, j);

			for (int k = 0; k < l; k++)
				tmp_sum_of_multiplies_l_u += multiply_complex(LU[i*AMOUNT_OF_PICS + k], LU[k*AMOUNT_OF_PICS + j]);
			
			 LU[i*AMOUNT_OF_PICS + j] -= tmp_sum_of_multiplies_l_u;
			
			 if (i > j)
				 LU[i * AMOUNT_OF_PICS + j] = divide_complex(LU[i * AMOUNT_OF_PICS + j], LU[j * AMOUNT_OF_PICS + j]);
		}
	}


	// Inverse:

	float2 tmp_sum = (float2)(0, 0);

	// L^-1
	for (int j = 0; j < AMOUNT_OF_PICS; j++)
	{

		for (int i = j+1; i < AMOUNT_OF_PICS; i++)
		{ 
			tmp_sum = LU[i* AMOUNT_OF_PICS + j];

			for (int k = j+1; k < i; k++)
				tmp_sum += multiply_complex(LU[i * AMOUNT_OF_PICS + k], LU[k * AMOUNT_OF_PICS + j]);

			LU[i * AMOUNT_OF_PICS + j] = -tmp_sum;
		}
	}


	// U^-1
	for (int j = AMOUNT_OF_PICS-1; j >= 0; j--)
	{
		for (int i = j; i >= 0 ; i--)
		{
			tmp_sum = (float2)(0, 0);

			for (int k = i+1; k <= j ; k++)
				tmp_sum += multiply_complex(LU[i * AMOUNT_OF_PICS + k], LU[k * AMOUNT_OF_PICS + j]);

			LU[i* AMOUNT_OF_PICS + j] = divide_complex((float2)((i == j) ? 1 : 0, 0) - tmp_sum, LU[i * AMOUNT_OF_PICS + i]);
		}
	}

	
	// U^-1*L^-1

	for (int i = 0; i < AMOUNT_OF_PICS; i++)
	{
		for (int j = 0; j < AMOUNT_OF_PICS; j++)
		{
			tmp_sum = (float2)(0, 0);
			for (int k = max(i, j); k < AMOUNT_OF_PICS; k++)
			{
				tmp_sum += multiply_complex(LU[i*AMOUNT_OF_PICS + k], (k == j) ? (float2)(1.0f, 0.0f) : LU[k*AMOUNT_OF_PICS + j]);
			}
			LU[i * AMOUNT_OF_PICS + j] = tmp_sum;
		}
	}

}

__kernel void pixel_recovery_kernel (__global const float *images_real, __global const float *images_imag,
							  __global const float *h_real, __global const float *h_imag,
							  __global float2 *tmp_multiply_matrix,
							  __global float2 *O,
							  __global float2 *prepared_vec_of_input_pixels,
							  __global float2 *tmp_multiply_vec,
							  __global float *result_real, __global float *result_imag)
{
	const int pixel_row_index = get_global_id(0);
	const int pixel_col_index = get_global_id(1);
	const ulong num_row = get_global_size(0);
	const ulong N = num_row * num_row;

	const ulong pixel_offset = pixel_row_index*num_row + pixel_col_index;
	__global float2 *tmp_multiply_matrix_local =  tmp_multiply_matrix + pixel_offset*AMOUNT_OF_PICS*AMOUNT_OF_PICS;

	__global float2 *tmp_multiply_vec_local =  tmp_multiply_vec + pixel_offset*AMOUNT_OF_PICS;

	__global float2 *prepared_vec_of_input_pixels_local = prepared_vec_of_input_pixels + pixel_offset * AMOUNT_OF_PICS;
	
	
	/* 
		A_real[i][k] = h_real[N*abs(i-k) + pixel_offset];
		
		A = np.mat([ [mas_of_h[0][u][v], mas_of_h[1][u][v], mas_of_h[2][u][v]], 
			[mas_of_h[1][u][v], mas_of_h[0][u][v], mas_of_h[1][u][v]],
			[mas_of_h[2][u][v], mas_of_h[1][u][v], mas_of_h[0][u][v]] ], dtype = np.complex128)
	*/

	// (mu * A.H.dot(A))
	for (int i = 0; i < AMOUNT_OF_PICS; i++)
	{
		const int cur_row = i*AMOUNT_OF_PICS;

		for (int j = 0; j < AMOUNT_OF_PICS; j++)
		{
			float2 res = (float2)(0.0f, 0.0f);

			for (int k = 0; k < AMOUNT_OF_PICS; k++)
			{
				const ulong offset_i_k = N * abs(i - k) + pixel_offset;
				const ulong offset_k_j = N * abs(j - k) + pixel_offset;

				res += multiply_complex((float2)(h_real[offset_i_k], -h_imag[offset_i_k]),
										(float2)(h_real[offset_k_j], h_imag[offset_k_j]));
			}

			tmp_multiply_matrix_local[cur_row + j] = MU * res;
		}	
	}


	// E + mu*A.H.dot(A)
	for (int i = 0; i < AMOUNT_OF_PICS; i++)
		tmp_multiply_matrix_local[i * AMOUNT_OF_PICS + i] += (float2)(1.0f, 0.0f);	


	// (A.H.dot(b))
	for (int i = 0; i < AMOUNT_OF_PICS; i++)
	{
		float2 res = (float2)(0, 0);

		for (int j = 0; j < AMOUNT_OF_PICS; j++)
		{
			const ulong offset_h = N*abs(i - j) + pixel_offset;
			const ulong offset_images = j * N + pixel_offset;
	
			const float2 h = (float2)(h_real[offset_h], -h_imag[offset_h]);
			const float2 images = (float2)(images_real[offset_images], images_imag[offset_images]) / N;

			res += multiply_complex(h, images);
		}

		tmp_multiply_vec_local[i] = res;
	}

	// factor = np.linalg.inv(E + mu*A.H.dot(A))

	inverse_matrix(tmp_multiply_matrix_local);

	// factor.dot( (A.H.dot(b))

	multiply_matrix_by_vector(tmp_multiply_matrix_local, tmp_multiply_vec_local, prepared_vec_of_input_pixels_local);


	__global float2 *O_local = O + pixel_offset * AMOUNT_OF_PICS;
	
	for (int i = 0; i < AMOUNT_OF_PICS; i++)
		O_local[i] = prepared_vec_of_input_pixels_local[i];
		
	/*
	for k in range(iters) :
		lala = (factor).dot(O)
		O = (lala + factor.dot((A.H.dot(b)))
	*/
	for (int k = 1; k < ITERS; k++)
	{
		//  lala = (factor).dot(O)
		multiply_matrix_by_vector(tmp_multiply_matrix_local, O_local, tmp_multiply_vec_local);
		for (int i = 0; i < AMOUNT_OF_PICS; i++)
			O_local[i] = tmp_multiply_vec_local[i] + prepared_vec_of_input_pixels_local[i];
	}

	/*
	for i in range(amount_of_pics):
        res[i][u][v] = O[i]
	*/
	for (int j = 0; j < AMOUNT_OF_PICS; j++)
	{
		const ulong offset = j * N + pixel_offset;
		const float2 O_local_j = O_local[j];

		result_real[offset] = O_local_j.x;
		result_imag[offset] = O_local_j.y;
	}
}
