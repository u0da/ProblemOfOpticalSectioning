#include "Cl_Buffer_pair.h"
#include "FFT_OpenCL_data.h"
#include "Image.h"

#include "clFFT.h"

#include <cstring>
#include <cmath>
#include <complex>
#include <ctime>
#include <cinttypes>
#include <cstdarg>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <numbers>
#include <sstream>
#include <algorithm>
#include <numeric>

#define MAX_SOURCE_SIZE (0x100000)
FILE* last_run_log_file = nullptr;
FILE* list_of_runs_log_file = nullptr;

void show_status_string(const char* format, ...)
{
	char str[256] = { '\0' };

	va_list args;
	va_start(args, format);

	vsprintf(str, format, args);

	va_end(args);

	// console
	printf("### ");
	puts(str);

	// log file
	fputs(str, last_run_log_file);
	fputc('\n', last_run_log_file);
}


///////// OpenCL FFT 2D function ///////////
int FFT_2D_OpenCL(Cl_Buffer_pair* input_output, clfftDirection direction, cl_command_queue queue, cl_int finishFlag, FFT_OpenCL_data* data)
{

	// заполнение буферов на GPU нулями
	cl_int err = clfftEnqueueTransform(data->getPlanHandle(), direction, 1, &queue, 0, nullptr, nullptr,
		input_output->getBuffers(), input_output->getBuffers(), data->getTmpBuffer());

	// Wait for calculations to be finished
	if (finishFlag == CL_TRUE)
		err = clFinish(queue);

	return err;
}



Cl_Buffer_pair read_and_fft_pics(cl_context ctx, cl_command_queue queue, int amount_of_pics, int rash_extent)
{
	cl_int err;
	Cl_Buffer_pair all_pics_buffer;
	FFT_OpenCL_data fft_rash_size;
	size_t rash_dim_of_picture = rash_extent * rash_extent;
	std::vector<float> Array(rash_dim_of_picture, 0.0f);

	clock_t creation_of_helpers_time_start = clock();
	all_pics_buffer.Init(ctx, queue, CL_MEM_READ_WRITE, rash_dim_of_picture * amount_of_pics);

	fft_rash_size.Init(rash_extent, rash_extent, ctx, queue, amount_of_pics, CLFFT_BACKWARD);
	clock_t creation_of_helpers_time_end = clock();
	show_status_string("Time for initiating buffer(helpers) for pics: %f", (float)(creation_of_helpers_time_end - creation_of_helpers_time_start) / CLOCKS_PER_SEC);

	clock_t  sumtime = 0;

	const size_t pic_size_in_bytes = rash_dim_of_picture * sizeof(cl_float);

	for (int i = 0; i < amount_of_pics; i++)
	{
		clock_t start_time_load_pic = clock();
		char filename[64] = { '\0' };
		sprintf(filename, "%dx%d\\image%02d.png", fft_rash_size.getSizeX() / 2, fft_rash_size.getSizeY() / 2, i + 1);
		printf("### filename: %s\n", filename);

		Image image = Image::readFromPngFile(filename);
		

		if (image.getRowPointers().empty() || image.getWidth() * 2 != fft_rash_size.getSizeX() || image.getHeight() * 2 != fft_rash_size.getSizeY())
			return Cl_Buffer_pair();

		for (int l = 0; l < image.getHeight(); l++)
		{
			const unsigned char* p_row = image.getRowPointers()[l];
			for (int p = 0; p < image.getWidth(); p++)
				Array[l * image.getWidth() * 2 + p] = p_row[p];
		}

		err = clEnqueueWriteBuffer(queue, all_pics_buffer.getBuffers()[0], CL_TRUE, pic_size_in_bytes * i,
			pic_size_in_bytes, Array.data(), 0, nullptr,nullptr);


		image = {};
		if (err != CL_SUCCESS)
		{
			printf("Error with pics[%d].buffers[0] clEnqueueWriteBuffer\n", i);
			return Cl_Buffer_pair();
		}

		clock_t tmpTime = clock() - start_time_load_pic;
		sumtime += tmpTime;
		printf("### %d loaded pic: %f seconds", i + 1, (float)tmpTime / CLOCKS_PER_SEC);
		printf("\n");
	}

	/// Прямое ПФ для КАРТИНОК

	clock_t fft_start = clock();
	if (FFT_2D_OpenCL(&all_pics_buffer, CLFFT_FORWARD, queue, CL_TRUE, &fft_rash_size) == 0)
	{

		clock_t fft_end = clock();
		printf("### all pics fft: %f seconds\n", (float)(fft_end - fft_start) / CLOCKS_PER_SEC);
	}
	else
	{
		printf("Problems w/ FFT\n");
		return Cl_Buffer_pair();
	}

	printf("\n");
	return all_pics_buffer;
}

cl_program init_kernel_program(cl_context ctx, cl_device_id device, int amount_of_pics, float mu, int iters)
{
	// Execute the OpenCL kernel on the list
	cl_int ret;
	FILE* rash_kernel;
	std::vector<char> source_str(MAX_SOURCE_SIZE);

	/// Открываем файл с kernel на чтение
	rash_kernel = fopen("rash_kernel.cl", "r");
	if (!rash_kernel)
	{
		fprintf(stderr, "Failed to load kernel.\n");
		return 0;
	}
	fread(source_str.data(), 1, MAX_SOURCE_SIZE, rash_kernel);
	fclose(rash_kernel);

	// Create program
	const char* tmp_str = source_str.data();
	cl_program program = clCreateProgramWithSource(ctx, 1, &tmp_str, nullptr, &ret);

	// Build the program
	std::ostringstream str;
	str << "-DITERS=" << iters;
	str << " -DMU=" << std::fixed << mu << "f";
	str << " -DAMOUNT_OF_PICS=" << amount_of_pics;

	str << " -w -Werror";


	ret = clBuildProgram(program, 1, &device, str.str().data(), nullptr, nullptr);

	/// в случае ошибок выводим log
	if (ret != CL_SUCCESS)
	{
		printf("Problems w/ building program\n");
		size_t source_size_log;
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &source_size_log);
		std::vector<char> log(source_size_log);
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, source_size_log, log.data(), &source_size_log);
		printf("%s\n", log.data());
		clReleaseProgram(program);
		return 0;
	}

	return program;
}

template<typename T>
void setKernelArg (const cl_kernel kernel, const int arg_index, const T &arg_value, const char *p_arg_name)
{
	const cl_int result = clSetKernelArg(kernel, arg_index, sizeof(arg_value), &arg_value);
	if (result != CL_SUCCESS)
	{
		char kernel_name[128] = {};
		clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, std::size(kernel_name), kernel_name, nullptr);

		printf("Problems w/ setting arg '%s' to kernel '%s'\n", p_arg_name, kernel_name);
	}
}


int main(void)
{

	cl_int err;
	cl_int ret;
	cl_platform_id platform = 0;
	cl_uint number_of_devices = 0;

	cl_context ctx = 0;
	cl_command_queue queue = 0;

	// Setup OpenCL environment
	err = clGetPlatformIDs(1, &platform, nullptr);

	// получение "устройства", котором будет выполнятся вычисление
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU, 0, nullptr, &number_of_devices);


	printf("### Amount of devices: %u\n", number_of_devices);
	printf("### List of devices:\n");

	char name[128] = { '\0' };
	std::vector<cl_device_id> devices(number_of_devices);
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU, number_of_devices, devices.data(), nullptr);
	for (int i = 0; i < number_of_devices; i++)
	{
		err = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(name), name, nullptr);
		printf("\t\t[%d]%s\n", i, name);
	}
	printf("\n");

	// выбираем нужный девайс
	int ptr = -1;

	while (ptr >= number_of_devices || ptr < 0)
	{
		printf("Choose device: ");
		scanf("%d", &ptr);
		printf("\n");
	}
	cl_device_id device = devices[ptr];
	devices.clear();
	devices.shrink_to_fit();

	char version[128] = { '\0' };
	err = clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(version), version, nullptr);

	int major_version = 0;
	int minor_version = 0;

	sscanf(version, "OpenCL %d.%d", &major_version, &minor_version);

	if (major_version <= 1 && minor_version < 2)
	{
		printf("Program requires OpenCL 1.2 and higher\n");
		exit(1);
	}

	int extent = 0;
	printf("Choose image size (like 512, 1024 etc): ");
	scanf("%d", &extent);
	printf("\n");

	int amount_of_pics = 0;

	while (amount_of_pics < 1)
	{
		printf("Choose amount of pics: ");
		scanf("%d", &amount_of_pics);
		printf("\n");
	}

	float mu = 0;
	printf("Choose MU: ");
	scanf("%f", &mu);
	printf("\n");

	int iters = 0;
	printf("Choose iters: ");
	scanf("%d", &iters);
	printf("\n");

	clock_t time_start_program = clock();

	char buff[100];
	time_t now = time(0);
	strftime(buff, 100, "%Y-%m-%d I %H-%M-%S", localtime(&now));

	char str_name_of_log_file[128];
	sprintf(str_name_of_log_file, "log_file I %d I %d I %s.txt", extent, amount_of_pics, buff);
	last_run_log_file = fopen(str_name_of_log_file, "wb");

	fprintf(last_run_log_file, "Amount of devices: %u\n", number_of_devices);
	err = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, nullptr);
	fprintf(last_run_log_file, "You chose this device: [%s]\n", name);
	fprintf(last_run_log_file, "Your OpenCL version: %s\n", version);
	fprintf(last_run_log_file, "You chose image size: %dx%d\n", extent, extent);
	fprintf(last_run_log_file, "You chose this amount of pics: %d\n", amount_of_pics);

	cl_ulong device_memsize_in_bytes = 0;
	err = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(device_memsize_in_bytes), &device_memsize_in_bytes, nullptr);

	show_status_string("GPU mem space: %llu MB", device_memsize_in_bytes / ((cl_ulong)1024 * (cl_ulong)1024));

	cl_ulong min_memsize_in_bytes_required = (cl_ulong)(extent * extent) * sizeof(float) * (36 * amount_of_pics + 4* amount_of_pics* amount_of_pics + 7);
	if (min_memsize_in_bytes_required >= device_memsize_in_bytes)
	{
		printf("### Not enough GPU memory\n");
		printf("### Min required GPU mem space: %llu MB\n", min_memsize_in_bytes_required / ((cl_ulong)1024 * (cl_ulong)1024));
		return 1;
	}

	// Create an OpenCL context
	ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);

	// Create a command queue
	queue = clCreateCommandQueue(ctx, device, 0, &err);

	show_status_string("Initializing FFT library...");
	// Setup clFFT
	clfftSetupData fftSetup;
	err = clfftInitSetupData(&fftSetup);
	err = clfftSetup(&fftSetup);

	int rash_extent = extent * 2;

	// Total size of FFT ( размер расширенных матриц )
	size_t rash_dim_of_picture = rash_extent * rash_extent;
	// исходный размер картинки ( используется только для h )
	size_t dim_of_picture = extent * extent;



	cl_program program = init_kernel_program(ctx, device, amount_of_pics, mu, iters);
	if (program == 0)
	{
		clfftTeardown(); // Release clFFT library
		clReleaseCommandQueue(queue); // Release OpenCL working objects
		clReleaseContext(ctx);
		fclose(last_run_log_file);
		exit(1);
	}

	///=================================================================

	/// НАЧАЛО РАБОТЫ С КАРТИНКОЙ
	Cl_Buffer_pair all_pics_buffer;

	show_status_string("Reading and FFT-ing input pics...");
	clock_t start = clock();
	all_pics_buffer = read_and_fft_pics(ctx, queue, amount_of_pics, rash_extent);

	printf("### Reading and fft'ing pics ends in: %f seconds\n", (float)(clock() - start) / CLOCKS_PER_SEC);
	if (all_pics_buffer.getBuffers()[0] == 0)
	{
		clfftTeardown(); // Release clFFT library
		clReleaseCommandQueue(queue); // Release OpenCL working objects
		clReleaseProgram(program);
		clReleaseContext(ctx);
		fclose(last_run_log_file);
		exit(1);
	}

	/// КОНЕЦ РАБОТЫ С КАРТИНКОЙ


	/// НАЧАЛО РАБОТЫ С h

	// кол-во картинок равно 3 => amount_of_pics = 3;
	int amount_of_h = amount_of_pics;

	clock_t start_h_CL_time = clock();
	FFT_OpenCL_data fft_orig_size;
	err = fft_orig_size.Init(extent, extent, ctx, queue, 1, CLFFT_BACKWARD);

	// Создаем пару буферов для h размером исходной картинки и h расширенной
	Cl_Buffer_pair all_h_rash_CL_buffer;

	Cl_Buffer_pair h_CL_k;

	show_status_string("Init buffer for h_original_size and h_extended_size");

	all_h_rash_CL_buffer.Init(ctx, queue, CL_MEM_READ_WRITE, rash_dim_of_picture * amount_of_h);

	h_CL_k.Init(ctx, queue, CL_MEM_READ_WRITE, dim_of_picture);

	// Создаем kernel для инициализации h и передаем туда аргументы ( delta_z и два буфера для вещественной и мнимой части )
	cl_kernel h_init_kernel = clCreateKernel(program, "h_init_kernel", &ret);
	cl_kernel h_squared_abs_kernel = clCreateKernel(program, "h_squared_abs_kernel", &ret);
	cl_kernel fft_shift_row_kernel = clCreateKernel(program, "fft_shift_row_kernel", &ret);
	cl_kernel fft_shift_col_kernel = clCreateKernel(program, "fft_shift_col_kernel", &ret);


	setKernelArg(h_init_kernel, 1, h_CL_k.getBuffers()[0], "h[0]");
	setKernelArg(h_init_kernel, 2, h_CL_k.getBuffers()[1], "h[1]");

	setKernelArg(h_squared_abs_kernel, 0, h_CL_k.getBuffers()[0], "h[0]");
	setKernelArg(h_squared_abs_kernel, 1, h_CL_k.getBuffers()[1], "h[1]");

	// Генерация h
	for (int k = 0; k < amount_of_h; k++)
	{
		const float delta_z = k * std::numbers::pi_v<float>;

		setKernelArg(h_init_kernel, 0, delta_z, "delta_z");

		size_t global_group_size[] = { extent, extent };
		/// Кладем в очередь команды для вызова kernel, который создает матрицу h размерами исходной картинки

		ret = clEnqueueNDRangeKernel(queue, h_init_kernel, 2, nullptr, global_group_size, nullptr, 0, nullptr, nullptr);
		if (ret != CL_SUCCESS)
			printf("Problems w/ clEnqueueNDRangeKernel h_init_kernel");
		ret = clFinish(queue);
		if (ret != CL_SUCCESS)
			printf("Problems w/ clFinish");



		show_status_string("Making FFT for h_original_size");
		/// Прямое ПФ для h
		if (FFT_2D_OpenCL(&h_CL_k, CLFFT_FORWARD, queue, CL_TRUE, &fft_orig_size) == 0)
			;
		else
			printf("FFT for h func NOT passed !\n");

		/// FFTShift для h

		for (int i = 0; i < 2; i++)
		{
			const int offset_local = 0;

			setKernelArg(fft_shift_row_kernel, 0, h_CL_k.getBuffers()[i], "h[i]");
			setKernelArg(fft_shift_row_kernel, 1, extent, "extent");
			setKernelArg(fft_shift_row_kernel, 2, offset_local, "offset_local");

			size_t sizey_t = extent;
			ret = clEnqueueNDRangeKernel(queue, fft_shift_row_kernel, 1, nullptr, &sizey_t, nullptr, 0, nullptr, nullptr);
			if (ret != CL_SUCCESS)
				printf("Problems w/ clEnqueueNDRangeKernel fft_shift_row_kernel");
			ret = clFinish(queue);
			if (ret != CL_SUCCESS)
				printf("Problems w/ clFinish");

			setKernelArg(fft_shift_col_kernel, 0, h_CL_k.getBuffers()[i], "h[i]");
			setKernelArg(fft_shift_col_kernel, 1, extent, "extent");
			setKernelArg(fft_shift_col_kernel, 2, offset_local, "offset_local");

			size_t sizex_t = extent;
			ret = clEnqueueNDRangeKernel(queue, fft_shift_col_kernel, 1, nullptr, &sizex_t, nullptr, 0, nullptr, nullptr);
			if (ret != CL_SUCCESS)
				printf("Problems w/ clEnqueueNDRangeKernel fft_shift_col_kernel");
			ret = clFinish(queue);
			if (ret != CL_SUCCESS)
				printf("Problems w/ clFinish");

		}

		/// Модуль для h^2

		size_t dim_of_picture_local = dim_of_picture;
		ret = clEnqueueNDRangeKernel(queue, h_squared_abs_kernel, 1, nullptr, &dim_of_picture_local, nullptr, 0, nullptr, nullptr);
		if (ret != CL_SUCCESS)
			printf("Problems w/ clEnqueueNDRangeKernel h_squared_abs_kernel");
		ret = clFinish(queue);
		if (ret != CL_SUCCESS)
			printf("Problems w/ clFinish");

		// Расширяем матрицу h ( теперь она становится h_rash )
		for (int i = 0; i < 2; i++)
		{
			for (int j = 0; j < extent; j++)
			{
				err = clEnqueueCopyBuffer(queue, h_CL_k.getBuffers()[i], all_h_rash_CL_buffer.getBuffers()[i],
					j * extent * sizeof(cl_float), (k * rash_dim_of_picture + j * rash_extent) * sizeof(cl_float),
					extent * sizeof(cl_float), 0, nullptr, nullptr);
				if (err != CL_SUCCESS)
				{
					printf("Error with clEnqueueCopyBuffer %d\n", j);
					fclose(last_run_log_file);
					clReleaseProgram(program);
					clfftTeardown(); // Release clFFT library
					clReleaseCommandQueue(queue); // Release OpenCL working objects
					clReleaseContext(ctx);
					return err;
				}
			}
		}

		ret = clFinish(queue);
		if (ret != CL_SUCCESS)
			printf("Problems w/ clFinish after copy");
	}
	
	std::vector<cl_float> h_real(rash_dim_of_picture, 0);
	
	for (int i = 0; i < amount_of_h; i++)
	{
		ret = clEnqueueReadBuffer(
			queue, all_h_rash_CL_buffer.getBuffers()[0], CL_TRUE, rash_dim_of_picture * i * sizeof(cl_float), rash_dim_of_picture * sizeof(float),
			h_real.data(), 0, NULL, NULL);
		
		const double sum_real = std::accumulate(h_real.begin(), h_real.end(), 0.0);

		for (int k = 0; k < rash_dim_of_picture; k++)
			h_real[k] = cl_float(h_real[k] / sum_real);

		err = clEnqueueWriteBuffer(
			queue, all_h_rash_CL_buffer.getBuffers()[0], CL_TRUE, rash_dim_of_picture * i * sizeof(cl_float), rash_dim_of_picture * sizeof(float),
			h_real.data(), 0, nullptr, nullptr);
		
	}
	

	clReleaseKernel(h_init_kernel);
	clReleaseKernel(h_squared_abs_kernel);

	fft_orig_size = {};
	h_CL_k = {};

	clock_t end_h_CL_time = clock();

	clock_t start_h_CL_fft_time = clock();
	FFT_OpenCL_data fft_rash_size;
	err = fft_rash_size.Init(rash_extent, rash_extent, ctx, queue, amount_of_h, CLFFT_BACKWARD);

	// Прямое ПФ для расширенной матрицы h
	if (FFT_2D_OpenCL(&all_h_rash_CL_buffer, CLFFT_FORWARD, queue, CL_TRUE, &fft_rash_size) == 0)
		;
	else
		printf("FFT for all_h_rash_CL_buffer func NOT passed !\n");

	clock_t end_h_CL_fft_time = clock();



	/// РАБОТА С h ЗАКОНЧЕНА
	float h_gen_time = (float)(end_h_CL_time - start_h_CL_time) / CLOCKS_PER_SEC;
	float h_fft_time = (float)(end_h_CL_fft_time - start_h_CL_fft_time) / CLOCKS_PER_SEC;

	show_status_string("");
	show_status_string("Time for generating h: %f", h_gen_time);
	show_status_string("Time for h fft: %f", h_fft_time);
	show_status_string("Total time for generating and fft'ing h: %f", h_gen_time + h_fft_time);
	show_status_string("");


	std::vector<float> result(rash_dim_of_picture, 0);
	Image image_result(extent, extent);
	for (int i = 0; i < image_result.getHeight(); i++)
		image_result.allocateRow(i, image_result.getWidth() * sizeof(image_result.getRowPointers()[0][0]));

	const clock_t pixel_recovery_start_time = clock();

	cl_kernel pixel_recovery_kernel = clCreateKernel(program, "pixel_recovery_kernel", &ret);

	// __global const float* images_real, __global const float* images_imag
	setKernelArg(pixel_recovery_kernel, 0, all_pics_buffer.getBuffers()[0], "all_pics_buffer[0]");
	setKernelArg(pixel_recovery_kernel, 1, all_pics_buffer.getBuffers()[1], "all_pics_buffer[1]");

	// __global const float* h_real, __global const float* h_imag
	setKernelArg(pixel_recovery_kernel, 2, all_h_rash_CL_buffer.getBuffers()[0], "all_h_rash_CL_buffer[0]");
	setKernelArg(pixel_recovery_kernel, 3, all_h_rash_CL_buffer.getBuffers()[1], "all_h_rash_CL_buffer[1]");

	//__global float2 *tmp_multiply_matrix
	cl_mem tmp_multiply_matrix =
		clCreateBuffer(ctx, CL_MEM_READ_WRITE, amount_of_pics * amount_of_pics * rash_dim_of_picture * sizeof(cl_float2), nullptr, &err);
	setKernelArg(pixel_recovery_kernel, 4, tmp_multiply_matrix, "tmp_multiply_matrix");

	//__global float2 *O
	cl_mem O = clCreateBuffer(ctx, CL_MEM_READ_WRITE, amount_of_pics * rash_dim_of_picture * sizeof(cl_float2), nullptr, &err);
	setKernelArg(pixel_recovery_kernel, 5, O, "O");

	//__global float2 *prepared_vec_of_input_pixels
	cl_mem prepared_vec_of_input_pixels = clCreateBuffer(ctx, CL_MEM_READ_WRITE, amount_of_pics * rash_dim_of_picture * sizeof(cl_float2), nullptr, &err);
	setKernelArg(pixel_recovery_kernel, 6, prepared_vec_of_input_pixels, "prepared_vec_of_input_pixels");

	//__global float2* tmp_multiply_vec
	cl_mem tmp_multiply_vec = clCreateBuffer(ctx, CL_MEM_READ_WRITE, amount_of_pics * rash_dim_of_picture * sizeof(cl_float2), nullptr, &err);
	setKernelArg(pixel_recovery_kernel, 7, tmp_multiply_vec, "tmp_multiply_vec");

	//__global float* result_real, __global float* result_imag)
	Cl_Buffer_pair all_pics_result_CL;
	all_pics_result_CL.Init(ctx, queue, CL_MEM_WRITE_ONLY, amount_of_pics * rash_dim_of_picture);
	setKernelArg(pixel_recovery_kernel, 8, all_pics_result_CL.getBuffers()[0], "all_pics_result_CL[0]");
	setKernelArg(pixel_recovery_kernel, 9, all_pics_result_CL.getBuffers()[1], "all_pics_result_CL[1]");

	size_t global_group_size[] = { rash_extent, rash_extent };

	ret = clEnqueueNDRangeKernel(queue, pixel_recovery_kernel, std::size(global_group_size), nullptr, global_group_size, nullptr, 0, nullptr, nullptr);
	if (ret != CL_SUCCESS)
		printf("Problems w/ clEnqueueNDRangeKernel pixel_recovery_kernel");
	ret = clFinish(queue);
	if (ret != CL_SUCCESS)
		printf("Problems w/ clFinish");

	clReleaseKernel(pixel_recovery_kernel);

	for (cl_mem tmp_mem : {tmp_multiply_matrix, O, prepared_vec_of_input_pixels, tmp_multiply_vec})
		clReleaseMemObject(tmp_mem);

	if (FFT_2D_OpenCL(&all_pics_result_CL, CLFFT_BACKWARD, queue, CL_TRUE, &fft_rash_size) == 0)
		;
	else
		printf("IFFT for all_pics_result_CL func NOT passed !\n");

	for (int i = 0; i < amount_of_pics; i++)
	{
		const auto offset_local = int(rash_dim_of_picture) * i;

		setKernelArg(fft_shift_row_kernel, 0, all_pics_result_CL.getBuffers()[0], "all_pics_result_CL[0]");
		setKernelArg(fft_shift_row_kernel, 1, rash_extent, "rash_extent");
		setKernelArg(fft_shift_row_kernel, 2, offset_local, "offset_local");

		size_t sizey_t = rash_extent;
		ret = clEnqueueNDRangeKernel(queue, fft_shift_row_kernel, 1, nullptr, &sizey_t, nullptr, 0, nullptr, nullptr);
		if (ret != CL_SUCCESS)
			printf("Problems w/ clEnqueueNDRangeKernel fft_shift_row_kernel");
		ret = clFinish(queue);
		if (ret != CL_SUCCESS)
			printf("Problems w/ clFinish");

		setKernelArg(fft_shift_col_kernel, 0, all_pics_result_CL.getBuffers()[0], "all_pics_result_CL[0]");
		setKernelArg(fft_shift_col_kernel, 1, rash_extent, "rash_extent");
		setKernelArg(fft_shift_col_kernel, 2, offset_local, "offset_local");

		size_t sizex_t = rash_extent;
		ret = clEnqueueNDRangeKernel(queue, fft_shift_col_kernel, 1, nullptr, &sizex_t, nullptr, 0, nullptr, nullptr);
		if (ret != CL_SUCCESS)
			printf("Problems w/ clEnqueueNDRangeKernel fft_shift_col_kernel");
		ret = clFinish(queue);
		if (ret != CL_SUCCESS)
			printf("Problems w/ clFinish");
	}

	fft_rash_size = {};

	const clock_t pixel_recovery_end_time = clock();

	show_status_string("");
	show_status_string("Time for pixel recovery (all pics): %f", double(pixel_recovery_end_time - pixel_recovery_start_time) / double(CLOCKS_PER_SEC));
	show_status_string("");

	for (int i = 0; i < amount_of_pics; i++)
	{
		ret =
			clEnqueueReadBuffer(
				queue, all_pics_result_CL.getBuffers()[0], CL_TRUE, rash_dim_of_picture * i * sizeof(cl_float), rash_dim_of_picture * sizeof(cl_float),
				result.data(), 0, NULL, NULL);

		
		auto minmax_iter_pair = std::minmax_element(result.begin(), result.end());

		// Запись картинки в файл
		for (int k = 0; k < image_result.getHeight(); k++)
		{
			for (int l = 0; l < image_result.getWidth(); l++)
			{
				double res = result[(k + image_result.getHeight() / 2) * rash_extent + (l + image_result.getWidth() / 2)];
				res -= (result[0] - *minmax_iter_pair.first);
				res = (res - *minmax_iter_pair.first) / double(*minmax_iter_pair.second - *minmax_iter_pair.first) * 255.0;
				image_result.getRowPointers()[k][l] = (unsigned char)std::clamp(res, 0.0, 255.0);
			}
		}

		char filename_png[64] = { '\0' };
		sprintf(filename_png, "result/image%02d.png", i + 1);
		show_status_string("Writing data to file");

		image_result.writeToPngFile(filename_png);
	}


	printf("### Cleaning...\n");

	// fputc('\n', list_of_runs_log_file);
	fclose(last_run_log_file);
	clReleaseKernel(fft_shift_col_kernel);
	clReleaseKernel(fft_shift_row_kernel);
	clReleaseProgram(program);
	clfftTeardown(); // Release clFFT library
	clReleaseCommandQueue(queue); // Release OpenCL working objects
	clReleaseContext(ctx);


	clock_t time_end_program = clock();

	strftime(buff, 100, "%d-%m-%Y %H:%M:%S", localtime(&now));
	list_of_runs_log_file = fopen("list_of_runs_log_file.txt", "a");
	err = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, nullptr);

	if (ftell(list_of_runs_log_file) == 0)
		fprintf(list_of_runs_log_file, "|%-20s |%-19s |%-22s |%-15s |%-15s |%-15s\n", "Date", "time(recovery)", "full time of program", "Size", "Amount of pics", "Device");
 
	fprintf(list_of_runs_log_file, "|%-20s |%-19f |%-22f |%-15d |%-15d |%-15s\n\n", buff, double(pixel_recovery_end_time - pixel_recovery_start_time) / double(CLOCKS_PER_SEC), (float)(time_end_program - time_start_program) / CLOCKS_PER_SEC, extent, amount_of_pics, name);

	fclose(list_of_runs_log_file);
	return 0;
}