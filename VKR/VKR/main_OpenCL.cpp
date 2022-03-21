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
			for (int p = 0; p < image.getWidth(); p++)
				Array[l * fft_rash_size.getSizeY() + p] = image.getRowPointers()[l][p];
		}

		cl_event write_future = 0;
		err = clEnqueueWriteBuffer(queue, all_pics_buffer.getBuffers()[0], CL_FALSE, pic_size_in_bytes * i,
			pic_size_in_bytes, Array.data(), 0, nullptr, &write_future);


		image = {};
		if (err != CL_SUCCESS)
		{
			printf("Error with pics[%d].buffers[0] clEnqueueWriteBuffer\n", i);
			clReleaseEvent(write_future);
			return Cl_Buffer_pair();
		}
		err = clWaitForEvents(1, &write_future);
		cl_int err1 = clReleaseEvent(write_future);
		if (err != CL_SUCCESS || err1 != CL_SUCCESS)
		{
			printf("ERROR with events\n");
			return Cl_Buffer_pair();
		}

		clock_t tmpTime = clock() - start_time_load_pic;
		sumtime += tmpTime;
		printf("### %d loaded pic: %f seconds", i + 1, (float)tmpTime / CLOCKS_PER_SEC);
		printf("\n");
	}

	/// Прямое ПФ для КАРТИНК

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

cl_program init_kernel_program(cl_context ctx, cl_device_id device)
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
	ret = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

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



//// СКОЛЬКО ПАМЯТИ ТРАТИТСЯ ////
// x^2 - размер одой картинки в пикселях ( оригинальный )
// тк мы работаем с раширенными матрицами => (2x)^2 - размер одной картинки в пикселях ( расширенный )
// переводим в байты => (2x)^2 * sizeof(float) - размер одной картинки в байтах ( расширенный )
// а тк у нас есть и мнимая часть => (2x)^2 * sizeof(float) * 2 - размер Фурье-образа одной картики в байтах ( расширенный )
// тк у нас h_rash_CL тоже имеет размер (2x)^2 * sizeof(float) * 2 => (2x)^2 * sizeof(float) * 2 * 2 - минимальный необходимый объем памяти на GPU
// также нужно место для части результата ( матрица и тут должна быть расширена ) => (2x)^2 * sizeof(float) * 2 (result_part_CL)
// нужно (2x)^2 * sizeof(float) - для итогового результата ( сумма получивших картинок ) (result_CL)

// тогда минимальный объем памяти для N картинок на GPU - (2x)^2 * sizeof(float) * 2 * (2*N + 1) + x^2* sizeof(float)*2  + (2x)^2 * sizeof(float) ( предпоследнее слагаемое - h оригинального размера, последнее - результат )
// x^2 * sizeof(float) * (16N + 14)

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
	std::vector<cl_device_id> devices (number_of_devices);
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

	cl_ulong min_memsize_in_bytes_required = (cl_ulong)(extent * extent) * sizeof(float) * (32 * amount_of_pics + 22);
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

	cl_program program = init_kernel_program(ctx, device);
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

	clock_t start_h_CL_time = clock();
	FFT_OpenCL_data fft_orig_size;
	err = fft_orig_size.Init(extent, extent, ctx, queue, 1, CLFFT_FORWARD);

	// кол-во картинок равно 3 => amount_of_pics = 3;
	int amount_of_h = amount_of_pics;

	/// Создаем пару буферов для h размером исходной картинки и h расширенной
	std::vector<Cl_Buffer_pair> h_rash_CL(amount_of_h);
	Cl_Buffer_pair h_CL_k;

	show_status_string("Init buffer for h_original_size and h_extended_size");

	for (Cl_Buffer_pair& h_rash_CL_i: h_rash_CL)
	{
		h_rash_CL_i.Init(ctx, queue, CL_MEM_READ_WRITE, rash_dim_of_picture);
	}
	h_CL_k.Init(ctx, queue, CL_MEM_READ_WRITE, dim_of_picture);

	// Создаем kernel для инициализации h и передаем туда аргументы ( delta_z и два буфера для вещественной и мнимой части )
	cl_kernel h_init_kernel = clCreateKernel(program, "h_init_kernel", &ret);
	cl_kernel h_squared_abs_kernel = clCreateKernel(program, "h_squared_abs_kernel", &ret);
	cl_kernel fft_shift_row_kernel = clCreateKernel(program, "fft_shift_row_kernel", &ret);
	cl_kernel fft_shift_col_kernel = clCreateKernel(program, "fft_shift_col_kernel", &ret);


	ret = clSetKernelArg(h_init_kernel, 1, sizeof(cl_mem), &h_CL_k.getBuffers()[0]);
	if (ret != CL_SUCCESS)
		printf("Problems w/ setting KernelArgs for h[0] h_init_kernel\n");
	ret = clSetKernelArg(h_init_kernel, 2, sizeof(cl_mem), &h_CL_k.getBuffers()[1]);
	if (ret != CL_SUCCESS)
		printf("Problems w/ setting KernelArgs for h[1] h_init_kernel\n");


	ret = clSetKernelArg(h_squared_abs_kernel, 0, sizeof(cl_mem), &h_CL_k.getBuffers()[0]);
	if (ret != CL_SUCCESS)
		printf("Problems w/ setting KernelArgs for h[0] h_squared_abs_kernel\n");
	ret = clSetKernelArg(h_squared_abs_kernel, 1, sizeof(cl_mem), &h_CL_k.getBuffers()[1]);
	if (ret != CL_SUCCESS)
		printf("Problems w/ setting KernelArgs for h[1] h_squared_abs_kernel\n");

	// Генерация h
	for (int k = 0; k < amount_of_h; k++)
	{

		float delta_z = k * std::numbers::pi_v<float>;

		ret = clSetKernelArg(h_init_kernel, 0, sizeof(delta_z), &delta_z);
		if (ret != CL_SUCCESS)
			printf("Problems w/ setting KernelArgs for delta_z h_init_kernel\n");


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
			ret = clSetKernelArg(fft_shift_row_kernel, 0, sizeof(cl_mem), &h_CL_k.getBuffers()[i]);
			if (ret != CL_SUCCESS)
				printf("Problems w/ setting KernelArgs for h[%d] fft_shift_row_kernel\n", i);
			ret = clSetKernelArg(fft_shift_row_kernel, 1, sizeof(extent), &extent);
			if (ret != CL_SUCCESS)
				printf("Problems w/ setting KernelArgs for h[%d] fft_shift_row_kernel\n", i);

			size_t sizey_t = extent;
			ret = clEnqueueNDRangeKernel(queue, fft_shift_row_kernel, 1, nullptr, &sizey_t, nullptr, 0, nullptr, nullptr);
			if (ret != CL_SUCCESS)
				printf("Problems w/ clEnqueueNDRangeKernel fft_shift_row_kernel");
			ret = clFinish(queue);
			if (ret != CL_SUCCESS)
				printf("Problems w/ clFinish");

			ret = clSetKernelArg(fft_shift_col_kernel, 0, sizeof(cl_mem), &h_CL_k.getBuffers()[i]);
			if (ret != CL_SUCCESS)
				printf("Problems w/ setting KernelArgs for h[%d] fft_shift_col_kernel\n", i);
			ret = clSetKernelArg(fft_shift_col_kernel, 1, sizeof(extent), &extent);
			if (ret != CL_SUCCESS)
				printf("Problems w/ setting KernelArgs for h[%d] fft_shift_col_kernel\n", i);


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
			for (int j = 0; j < extent; j++)
			{
				err = clEnqueueCopyBuffer(queue, h_CL_k.getBuffers()[i], h_rash_CL[k].getBuffers()[i],
					j * extent * sizeof(cl_float), j * rash_extent * sizeof(cl_float),
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

		ret = clFinish(queue);
		if (ret != CL_SUCCESS)
			printf("Problems w/ clFinish after copy");
	}


	clReleaseKernel(h_init_kernel);
	clReleaseKernel(h_squared_abs_kernel);
	clReleaseKernel(fft_shift_col_kernel);
	clReleaseKernel(fft_shift_row_kernel);
	fft_orig_size = {};
	h_CL_k = {};

	clock_t end_h_CL_time = clock();

	clock_t start_h_CL_fft_time = clock();
	FFT_OpenCL_data fft_rash_size;
	err = fft_rash_size.Init(rash_extent, rash_extent, ctx, queue, 1, CLFFT_BACKWARD);

	for (int k = 0; k < amount_of_h; k++)
	{
		// Прямое ПФ для расширенной матрицы h
		if (FFT_2D_OpenCL(&h_rash_CL[k], CLFFT_FORWARD, queue, CL_FALSE, &fft_rash_size) == 0)
			;
		else
			printf("FFT for h_rash[%d] func NOT passed !\n", k);
	}
	ret = clFinish(queue);
	if (ret != CL_SUCCESS)
		printf("Problems w/ clFinish after 2nd FFT for h_rash");

	clock_t end_h_CL_fft_time = clock();


	/// РАБОТА С h ЗАКОНЧЕНА
	float h_gen_time = (float)(end_h_CL_time - start_h_CL_time) / CLOCKS_PER_SEC;
	float h_fft_time = (float)(end_h_CL_fft_time - start_h_CL_fft_time) / CLOCKS_PER_SEC;

	show_status_string("");
	show_status_string("Time for generating h: %f", h_gen_time);
	show_status_string("Time for h fft: %f", h_fft_time);
	show_status_string("Total time for generating and fft'ing h: %f", h_gen_time + h_fft_time);
	show_status_string("");


	/// Умножение картинки и элементов матрицы h_rash

	std::vector<float> result(rash_dim_of_picture, 0);
	Image image_result(extent, extent);
	for (int i = 0; i < image_result.getHeight(); i++)
		image_result.allocateRow(i, image_result.getWidth() * sizeof(image_result.getRowPointers()[0][0]));


	// 1) единичная матрица
	std::vector<float> E(rash_dim_of_picture, 0);
	for (int i = 0; i < rash_extent; i++)
			E[i * rash_extent + i] =  1.0f;
			
	 //2) генерация матрицы из матриц h extent
	


	for (int k = 0; k < image_result.getHeight(); k++)
		for (int l = 0; l < image_result.getWidth(); l++)
		{
			float res = result[(k + image_result.getHeight() / 2) * fft_rash_size.getSizeX() + (l + image_result.getWidth() / 2)];
			image_result.getRowPointers()[k][l] = (char)res;
		}

	char filename_png[64] = { '\0' };
	//sprintf(filename_png, "result/image%02d.png", m + 1);
	//show_status_string("Writing data to file");

	image_result.writeToPngFile(filename_png);


	printf("### Cleaning...\n");

	// fputc('\n', list_of_runs_log_file);
	fclose(last_run_log_file);
	clReleaseProgram(program);
	clfftTeardown(); // Release clFFT library
	clReleaseCommandQueue(queue); // Release OpenCL working objects
	clReleaseContext(ctx);


	clock_t time_end_program = clock();

	strftime(buff, 100, "%d-%m-%Y %H:%M:%S", localtime(&now));
	list_of_runs_log_file = fopen("list_of_runs_log_file.txt", "a");
	err = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, nullptr);

	if (ftell(list_of_runs_log_file) == 0)
		fprintf(list_of_runs_log_file, "I%-20s I%-19s I%-22s I%-15s I%-15s I%-15s\n\n", "Date", "time(multiply+add)", "full time of program", "Size", "Amount of pics", "Device");

	//fprintf(list_of_runs_log_file, "I%-20s I%-19f I%-22f I%-15d I%-15d I%-15s\n", buff, tmp_time_of_calc, (float)(time_end_program - time_start_program) / CLOCKS_PER_SEC, extent, amount_of_pics, name);

	fclose(list_of_runs_log_file);
	return 0;
}