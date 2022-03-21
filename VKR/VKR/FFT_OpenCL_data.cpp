#include "FFT_OpenCL_data.h"
#include <cstring>
#include <cmath>
#include <cstdio>

cl_int FFT_OpenCL_data::Init(int sizex, int sizey, cl_context ctx, cl_command_queue queue, int amount_of_buffers_to_transform, clfftDirection direction_normalize)
{
	cl_int err = CL_SUCCESS;
	*this = {};
	setSizeX(sizex);
	setSizeY(sizey);
	int N = sizex * sizey;
	size_t clLengths[2] = { sizex, sizey };
	size_t tmpBufferSize = 0; 

	// Create a default plan for a complex FFT
	err = clfftCreateDefaultPlan(&m_planHandle, ctx, CLFFT_2D, clLengths);
	if (err != CL_SUCCESS)
		return err;

	// Set plan parameters
	err = clfftSetPlanPrecision(m_planHandle, CLFFT_SINGLE);
	if (err != CL_SUCCESS)
		return err;

	err = clfftSetLayout(m_planHandle, CLFFT_COMPLEX_PLANAR, CLFFT_COMPLEX_PLANAR);
	if (err != CL_SUCCESS)
		return err;

	err = clfftSetResultLocation(m_planHandle, CLFFT_INPLACE);
	if (err != CL_SUCCESS)
		return err;


	err = clfftSetPlanBatchSize(m_planHandle, amount_of_buffers_to_transform);
	if (err != CL_SUCCESS)
		return err;

	err = clfftSetPlanDistance(m_planHandle, N, N);
	if (err != CL_SUCCESS)
		return err;

	err = clfftSetPlanScale(m_planHandle, direction_normalize, 1.0f / sqrtf(N));
	if (err != CL_SUCCESS)
		return err;

	// Bake the plan
	err = clfftBakePlan(m_planHandle, 1, &queue, nullptr, nullptr);
	if (err != CL_SUCCESS)
		return err;

	// Create temporary buffer
	err = clfftGetTmpBufSize(m_planHandle, &tmpBufferSize);
	if (err != CL_SUCCESS)
		return err;

	if (tmpBufferSize > 0)
	{
		m_tmpBuffer= clCreateBuffer(ctx, CL_MEM_READ_WRITE, tmpBufferSize, nullptr, &err);
		if (err != CL_SUCCESS)
		{
			printf("Error with tmpBuffer clCreateBuffer\n");
			return err;
		}
	}

	return err;
}

FFT_OpenCL_data& FFT_OpenCL_data::operator = (FFT_OpenCL_data&& rhv) noexcept
{
	destroy();
	this->m_tmpBuffer = rhv.m_tmpBuffer;
	rhv.m_tmpBuffer = nullptr;
	this->m_planHandle = rhv.m_planHandle;
	rhv.m_planHandle = 0;

	return *this;
}

void FFT_OpenCL_data::destroy()
{
	// Release OpenCL memory objects
	clReleaseMemObject(m_tmpBuffer);
	clfftDestroyPlan(&m_planHandle);
	std::memset(this, 0, sizeof(*this));
}