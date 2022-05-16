#include "Cl_Buffer_pair.h"
#include <cstdio>
#include <cstring>

cl_int  Cl_Buffer_pair::Init(cl_context ctx, cl_command_queue queue, cl_bitfield mode, size_t N)
{
	const float zero = 0;
	cl_int err = CL_SUCCESS;
	*this = {};

	for (int i = 0; i < 2; i++)
	{
		m_buffers[i] = clCreateBuffer(ctx, mode, N * sizeof(cl_float), nullptr, &err);
		if (err != CL_SUCCESS)
		{
			printf("InitCl_Buffer_pair: Error with buffers[%d] clCreateBuffer\n", i);
			return err;
		}
		err = clEnqueueFillBuffer(queue, m_buffers[i], &zero, sizeof(zero), 0, N * sizeof(float), 0, nullptr, nullptr);
		if (err != CL_SUCCESS)
		{
			printf("InitCl_Buffer_pair: Error with buffers[%d] clEnqueueFillBuffer\n", i);
			return err;
		}

	}
	err = clFinish(queue);
	if (err != CL_SUCCESS)
	{
		printf("InitCl_Buffer_pair: Error with clFinish\n");
		return err;
	}
	return  err;
}

void Cl_Buffer_pair::destroy()
{
	// Release OpenCL memory objects
	clReleaseMemObject(m_buffers[0]);
	clReleaseMemObject(m_buffers[1]);
	std::memset(this, 0, sizeof(*this));
}

Cl_Buffer_pair& Cl_Buffer_pair::operator = (Cl_Buffer_pair&& rhv) noexcept
{
	destroy();
	
	this->m_buffers[0] = rhv.m_buffers[0];
	rhv.m_buffers[0] = 0;

	this->m_buffers[1] = rhv.m_buffers[1];
	rhv.m_buffers[1] = 0;

	return *this;
}