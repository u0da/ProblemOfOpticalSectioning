#pragma once

#include "CL/cl.h"

class Cl_Buffer_pair
{

public:
	cl_int  Init(cl_context ctx, cl_command_queue queue, cl_bitfield mode, size_t N);

	inline ~Cl_Buffer_pair();

	inline cl_mem* getBuffers();

private:
	cl_mem m_buffers[2] = {};
};

Cl_Buffer_pair::~Cl_Buffer_pair()
{
	clReleaseMemObject(m_buffers[0]);
	clReleaseMemObject(m_buffers[1]);
}

cl_mem* Cl_Buffer_pair::getBuffers()
{
	return m_buffers;
}