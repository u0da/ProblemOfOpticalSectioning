#pragma once

#include "CL/cl.h"
#include <utility>

class Cl_Buffer_pair
{

public:
	Cl_Buffer_pair() = default;
	inline Cl_Buffer_pair(Cl_Buffer_pair&& rhv) noexcept;

	cl_int  Init(cl_context ctx, cl_command_queue queue, cl_bitfield mode, size_t N);

	inline ~Cl_Buffer_pair();

	inline cl_mem* getBuffers();

	Cl_Buffer_pair& operator = (Cl_Buffer_pair&& rhv) noexcept;

private:
	void destroy();

private:
	cl_mem m_buffers[2] = {};
};

Cl_Buffer_pair::~Cl_Buffer_pair()
{
	destroy();
}

cl_mem* Cl_Buffer_pair::getBuffers()
{
	return m_buffers;
}


Cl_Buffer_pair::Cl_Buffer_pair(Cl_Buffer_pair&& rhv) noexcept
{
	*this = std::move(rhv);
}

