#pragma once

#include "clFFT.h"
#include <utility>


class FFT_OpenCL_data
{
public:

	FFT_OpenCL_data() = default;
	cl_int Init(int sizex, int sizey, cl_context ctx, cl_command_queue queue, int amount_of_buffers_to_transform, clfftDirection direction_normalize);

	FFT_OpenCL_data& operator = (FFT_OpenCL_data&& rhv) noexcept;

	inline FFT_OpenCL_data(FFT_OpenCL_data&& rhv) noexcept;
	inline ~FFT_OpenCL_data();

	inline int getSizeX() const;
	inline void setSizeX(int sizex);

	inline int getSizeY() const;
	inline void setSizeY(int sizey);

	inline cl_mem getTmpBuffer() const;

	inline clfftPlanHandle getPlanHandle() const;

private:
	void destroy();

private:
	int m_sizex = 0;
	int m_sizey = 0;

	// Temporary buffer
	cl_mem m_tmpBuffer = nullptr;

	// FFT library related declarations
	clfftPlanHandle m_planHandle = 0;
};

FFT_OpenCL_data::FFT_OpenCL_data(FFT_OpenCL_data&& rhv) noexcept
{
	*this = std::move(rhv);
}

FFT_OpenCL_data::~FFT_OpenCL_data()
{
	destroy();
}


int FFT_OpenCL_data::getSizeX() const
{
	return m_sizex;
}
void FFT_OpenCL_data::setSizeX(int sizex)
{
	m_sizex = sizex;
}


int FFT_OpenCL_data::getSizeY() const
{
	return m_sizey;
}
void FFT_OpenCL_data::setSizeY(int sizey)
{
	m_sizey = sizey;
}

cl_mem FFT_OpenCL_data::getTmpBuffer() const
{
	return m_tmpBuffer;
}


clfftPlanHandle FFT_OpenCL_data::getPlanHandle() const
{
	return m_planHandle;
}