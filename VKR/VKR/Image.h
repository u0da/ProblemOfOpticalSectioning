#pragma once

#include <vector>

class Image
{
public:
	Image() = default;

	inline Image(int width, int height);

	Image& operator = (Image&& rhv) noexcept;
	
	inline Image(Image&& rhv) noexcept;
	
	inline ~Image();


	inline int getWidth() const;
	inline int getHeight() const;
	inline const std::vector<unsigned char*>& getRowPointers() const;
	

	static Image readFromPngFile(const char* file_name);
	void writeToPngFile(const char* file_name);
	void allocateRow(int row_index, std::size_t amount_of_bytes);

private:
	void destroy();

private:
	int m_width = 0;
	int m_height = 0;
	std::vector<unsigned char*> m_row_pointers;
};

Image::Image(int width, int height) :m_width(width), m_height(height)
{
	m_row_pointers.resize(height, nullptr);
}


Image::Image(Image&& rhv) noexcept
{
	*this = std::move(rhv);
}

Image::~Image()
{
	destroy();
}

int Image::getWidth() const
{
	return m_width;
}

int Image::getHeight() const
{
	return m_height;
}

const std::vector<unsigned char*>& Image::getRowPointers() const
{
	return m_row_pointers;
}