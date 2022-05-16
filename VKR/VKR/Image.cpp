#include "Image.h"

#include "FreeImage.h"

#include <memory>

Image Image::readFromPngFile(const char* file_name)
{
	auto destroyBitmap = [](FIBITMAP* p_bitmap) { FreeImage_Unload(p_bitmap); };
	std::unique_ptr<FIBITMAP, decltype(destroyBitmap)> bitmap_ptr(FreeImage_Load(FIF_PNG, file_name, PNG_DEFAULT), destroyBitmap);
	
	if (bitmap_ptr == nullptr)
	{
		printf("[readFromPngFile] File %s is not a valid PNG\n", file_name);
		return Image();
	}

	if (FreeImage_GetBPP(bitmap_ptr.get()) != 8)
	{
		printf("[readFromPngFile] File %s. Only grayscale images are supported\n", file_name);
		return Image();
	}
	
	Image image(FreeImage_GetWidth(bitmap_ptr.get()), FreeImage_GetHeight(bitmap_ptr.get()));

	const auto row_size_in_bytes = FreeImage_GetLine(bitmap_ptr.get());
	for (int y = 0; y < image.m_height; y++)
	{
		image.allocateRow(y, row_size_in_bytes);
		memcpy(image.m_row_pointers[y], FreeImage_GetScanLine(bitmap_ptr.get(), y), row_size_in_bytes);
	}

	return image;
}

void Image::writeToPngFile (const char* file_name)
{
	auto destroyBitmap = [](FIBITMAP* p_bitmap) { FreeImage_Unload(p_bitmap); };
	std::unique_ptr<FIBITMAP, decltype(destroyBitmap)> bitmap_ptr(FreeImage_Allocate(m_width, m_height, 8), destroyBitmap);

	const auto row_size_in_bytes = FreeImage_GetLine(bitmap_ptr.get());
	for (int y = 0; y < m_height; y++)
		memcpy(FreeImage_GetScanLine(bitmap_ptr.get(), y), m_row_pointers[y], row_size_in_bytes);

	if (!FreeImage_Save(FIF_PNG, bitmap_ptr.get(), file_name, PNG_DEFAULT))
		printf("[writeToPngFile] Problems w/ saving file: %s\n", file_name);
}


void Image::allocateRow(int row_index, std::size_t amount_of_bytes) 
{
	m_row_pointers[row_index] = new unsigned char[amount_of_bytes];
}

Image& Image::operator = (Image&& rhv) noexcept
{
	destroy();
	this->m_row_pointers = std::move(rhv.m_row_pointers);
	this->m_height = rhv.m_height;
	this->m_width = rhv.m_width;
	rhv.destroy();

	return *this;
}

void Image::destroy()
{
	for (unsigned char* one_row : m_row_pointers)
		delete[] one_row;
	m_row_pointers.clear();
	m_width = 0;
	m_height = 0;
}