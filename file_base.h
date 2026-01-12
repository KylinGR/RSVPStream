#ifndef _FILE_BASE__
#define _FILE_BASE__
 
 
#include <string>
#include <fstream>
#include <vector>
#include <sys/stat.h>
//template<typename ... Args>
//std::string strFormat(const std::string& format, Args ... args) {
//	size_t size = 1 + snprintf(nullptr, 0, format.c_str(), args ...);  // Extra space for \0
//	std::string bytes;
//	snprintf(bytes.c_str(), size, format.c_str(), args ...);
//	return bytes;
//}


typedef enum _FILE_TYPE {
    BINARY=0,
    TXT=1
}FILE_TYPE;
typedef enum _FILE_DIR {
	FIN=0,
	FOUT
}FILE_DIR;

class file_base
{
public:
	file_base(std::string& strFileName, FILE_DIR eDir = FILE_DIR::FIN, FILE_TYPE eFileType = FILE_TYPE::BINARY);
	int32_t read(char* arrOut, uint32_t size);
	uint64_t readAll(char* arrOut);
	uint64_t readAll(std::vector<uint32_t>& arrOut);
	uint64_t readAll(std::vector<uint16_t>& arrOut);
    void readLine(std::string& strOut);
	void  write(char* arrIn, uint32_t siz);
    void  write(std::string& strIn );
	// uint64_t FileSize();
	void close();
	~file_base();
private:
	std::fstream file;
	std::string strFile;
	std::ios_base::openmode file_mode;
	uint64_t file_size;

};

#endif