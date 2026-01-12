#include "file_base.h"
#include <iostream>

file_base::file_base(std::string& strFileName, FILE_DIR eDir,FILE_TYPE eFileType)
{
	// std::cout <<strFileName<< ",file open failed" << std::endl;
	strFile =strFileName ;
    if(eFileType==FILE_TYPE::BINARY)
        file_mode = std::ios::binary;
    else
        file_mode = std::ios::app;
	if (eDir == FILE_DIR::FIN)
		file_mode = std::ios::in | file_mode;
	else
		file_mode = std::ios::out | file_mode;

	

	file.open(strFileName.c_str(), file_mode);
	if (file.is_open() == false)
	{
		std::cout << "file open failed" << std::endl;
		exit(-1);
	}

}
uint64_t file_base::readAll(char* arrOut)
{
	int32_t siz;
	int32_t siz_tot =0 ;
	char* pt = arrOut;
	// printf(" file_base::readAll,00..\n");
	while (1)
	{
		siz =  read(pt, 1048576);
		// printf(" file_base::readAll,read size=0x%x\n",siz);
		if (siz <= 0)
		{
			// printf(" file_base::readAll,size=0x%x\n",siz_tot);
			return siz_tot;
		}
		pt += siz;
		siz_tot += siz;
	}
}

uint64_t file_base::readAll(std::vector<uint32_t>& arrOut)
{
	int32_t siz;
	int32_t siz_tot =0 ;
 
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::cout<<"file size="<<size<<std::endl;
    // std::vector<uint32_t> buffer(size / sizeof(uint32_t));
	arrOut.resize((size + sizeof(uint32_t) - 1) / sizeof(uint32_t)); // 向上取整
    file.read(reinterpret_cast<char*>(arrOut.data()), size);
    
    // file.close();
    return size;
}


uint64_t file_base::readAll(std::vector<uint16_t>& arrOut)
{
	int32_t siz;
	int32_t siz_tot =0 ;
 
    file.seekg(0, std::ios::end);
    size_t size = file.tellg(); 
    file.seekg(0, std::ios::beg);
    std::cout<<"file size="<<size<<std::endl;
 
	arrOut.resize((size + sizeof(uint16_t) - 1) / sizeof(uint16_t)); // 向上取整
    file.read(reinterpret_cast<char*>(arrOut.data()), size);
    
    // file.close();
    return size;
}

int32_t file_base::read(char* arrOut,uint32_t size)
{
	if ((file_mode & std::ios::in)==0)
	{
		std::cout << "out file can't read!" << std::endl;
		return -1;
	}
	if (file.eof())
	{
		// std::cout << "file end!" << std::endl;
		return -1;
	}
	file.read( arrOut, size *sizeof(uint8_t));
	int32_t u32Siz = file.gcount();
	if (u32Siz < (size * sizeof(uint8_t)))
	{
		// std::cout << "read file end!u32Siz="<<u32Siz<<",size="<<size << std::endl;

	}
	return u32Siz;
}

void file_base::readLine(std::string& strOut)
{
    if ((file_mode & std::ios::in)==0)
    {
        std::cout << "out file can't read!" << std::endl;
        exit(-1);
    }
    if (file.eof())
    {
        std::cout << "file end!" << std::endl;
        return  ;
    }
    if((file_mode& std::ios::binary) !=0)
    {
        std::cout << "file end!" << std::endl;
        exit(-1);
    }
    file>> strOut;
}

void  file_base::write(char* arrIn, uint32_t siz)
{
	if ((file_mode & std::ios::out) == 0)
	{
		std::cout << "in file can't write!" << std::endl;
        exit(-1);
	}
	file.write((char*)arrIn, siz);
}

void  file_base::write(std::string& strIn )
{
    if ((file_mode & std::ios::out) == 0)
    {
        std::cout << "in file can't write!" << std::endl;
        exit(-1);
    }
    if ((file_mode & std::ios::binary) != 0)
    {
        std::cout << "txt dat can't write to binary file !" << std::endl;
        exit(-1);
    }
    file<<strIn;
}
// uint64_t  file_base::FileSize()
// {
//     struct stat st;
//     if (stat(strFile.c_str(), &st) == 0) {
// 		// std::cout << "stat..." << std::endl;
//         file_size = static_cast<uint64_t>(st.st_size); // 直接返回字节数
//     }
// 	return file_size;
// }
void file_base::close()
{
	// printf("~file_base..0\n");
	file.close();
	// printf("~file_base..1\n");
}
file_base::~file_base()
{
	// printf("~file_base\n");
	close();
}
