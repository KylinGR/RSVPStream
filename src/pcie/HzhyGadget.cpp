/*
 * hzhy_cal.c
 *
 *  Created on: Jan 17, 2022
 *      Author: ZhaoBinBin
 */

#include"HzhyGadget.h"


/*
 *
 */
uint64_t HzhyGetNowUs()
{
    struct timespec time = {0, 0};
    clock_gettime(CLOCK_MONOTONIC, &time);
    return (uint64_t)time.tv_sec * 1000000 + (uint64_t)time.tv_nsec / 1000; /* microseconds */
}


/*
 *
 */
void *HzhyMalloc(uint32_t len)
{
	void *base;

	base = malloc(len);
	if(base == NULL)
	{
		//dzlog_error("Failed to malloc!");
		return NULL;
	}
	memset(base, 0, len);

	return base;
}



/******************************************************************
 * 函数:	void DumpAddrLen(unsigned char *addr,unsigned int byte)
 * 描述:	打印指定位置数据
 * 参数:	[in]	addr	 地址
 *      [in]	byte	 字节长度
 * 返回:	无
 ******************************************************************/
void DumpAddrLen(unsigned char *addr,unsigned int byte)
{
	int iloop;
	int realByte = (byte/4*4);

	printf("%p\n", addr);

	printf("           | 00 01 02 03 04 05 06 07 08 09 0A 0B 0C 0D 0E 0F\r\n");
	printf("------------------------------------------------------------\r\n");
	iloop = realByte/16+((realByte%16)?1:0);

	for(int i = 0;i < iloop;i++)
	{
		int loopbyte = (byte-i*16 > 16)?16:(byte-i*16);
		//printf("loopbyte = %d\n",loopbyte);ARGV_WR_OFF_POS
		//printf("0x%08X | ",addr+i*16);
		printf("0x%08X | ",addr+i*16);
		for(int j = 0;j < loopbyte;j++)
		{
			printf("%02X ", *(addr+i*16+j));//%02x     格式控制: 以十六进制输出,2为指定的输出字段的宽度.如果位数小于2,则左端补0
		}
		printf("\r\n");
	}
}


/******************************************************************
 * 函数:	int writeFile(const char* fileName, void* _buf, int _bufLen)
 * 描述:	将指定位置的数据长写入到文件
 * 参数:	[in]	fileName	 文件名称
 *      [in]	_buf	     起始地址
 *      [in]	_bufLen	     字节长度
 * 返回:	0	-	成功
 * 		<0	-	失败
 ******************************************************************/
int writeFile(const char* fileName, void* _buf, int _bufLen)
{
	FILE * fp = NULL;
	if( NULL == _buf || _bufLen <= 0 || fileName == NULL)
		return (-1);

	fp = fopen(fileName, "ab+"); // 必须确保是以 二进制写入的形式打开

	if( NULL == fp )
	{
		return (-1);
	}

	fwrite(_buf, _bufLen, 1, fp); //二进制写

	fclose(fp);
	fp = NULL;

	return 0;
}


