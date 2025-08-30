/*
 * hzhy_cal.h
 *
 *  Created on: Jan 17, 2022
 *      Author: ZhaoBinBin
 */

#ifndef INC_HZHY_CAL_H_
#define INC_HZHY_CAL_H_

#include <SystemGlobal.h>

/*
 *
 */
uint64_t HzhyGetNowUs();


/*
 *
 */
void *HzhyMalloc(uint32_t len);

/*
 *
 */
void DumpAddrLen(unsigned char *addr,unsigned int byte);



/*
 *
 */
int writeFile(const char* fileName, void* _buf, int _bufLen);





#endif /* INC_HZHY_CAL_H_ */
