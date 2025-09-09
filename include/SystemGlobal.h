/*
 * SystemGlobal.h
 *
 *  Created on: Feb 16, 2023
 *      Author: howard
 */

#ifndef INC_MISC_HZHY_SYSTEM_H_
#define INC_MISC_HZHY_SYSTEM_H_

#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <pthread.h>
#include <semaphore.h>
#include <sys/mman.h>
#include <termios.h>
#include <sys/ioctl.h>
#include <sys/time.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <arpa/inet.h>

#include <sys/types.h>
#include <sys/socket.h>
//#include "zlog.h"
#include <iostream>
using namespace std;


#pragma pack(1) //编译器将按照1个字节对齐


#define TRUE   1
#define FALSE  0

#define SAFT_FREE(x) {if(x != NULL) {free(x); x=NULL;}}






#endif /* INC_MISC_HZHY_SYSTEM_H_ */
