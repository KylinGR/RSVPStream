/*
 * dma_utils.h
 *
 *  Created on: 2024年4月29日
 *      Author: 86150
 */

#ifndef INC_DMA_UTILS_H_
#define INC_DMA_UTILS_H_

#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>

uint64_t getopt_integer(char *optarg);

ssize_t read_to_buffer(char *fname, int fd, char *buffer, uint64_t size,uint64_t base);


ssize_t write_from_buffer(char *fname, int fd, char *buffer, uint64_t size,uint64_t base);


int timespec_check(struct timespec *t);

void timespec_sub(struct timespec *t1, struct timespec *t2);
#endif /* INC_DMA_UTILS_H_ */
