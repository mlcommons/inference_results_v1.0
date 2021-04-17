#ifndef IOUTILS_H
#define IOUTILS_H

#include <string>
#include <fstream>

bool ReadFile(std::string filePath, unsigned char *_data, int *datalen);
int WriteToFile(std::string filePath, unsigned char* data, int data_len);

#endif