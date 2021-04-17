#include "ioutils.h"

bool ReadFile(std::string filePath, unsigned char *_data, int *datalen) { 
    std::ifstream is(filePath, std::ifstream::binary); 
    
    if (is) { 
        is.seekg(0, is.end); 
        int length = (int)is.tellg(); 
        is.seekg(0, is.beg); 
        is.read((char*) _data, length); 
        is.close(); 
        *datalen = length; 
    } 
    
    return true; 

}

int WriteToFile(std::string filePath, unsigned char* data, int data_len) { 
    std::ofstream fout; 
    fout.open(filePath, std::ios::out | std::ios::binary); 
    if (fout.is_open()) { 
        fout.write((const char*)data, data_len); 
        fout.close(); 
    } 
    
    return 0; 
}