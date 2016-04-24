#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

//Function Prototypes 
void writeArrayAsCsvFile(char* filename, char* arrayState, int* array, int arrayLength);
void writeBlockElementCsvFile(int* values, char* arrayState, int threadCount, int deviceBlocks);
void writeTimeTaken(int* values, char* arrayState, int threadCount, int deviceBlocks);
	
void incrementFileId(char* fileDirAndName);
int fileExists(const char *fileName);