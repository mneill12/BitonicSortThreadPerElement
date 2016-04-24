#include "writeToCSVFileHeader.h"

void writeArrayAsCsvFile(char* filename, char* arrayState, int* array, int arrayLength){

	struct tm *tm;
	time_t t;
	char str_time[100];
	char str_date[100];

	t = time(NULL);
	tm = localtime(&t);

	strftime(str_time, sizeof(str_time), "-Time-%H-%M-%S", tm);
	strftime(str_date, sizeof(str_date), "-Date-%d-%m-%Y", tm);
	
	char fileDirAndName[120] = "C:/BitonicSortArrayCSVFiles/";
	//Use array state as folder name, Create file name from string of array
	strcat(fileDirAndName, arrayState);
	strcat(fileDirAndName, "/");
	strcat(fileDirAndName, filename);
	strcat(fileDirAndName,str_date);
	strcat(fileDirAndName, str_time);

	strcat(fileDirAndName, ".csv");

	printf("");
	FILE* file = fopen(fileDirAndName, "w");

	for (int i = 0; i < arrayLength; i++){
		fprintf(file, "%d \n", *array);
		array++;
	}

	//Add Array State at the end
	fprintf(file, "\n ");
	fprintf(file, arrayState);

	fclose(file);
}
void writeBlockElementCsvFile(int* values, char* arrayState, int threadCount, int deviceBlocks){

	char* string = "SortElements%dBlocks%d";
	char filename[28];
	sprintf(filename, string, threadCount, deviceBlocks);

	writeArrayAsCsvFile(filename, arrayState, values, threadCount);

}

void writeTimeTaken(int* timeResults, int timesCount, char* sortType, char* arrayState, int* threadCounts, int* deviceBlocks){


	char* filename = "TimedResultsFor";
	strcat(filename, sortType);

	char fileDirAndName[120] = "C:/BitonicSortArrayCSVFiles/";
	//Use array state as folder name
	strcat(fileDirAndName, "timedResults");
	strcat(fileDirAndName, "/");
	strcat(fileDirAndName, filename);
	strcat(fileDirAndName, ".csv");

	//if the file already exists we're just going to change the number at the end of the string 
	if (fileExists(fileDirAndName)){

		incrementFileId(fileDirAndName);	
	}

	FILE* file = fopen(fileDirAndName, "w");

	fprintf(file, fileDirAndName);

	//Column headers 
	fprintf(file, "Sort Type, Threads, Blocks, time, result");

	for (int i = 0; i < timesCount; i++){

		fprintf(file, "/n,%d ", sortType);
		fprintf(file, ",%d ", threadCounts);
		fprintf(file, ",%d ", deviceBlocks);
		fprintf(file, ",%d ", timeResults);
		fprintf(file, ",%d ", arrayState);
		sortType++;
		threadCounts++;
		deviceBlocks++;
		timeResults++;
		arrayState++;

	}

	//Add Array State at the end
	fprintf(file, "\n ,%d ");

	fclose(file);


}

int fileExists(const char *fileName)
{
	FILE *file;
	if (file = fopen(fileName, "r"))
	{
		fclose(file);
		return 1;
	}
	return 0;
}

void incrementFileId(char* fileDirAndName){

	char fileIdx = fileDirAndName[strlen(fileDirAndName) - 1];

	int fileIdxInt = fileIdx - '0';
	fileIdxInt++;
	fileIdx = fileIdxInt + '0';
	fileDirAndName[strlen(fileDirAndName) - 1] = fileIdx;

}