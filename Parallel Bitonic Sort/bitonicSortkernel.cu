
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <cuda_runtime_api.h>

#include "writeToCSVFileHeader.h"
#include "userInputHeader.h"

void printArray(int *elements);


int deviceBlocks;
int threadsPerBlock;
int elementsToSort;
int threadCount;

//Max times we cann run the process
int executionCount;


const int randMax = 10000;

void createUnsortedArray(int* elements){

	//Get size and cuda dimentions from user input

	for (int i = 0; i < elementsToSort; ++i){
		elements[i] = rand() % randMax - rand() % 5;
	}

}

bool isSorted(int *elements){

	bool sorted = true;
	for (int i = 0; i < (elementsToSort - 1); ++i){
		if (elements[i] > elements[i + 1]){
			sorted = false;
		}
	}
	return sorted;
}


double getElapsedTime(clock_t start, clock_t stop)
{
	double elapsed = ((double)(stop - start)) / CLOCKS_PER_SEC;
	printf("Elapsed time: %.3fs\n", elapsed);

	return elapsed;
}

int random_int()
{
	return (int)rand() / (int)2048;
}


/*
We get our indexes to swap buy xoring our thread index with the step. This essentially wraps the thread Id round the step value meaning 
the only values porduced that are bigger than the Thread Id will be those within the requied step length
*/
__global__ void stepskernel(int *dev_values, int step, int phaseLength)
{
	unsigned int firstIndex, XoredSecondIndex;
	//Set it to the thread Id
	firstIndex = threadIdx.x + blockDim.x * blockIdx.x;

	XoredSecondIndex = firstIndex ^ step;

	//Threads i corrasponding to the desired bitonic element will be used for the swap
	if ((XoredSecondIndex)>firstIndex) {

		if ((firstIndex&phaseLength) == 0) {
			if (dev_values[firstIndex]>dev_values[XoredSecondIndex]) {
		
				int temp = dev_values[firstIndex];
				dev_values[firstIndex] = dev_values[XoredSecondIndex];
				dev_values[XoredSecondIndex] = temp;
			}
		}
		if ((firstIndex&phaseLength) != 0) {

			if (dev_values[firstIndex]<dev_values[XoredSecondIndex]) {
				int temp = dev_values[firstIndex];
				dev_values[firstIndex] = dev_values[XoredSecondIndex];
				dev_values[XoredSecondIndex] = temp;
			}
		}
	}

}

/*
Main function call. Created array and calls stepskernel based of the size of the bitonic sequences and step.
*/
void bitonic_sort(int *values)
{
	int *dev_values;
	size_t size = elementsToSort* sizeof(int);

	//Allocate memory on the device then copy our host array to device pointer
	cudaMalloc((void**)&dev_values, size);
	cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);

	dim3 blocks(deviceBlocks, 1);   
	dim3 threads(threadsPerBlock, 1);  

	int step, phaseLength;

	for (phaseLength = 2; phaseLength <= elementsToSort; phaseLength <<= 1) {

		for (step = phaseLength >> 1; step>0; step = step >> 1) {
			stepskernel << <blocks, threads >> >(dev_values, step, phaseLength);
		}
	}

	cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
	cudaFree(dev_values);
}

void preExecution(){

	int values[7];
	values[0] = 10;
	values[1] = 13;
	values[2] = 9;
	values[3] = 18;
	values[4] = 26;
	values[4] = 100;
	values[6] = 3;

	bitonic_sort(values);

}

int main(void)
{
	
	executionCount = getMaxProcessCount();
	int fixedExecutionCount = executionCount;

	preExecution();

	bool runSort = true;

	runSort = runSortAgain();

	//Pointers to store our results that we're writing to CSV files, allocate space entered buy the user
	int* threadCounts = (int*)malloc(executionCount*sizeof(int));
	int* allBlocks = (int*)malloc(executionCount*sizeof(int));;
	double* timeResults = (double*)malloc(executionCount*sizeof(double));;
	char* arrayStates = (char*)malloc(executionCount*sizeof(char));

	double time;
	clock_t start, stop;
	//Counter so we can assine values to the array in the execution loop

	while (runSort && executionCount != 0){

		//Get thread, blocks and  element count

		//Get total elements and suggested block thread configurations
		blockAndThreadCounts inputCountandSuggestedThreadBlockCount;
		inputCountandSuggestedThreadBlockCount = getElementCounts();
		elementsToSort = inputCountandSuggestedThreadBlockCount.elementCount;

		//wirte possible thread and block configurations to text file
		printf("Writing suggested block thread configuration...");
		writeSuggestedBlockThreadConfigToCsv(inputCountandSuggestedThreadBlockCount.threadCounts,
												inputCountandSuggestedThreadBlockCount.blockCounts, 
												inputCountandSuggestedThreadBlockCount.combinationsCount
												);
		printf("Done \n");
		//elementsToSort = inputCountandSuggestedThreadBlockCount.elementCount;
		deviceBlocks = getBlockCount();
		threadsPerBlock = getThreadCount();

		threadCount = threadsPerBlock * deviceBlocks;

		//Malloc array, add values to it and write unsorted array to csv file
		int* values	 = (int*)malloc(elementsToSort*sizeof(int));
		createUnsortedArray(values);
		writeBlockElementCsvFile(values, "preSorted", threadCount, deviceBlocks);

		//Do Sort and time it
		start = clock();
		bitonic_sort(values);
		stop = clock();

		time = getElapsedTime(start, stop);

		char* arrayState;
		char arrayStateChar;

		if (isSorted(values)){

			printf("Is Sorted \n");
			arrayState = "sorted";
			arrayStateChar = 's';
		}
		else{

			printf("Not Sorted \n");
			arrayState = "unsorted"; 
			arrayStateChar = 'u';
		}

		writeBlockElementCsvFile(values, arrayState, threadCount, deviceBlocks);

		//Allocate results values to pointers 
		*threadCounts = threadCount;
		*allBlocks = deviceBlocks;
		*timeResults = time;
		*arrayStates = arrayStateChar;

		//Increment Result pointers
		threadCounts++;
		allBlocks++;
		timeResults++;
		arrayStates++;

		free(values);
		
		//Check again for user input
		
		executionCount--;
	}

	printf("Execution ended. Writing results to C:\BitonicSortArrayCSVFiles /n");

	writeSortResultsToCsv(timeResults, "ParallelBitonicSort", arrayStates, threadCounts, allBlocks, fixedExecutionCount);

	getchar();
}