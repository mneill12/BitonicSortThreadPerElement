#include "userInputHeader.h"

bool isPowerOfTwo(int n){

	double logBase2 = log2(double(n));

	if (round(logBase2) == logBase2)
	{
		return true;
	}

	else
	{
		return false;
	}
}

int getElementCount(){

	int inputCount;

	bool powerOfTwo = false;
	while (!powerOfTwo){

		printf("Enter amount of elements to be sorted: ");
		scanf_s("%d", &inputCount);
		printf("You entered: %d\n", inputCount);

		if (isPowerOfTwo(inputCount))
		{
			powerOfTwo = true;
		}
		else
		{
			printf("/nNot a power of 2, please re enter/n");
		}
	}

	return inputCount;
}

int getThreadCount(){

	int inputCount;

	//bool powerOfTwo = false;
	//while (!powerOfTwo){

	printf("Enter amount of threads per block: ");
	scanf_s("%d", &inputCount);
	printf("You entered: %d\n", inputCount);

	//if (isPowerOfTwo(inputCount))
	//{
	//powerOfTwo = true;
	//}
	//else
	//{
	//printf("/nNot a power of 2, please re enter/n");
	//}
	//	}

	return inputCount;
}

int getBlockCount(){

	int inputCount;

	bool powerOfTwo = false;
	while (!powerOfTwo){

		printf("Enter amount of blocks: ");
		scanf_s("%d", &inputCount);
		printf("You entered: %d\n", inputCount);

		if (isPowerOfTwo(inputCount))
		{
			powerOfTwo = true;
		}
		else
		{
			printf("/nNot a power of 2, please re enter/n");
		}
	}
	return inputCount;
}

bool runSortAgain(){

	int runAgain;

	printf("Do you wish to run the sort again? 1 for Yes 0 for No");
	scanf_s("%d", &runAgain);

	if (runAgain == 1){
		printf("You have selected Yes\n");
		return true;
	}

	else{
		printf("You have selected No\n");
		return false;
	}
}