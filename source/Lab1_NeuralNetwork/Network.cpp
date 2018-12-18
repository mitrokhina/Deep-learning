#include "Network.h"

double* softmax(double* input, int inputSize)
{
	double* softmax = new double[inputSize];
	double* numer = new double[inputSize];
	double denom = 0.0;

	for (int i = 0; i < inputSize; i++)
	{
		numer[i] = exp(input[i]);
		denom += numer[i];
	}
	for (int i = 0; i < inputSize; i++)
		softmax[i] = numer[i] / denom;

	return softmax;
}

double* sigmoid(double* input, int inputSize)
{
	double* sigmoid = new double[inputSize];

	for (int i = 0; i < inputSize; i++)
		sigmoid[i] = 1.0 / (1.0 + exp(-input[i]));

	return sigmoid;
}

int indexOfMaxElement(double *arr, int arrSize)
{
	double max = 0.0;
	int index = 0;

	for (int i = 0; i < arrSize; i++)
		if (max < arr[i])
		{
			max = arr[i];
			index = i;
		}

	return index;
}

double generateWeights(double min, double max)
{
	return rand()*(max - min) / (RAND_MAX + 0.1) + min;
}

Network::Network(int inputNeurons, int hiddenNeurons, int outputNeurons, double learningRate)
{
	this->inputNeurons = inputNeurons;
	this->hiddenNeurons = hiddenNeurons;
	this->outputNeurons = outputNeurons;
	this->learningRate = learningRate;

	inputLayer = new double[inputNeurons];
	hiddenLayer = new double[hiddenNeurons];
	outputLayer = new double[outputNeurons];

	firstLayerWeights = generateLayerWeights(inputNeurons, hiddenNeurons);
	secondLayerWeights = generateLayerWeights(hiddenNeurons, outputNeurons);

	gradientHideLayer = new double[hiddenNeurons];
	for (int i = 0; i < hiddenNeurons; i++)
		gradientHideLayer[i] = 0.0;

	gradientOutputLayer = new double[outputNeurons];
	for (int i = 0; i < outputNeurons; i++)
		gradientOutputLayer[i] = 0.0;
}

double** Network::generateLayerWeights(int firstLayerSize, int secondLayerSize)
{
	double** layerWeights = new double*[firstLayerSize];

	for (int i = 0; i < firstLayerSize; i++)
		layerWeights[i] = new double[secondLayerSize];
	for (int i = 0; i < firstLayerSize; i++)
		for (int j = 0; j < secondLayerSize; j++)
			layerWeights[i][j] = generateWeights(-1.0, 1.0);

	return layerWeights;
}

double* Network::calculateConvolution(double* firstLayer, int firstLayerSize, int secondLayerSize, double** layerWeights)
{
	double* summary = new double[secondLayerSize];

	for (int i = 0; i < secondLayerSize; i++)
		summary[i] = 0.0;

	for (int i = 0; i < secondLayerSize; i++)
		for (int j = 0; j < firstLayerSize; j++)
			summary[i] += firstLayer[j] * layerWeights[j][i];

	return summary;
}

void Network::calculateHideLayerOutput()
{
	double* conv = calculateConvolution(inputLayer, inputNeurons, hiddenNeurons, firstLayerWeights);

	hiddenLayer = sigmoid(conv, hiddenNeurons);
}

void Network::calculateOutputLayerOutput()
{
	double* conv = calculateConvolution(hiddenLayer, hiddenNeurons, outputNeurons, secondLayerWeights);

	outputLayer = softmax(conv, outputNeurons);
}

void Network::calculateGradient(double *y)
{
	double sum = 0.0;

	for (int i = 0; i < outputNeurons; i++)
			gradientOutputLayer[i] = outputLayer[i] - y[i];

	for (int i = 0; i < hiddenNeurons; i++)
	{
		for (int j = 0; j < outputNeurons; j++)
			sum += gradientOutputLayer[j] * secondLayerWeights[i][j];

		gradientHideLayer[i] = hiddenLayer[i] * (1 - hiddenLayer[i]) * sum;
	}
}

void Network::updateWeights(double * gradOutput, double * gradHidden)
{
	double deltaWeight = 0.0;

	for (int i = 0; i < inputNeurons; i++)
		for (int j = 0; j < hiddenNeurons; j++)
		{
			deltaWeight = learningRate * gradHidden[j] * inputLayer[i];
			firstLayerWeights[i][j] -= deltaWeight; 
		}

	for (int i = 0; i < hiddenNeurons; i++)
		for (int j = 0; j < outputNeurons; j++)
		{
			deltaWeight = learningRate * gradOutput[j] * hiddenLayer[i];
			secondLayerWeights[i][j] -= deltaWeight; 
		}
}

void Network::runTrain(double** dataSet, double* labels, int dataSize, int numberOfEpochs, double crossError, bool isTrain)
{
	double* expectedOutput = new double[outputNeurons];
	int epochCount = 0;
	double sum = 0.0;
	double crossEntrophy = 0.0;
	double accuracy = 0.0;
	int countCorrectAnswers = 0;

	while (epochCount < numberOfEpochs)
	{
		countCorrectAnswers = 0;
		mixData(dataSet, labels, dataSize);

		cout << "\n";
		cout << "Epoch number: " << epochCount << "\n";

		for (int i = 0; i < dataSize; i++)
		{
			for (int j = 0; j < inputNeurons; j++)
				inputLayer[j] = dataSet[i][j];

			for (int j = 0; j < outputNeurons; j++)
			{
				expectedOutput[j] = 0.0;
				if (j == labels[i])
					expectedOutput[j] = 1.0;
			}

			calculateHideLayerOutput();
			calculateOutputLayerOutput();

			for (int j = 0; j < outputNeurons; j++)
				sum += log(outputLayer[j]) * expectedOutput[j];

			if (expectedOutput[indexOfMaxElement(outputLayer, outputNeurons)] == 1.0)
				countCorrectAnswers++;

			if (isTrain)
			{
				calculateGradient(expectedOutput);
				updateWeights(gradientOutputLayer, gradientHideLayer);
			}
			else
			{
				epochCount = numberOfEpochs;
			}
		}

		crossEntrophy = -sum / dataSize; 
		cout << "Cross entrophy: " << crossEntrophy << "\n";

		cout << "Correct answers: " << countCorrectAnswers << "\n";

		accuracy = (double)countCorrectAnswers / dataSize; 
		cout << "Accuracy: " << accuracy << "\n";

		epochCount++;

		if ((crossEntrophy <= crossError) || (1 - accuracy <= crossError))
			break;
	}
}
void Network::mixData(double** dataSet, double* labels, int size)
{
	for (int i = 0; i < size; i++)
	{
		int firstPosition = rand() % size;
		int secondPosition = rand() % size;

		swap(dataSet[firstPosition], dataSet[secondPosition]);
		swap(labels[firstPosition], labels[secondPosition]);
	}
}

Network::~Network()
{
	delete[] inputLayer;
	delete[] hiddenLayer;
	delete[] outputLayer;
	delete[] gradientHideLayer;
	delete[] gradientOutputLayer;

	for (int i = 0; i < inputNeurons; i++)
		delete[] firstLayerWeights[i];
	delete[] firstLayerWeights;

	for (int i = 0; i < hiddenNeurons; i++)
		delete[] secondLayerWeights[i];
	delete[] secondLayerWeights;
}
