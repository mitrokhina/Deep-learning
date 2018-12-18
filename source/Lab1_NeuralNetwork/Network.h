#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <random>

using namespace std;

class Network {
public:
	Network(int inputNeurons, int hiddenNeurons, int outputNeurons, double learningRate);
	~Network();
	void runTrain(double** dataSet, double* labels, int dataSize, int numberOfEpochs, double crossError, bool isTrain);

private:
	int inputNeurons;
	int hiddenNeurons;
	int outputNeurons;
	double learningRate;

	double* inputLayer;
	double* hiddenLayer;
	double* outputLayer;
	double* gradientHideLayer;
	double* gradientOutputLayer;

	double** firstLayerWeights;
	double** secondLayerWeights;

	double** generateLayerWeights(int firstLayerSize, int secondLayerSize);
	double* calculateConvolution(double* firstLayer, int firstLayerSize, int secondLayerSize, double** layerWeights);
	void calculateHideLayerOutput();
	void calculateOutputLayerOutput();
	void calculateGradient(double *y); 
	void updateWeights(double * gradOutput, double * gradHidden);
	void mixData(double** dataSet, double* labels, int size);
};