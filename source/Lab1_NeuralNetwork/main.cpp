#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "ReadMNIST.h"
#include "Network.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	string fileTestImages = "t10k-images.idx3-ubyte";
	string fileTestLabels = "t10k-labels.idx1-ubyte";
	string fileTrainImages = "train-images.idx3-ubyte";
	string fileTrainLabels = "train-labels.idx1-ubyte";
	int number_of_images_test = 10000;
	int number_of_images_train = 60000;
	int image_size = 28 * 28;
	int numberOfEpochs = 10;
	double crossError = 0.005;
	double learningRate = 0.01;
	int hiddenNeurons = 60;
	int classNumber = 10;

	double** data_test = new double*[number_of_images_test];
	for (int i = 0; i < number_of_images_test; i++)
		data_test[i] = new double[image_size];
	read_Mnist(fileTestImages, data_test);

	double* labels_test = new double[number_of_images_test];
	read_Mnist_Label(fileTestLabels, labels_test);

	double** data_train = new double*[number_of_images_train];
	for (int i = 0; i < number_of_images_train; i++)
		data_train[i] = new double[image_size];
	read_Mnist(fileTrainImages, data_train);

	double* labels_train = new double[number_of_images_train];
	read_Mnist_Label(fileTrainLabels, labels_train);

	cout << "Network creation...\n";
	Network myNetwork = Network(image_size, hiddenNeurons, classNumber, learningRate);
	cout << "Network is created\n";

	cout << "Train \n";
	myNetwork.runTrain(data_train, labels_train, number_of_images_train, numberOfEpochs, crossError, true);
	cout << "\n";
	cout << "Test \n";
	myNetwork.runTrain(data_test, labels_test, number_of_images_test, numberOfEpochs, crossError, false);

	system("PAUSE");

	for (int i = 0; i < number_of_images_test; i++)
		delete[] data_test[i];
	delete[] data_test;

	for (int i = 0; i < number_of_images_train; i++)
		delete[] data_train[i];
	delete[] data_train;

	delete[] labels_test;
	delete[] labels_train;

	return 0;
}
