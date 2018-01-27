#include <iostream>
#include "Rotation.h"
#include "CameraModel.h"
#include "ModelData.h"

using namespace FishEye;

int pairNum = 300, trialNum = 500;
double sigma = 0.0, translateLen = 0.0;

int parseCmdArgs(int argc, char** argv)
{
	for (int i = 1; i < argc; i++)
	{
		if (std::string(argv[i]) == "-pairNum")
		{
			pairNum = atof(argv[i + 1]);
			i++;
		}
		else if (std::string(argv[i]) == "-sigma")
		{
			sigma = atof(argv[i + 1]);
			i++;
		}
		else if (std::string(argv[i]) == "-tl")
		{
			translateLen = atof(argv[i + 1]);
			i++;
		}
		else if (std::string(argv[i]) == "-trialNum")
		{
			trialNum = atof(argv[i + 1]);
			i++;
		}
	}

	return 0;
}


int main(int argc, char *argv[])
{
	parseCmdArgs(argc, argv);

	std::cout << "pairNum : " << pairNum << std::endl;
	std::cout << "trialNum : " << trialNum << std::endl;
	std::cout << "sigma : " << sigma << std::endl;
	std::cout << "translateLen : " << translateLen << std::endl;
	
	std::string classicModelName[3] = { "Equidistant", "Equisolid", "Stereographic" };
	double minFocal = 400, maxFocal = 600;
	double minFov = CV_PI * (160 / 180.0), maxFov = CV_PI * (200 / 180.0);
	double minAngle = CV_PI * (70 / 180.0), maxAngle = CV_PI * (110 / 180.0);
	
	std::ofstream fs("D:/Academic-Research/My Papers/FishEyeCodeMaterials/TestCodes/LensModel/SyntheticData.txt", std::ios::out);
	fs << trialNum << std::endl;
	ModelDataProducer producer;
	srand(time(NULL));

	for (size_t i = 0; i < trialNum; i++)
	{
		double fov = RandomInRange(minFov, maxFov);
		double f = RandomInRange(minFocal, maxFocal);
		int typeIdx = RandomInRange(0, 3);
		
		std::shared_ptr<CameraModel> pModel = createCameraModel(classicModelName[typeIdx], 0, 0, f, fov, 0);/*std::make_shared<Stereographic>(0, 0, f, fov)*/;
		std::shared_ptr<Rotation> pRotation = std::make_shared<Rotation>(minAngle, maxAngle);
		producer.produce(pModel, pRotation, pairNum, sigma, translateLen);
		producer.writeToFile(fs);
	}
	fs.close();
	return 0;
}