#define MAIN_FILE
#include <commonMacro.h>
#include "../common/OptimizeCommon.h"

using namespace cv;
using namespace FishEye;


double ratio = 15, base = 15;

int main(int argc, char *argv[])
{
	std::shared_ptr<Equidistant> baseModel = std::make_shared<Equidistant>(0, 0, 1, CV_PI);
	std::map<std::string, cv::Vec2d> generalModelInfo;
	generalModelInfo["PolynomialAngle"] = cv::Vec2d(1.000000, 0.000000);
	generalModelInfo["PolynomialRadius"] = cv::Vec2d(1.038552, -0.407288);
	generalModelInfo["GeyerModel"] = cv::Vec2d(0.976517, 1.743803);

	std::map<std::string, std::vector<std::vector<double>>> vErrors, vRotErrors;
	{
		vErrors["PolynomialAngle"] = vRotErrors["PolynomialAngle"] = std::vector<std::vector<double>>();
		vErrors["PolynomialRadius"] = vRotErrors["PolynomialRadius"] = std::vector<std::vector<double>>();
		vErrors["GeyerModel"] = vRotErrors["GeyerModel"] = std::vector<std::vector<double>>();
	}
	
	int maxLevel = 20;
	for (size_t j = 0; j < maxLevel; j++)
	{
		int pNum = j*ratio + base;

		std::cout << "\n\n\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^" << std::endl;
		std::cout << "Pairs Num Level = " << j << "th(" << pNum << " pairs)" << std::endl;

		std::map<std::string, std::vector<double>> Errors, RotErrors;
		{
			Errors["PolynomialAngle"] = RotErrors["PolynomialAngle"] = std::vector<double>();
			Errors["PolynomialRadius"] = RotErrors["PolynomialRadius"] = std::vector<double>();
			Errors["GeyerModel"] = RotErrors["GeyerModel"] = std::vector<double>();
		}
		
		std::stringstream ioStr;
		ioStr << "..\\x64\\Release\\CameraDataFactory.exe -trialNum 1000 -sigma 0 -tl 0 -pairNum " << pNum;
		//ioStr << "D:/Academic-Research/\"My Papers\"/FishEyeCodeMaterials/TestCodes/x64/Debug/LensModel.exe -tl 0 -pairNum 300 -trialNum 500 -sigma " << j;
		std::string command = ioStr.str();
		system(command.c_str());

		std::string dataFile = "SyntheticData.txt";
		std::ifstream fs(dataFile, std::ios::in);
		if (!fs.is_open())
			HL_CERR("Failed to open the file " << dataFile);

		int trialNum;
		fs >> trialNum;
		for (size_t i = 0; i < trialNum; i++)
		{
			std::shared_ptr<ModelDataProducer> pModelData = std::make_shared<ModelDataProducer>();
			std::string typeName = pModelData->readFromFile(fs);

			std::cout << "trial N.O. = " << j << "." << i << std::endl;
			//if (i != 547)continue;s

			double maxRadius = pModelData->mpCam->maxRadius;
			double f = maxRadius / baseModel->maxRadius;

			std::map<std::string, cv::Vec2d>::iterator iter = generalModelInfo.begin();
			for (; iter != generalModelInfo.end(); iter++)
			{
				//std::cout << "vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv" << std::endl;
				//std::cout << "Model -------> " << iter->first << std::endl;
				std::shared_ptr<CameraModel> pModel = std::static_pointer_cast<CameraModel>(
					createCameraModel(iter->first, 0, 0, f, 0, maxRadius, iter->second[0], iter->second[1]));

				std::shared_ptr<Rotation> pRot = std::make_shared<Rotation>(CV_PI*0.5, CV_PI*0.5);
				CalculateRotation(pModelData, pModel, pRot);
				std::vector<uchar> vMask(pModel->vpParameter.size(), 1);
				vMask[0] = vMask[1] = 0;

				//std::string logFileName = iter->first + "_error.txt";
				std::string logFileName;
				Ptr<FishModelRefineCallback> cb = makePtr<FishModelRefineCallback>(pModelData, pModel, pRot, vMask);
				Ptr<LMSolver> levmarpPtr = customCreateLMSolver(cb, 200, FLT_EPSILON, FLT_EPSILON, logFileName);

				//get the initial parameters in param;
				//we use fov = 180бу and known circle radius to init the focol length and relative parameters
				cv::Mat param(6, 1, CV_64FC1);
				{
					param.at<double>(0, 0) = *(pModel->vpParameter[2]);
					param.at<double>(1, 0) = *(pModel->vpParameter[3]);
					param.at<double>(2, 0) = *(pModel->vpParameter[4]);
					param.at<double>(3, 0) = pRot->axisAngle[0];
					param.at<double>(4, 0) = pRot->axisAngle[1];
					param.at<double>(5, 0) = pRot->axisAngle[2];
				}

				levmarpPtr->run(param);

				cv::Mat err, J;
				cb->compute(param, err, J);
				double error = norm(err);
				Errors[iter->first].push_back(error);

				double rotError = 0;
				cv::Vec3d rotResult(param.at<double>(3, 0), param.at<double>(4, 0), param.at<double>(5, 0));
				rotError = norm(rotResult - pModelData->mpRot->axisAngle);
				RotErrors[iter->first].push_back(rotError);

				std::cout << iter->first << " error : " << error << std::endl;
			}
		}

		std::map<std::string, std::vector<double>>::iterator nIter = Errors.begin();
		for (; nIter != Errors.end(); nIter++)
		{
			vErrors[nIter->first].push_back(nIter->second);
		}

		nIter = RotErrors.begin();
		for (; nIter != RotErrors.end(); nIter++)
		{
			vRotErrors[nIter->first].push_back(nIter->second);
		}
	}

	std::string dir = "";
	//SaveErrorsToFile(vErrors, ratio, base, dir, "PairsNum");
	//SaveErrorsToFile(vRotErrors, ratio, base, dir, "PairsNumRot");
	return 0;
}