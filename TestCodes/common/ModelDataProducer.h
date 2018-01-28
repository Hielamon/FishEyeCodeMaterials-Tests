#pragma once

#include "CameraModel.h"
#include "Rotation.h"
#include <fstream>
#include <chrono>
#include <random>

class ModelDataProducer
{
public:
	ModelDataProducer() { mcount = 0; }
	~ModelDataProducer(){}

	void produce(std::shared_ptr<CameraModel> &pCam, std::shared_ptr<Rotation> &pRot,
				 int pairNum, double sigma, double translateLen)
	{
		assert(pairNum > 0 && pCam.use_count() != 0 && pRot.use_count() != 0);
		mpCam = pCam;
		mpRot = pRot;

		if (!mvImgPt1.empty())mvImgPt1.clear();
		if (!mvImgPt2.empty())mvImgPt2.clear();
		if (!mvSpherePt1.empty())mvSpherePt1.clear();
		if (!mvSpherePt2.empty())mvSpherePt2.clear();

		cv::Vec3d direction = RandomAxis();
		cv::Vec3d translate = translateLen * direction;
		

		mcount = 0;
		while (mcount < pairNum)
		{
			double phi = pCam->fov * (rand() / double(RAND_MAX)) * 0.5;
			double theta = CV_2PI * (rand() / double(RAND_MAX));

			cv::Point3d spherePt(sin(phi)*cos(theta), sin(phi)*sin(theta), cos(phi));
			cv::Point3d spherePtByRot = RotatePoint(spherePt, *pRot) + cv::Point3d(translate);

			double phiByRot = atan2(sqrt(spherePtByRot.x*spherePtByRot.x +
										 spherePtByRot.y*spherePtByRot.y), spherePtByRot.z);
			if (phiByRot * 2 > pCam->fov) continue;

			cv::Point2d imgPt, imgPtByRot;
			pCam->mapS2I(spherePt, imgPt);
			pCam->mapS2I(spherePtByRot, imgPtByRot);
			mvSpherePt1.push_back(spherePt);
			mvSpherePt2.push_back(spherePtByRot);

			imgPt += cv::Point2d(_randPixel(sigma), _randPixel(sigma));
			imgPtByRot += cv::Point2d(_randPixel(sigma), _randPixel(sigma));

			mvImgPt1.push_back(imgPt);
			mvImgPt2.push_back(imgPtByRot);
			mcount++;
		}
	}

	std::string writeToFile(std::ofstream &fs)
	{
		assert(mcount > 0 && mpCam.use_count() != 0 && mpRot.use_count() != 0 && fs.is_open());

		std::string typeName = mpCam->getTypeName();
		fs << mcount << " " << typeName << std::endl;
		fs << mpCam->u0 << " " << mpCam->v0 << " " << mpCam->f << " " << mpCam->fov << " " << mpCam->maxRadius << " " << std::endl;
		fs << mpRot->axisAngle[0] << " " << mpRot->axisAngle[1] << " " << mpRot->axisAngle[2] << std::endl;
		for (size_t i = 0; i < mcount; i++)
		{
			cv::Point2d &imgPt1 = mvImgPt1[i], &imgPt2 = mvImgPt2[i];
			cv::Point3d &spherePt1 = mvSpherePt1[i], &spherePt2 = mvSpherePt2[i];
			fs << imgPt1.x << " " << imgPt1.y << " " << imgPt2.x << " " << imgPt2.y << std::endl;
			fs << spherePt1.x << " " << spherePt1.y << " " << spherePt1.z << " " <<
				spherePt2.x << " " << spherePt2.y << " " << spherePt2.z << std::endl;
		}

		return typeName;
	}


	//note that the fs must be matched to the ModelDataProducer
	std::string readFromFile(std::ifstream &fs)
	{
		std::string typeName;
		fs >> mcount >> typeName;
		//the fourth here is maxRadius for General Model but fov for Classic Model
		double u0, v0, f, fov, maxRadius;
		fs >> u0 >> v0 >> f >> fov >> maxRadius;
		mpCam = createCameraModel(typeName, u0, v0, f, fov, maxRadius);

		mpRot = std::make_shared<Rotation>();
		fs >> mpRot->axisAngle[0] >> mpRot->axisAngle[1] >> mpRot->axisAngle[2];
		mvImgPt1.resize(mcount); 
		mvImgPt2.resize(mcount);
		mvSpherePt1.resize(mcount);
		mvSpherePt2.resize(mcount);
		for (size_t i = 0; i < mcount; i++)
		{
			cv::Point2d &imgPt1 = mvImgPt1[i], &imgPt2 = mvImgPt2[i];
			cv::Point3d &spherePt1 = mvSpherePt1[i], &spherePt2 = mvSpherePt2[i];
			fs >> imgPt1.x >> imgPt1.y >> imgPt2.x >> imgPt2.y;
			fs >> spherePt1.x >> spherePt1.y >> spherePt1.z >>
				spherePt2.x >> spherePt2.y >> spherePt2.z;
		}

		return typeName;
	}

//private:
	std::shared_ptr<CameraModel> mpCam;
	std::shared_ptr<Rotation> mpRot;
	std::vector<cv::Point2d> mvImgPt1, mvImgPt2;
	std::vector<cv::Point3d> mvSpherePt1, mvSpherePt2;
	int mcount;

private:
	double _randNormal(double &start, double &end, double &mean, double &sigma)
	{
		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		std::default_random_engine generator(seed);
		std::normal_distribution<double> distribution(mean, sigma);
		double result = distribution(generator);
		while (result < start || result > end)
		{
			result = distribution(generator);
		}

		return result;
	}

	double _randPixel(double sigma)
	{
		
		double ratio = rand() / double(RAND_MAX);
		return ratio * (2 * sigma) - sigma;
	}
};
