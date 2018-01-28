#pragma once

#include <OpencvCommon.h>
#include <commonMacro.h>
#include "../common/OpenCVLevMarq.h"
#include "../common/ModelDataProducer.h"
#include <random>
#include <map>
#include <sstream>
#include <algorithm>

inline void CalculateRotation(const std::shared_ptr<ModelDataProducer> &pModelData,
					   const std::shared_ptr<CameraModel> &pModel,
					   const std::shared_ptr<Rotation> &pRot)
{
	assert(pModelData.use_count() != 0 && pModel.use_count() != 0 && pRot.use_count() != 0);

	double s[9] = { 0 };
	cv::Mat S(3, 3, CV_64FC1, s);
	for (size_t i = 0; i < pModelData->mcount; i++)
	{
		cv::Point2d &imgPt1 = pModelData->mvImgPt1[i], &imgPt2 = pModelData->mvImgPt2[i];
		cv::Point3d spherePt1, spherePt2;
		pModel->mapI2S(imgPt1, spherePt1);
		pModel->mapI2S(imgPt2, spherePt2);

		s[0] += (spherePt1.x * spherePt2.x);
		s[1] += (spherePt1.x * spherePt2.y);
		s[2] += (spherePt1.x * spherePt2.z);
		s[3] += (spherePt1.y * spherePt2.x);
		s[4] += (spherePt1.y * spherePt2.y);
		s[5] += (spherePt1.y * spherePt2.z);
		s[6] += (spherePt1.z * spherePt2.x);
		s[7] += (spherePt1.z * spherePt2.y);
		s[8] += (spherePt1.z * spherePt2.z);
	}

	cv::Mat w, u, vt;
	cv::SVD::compute(S, w, u, vt);
	cv::Mat I = cv::Mat::eye(3, 3, CV_64FC1);
	I.at<double>(2, 2) = cv::determinant(vt.t() * u.t());
	cv::Mat R = vt.t() * I * u.t();
	pRot->updataRotation(R);
}

class FishModelRefineCallback : public cv::LMSolver::Callback
{
public:
	FishModelRefineCallback(const std::shared_ptr<ModelDataProducer> &pModelData,
							const std::shared_ptr<CameraModel> &pModel,
							const std::shared_ptr<Rotation> &pRot,
							const std::vector<uchar> &vMask)
	{
		assert(pModel.use_count() != 0 && pModel.use_count() != 0 && pRot.use_count() != 0);
		mpModelData = pModelData;
		mpModel = pModel;
		mpRot = pRot;

		assert(vMask.size() == pModel->vpParameter.size());

		for (size_t i = 0; i < vMask.size(); i++)
		{
			if (vMask[i] != 0)
			{
				mvpParameter.push_back(pModel->vpParameter[i]);
				mvRotMask.push_back(false);
			}
		}

		for (size_t i = 0; i < 3; i++)
		{
			mvpParameter.push_back(&(mpRot->axisAngle[i]));
			mvRotMask.push_back(true);
		}
	}

	bool compute(cv::InputArray _param, cv::OutputArray _err, cv::OutputArray _Jac) const
	{
		cv::Mat param = _param.getMat();

		for (size_t i = 0; i < mvpParameter.size(); i++)
		{
			*(mvpParameter[i]) = param.at<double>(i, 0);
		}

		int pairNum = mpModelData->mcount;
		//err.create(pairNum * 3, 1, CV_64F);

		_err.create(pairNum * 3, 1, CV_64F);
		cv::Mat err = _err.getMat();
		if (!_calcError(err))return false;

		cv::Mat err2 = _err.getMat();

		if (_Jac.needed())
		{
			_Jac.create(pairNum * 3, mvpParameter.size(), CV_64F);
			cv::Mat J = _Jac.getMat();
			if (!_calcJacobian(J))return false;
		}

		/*std::cout << "average error = " << norm(err) << std::endl;
		std::cout << param.at<double>(0, 0) << std::endl;
		std::cout << param.at<double>(1, 0) << " " << param.at<double>(2, 0) << std::endl;
		std::cout << param.at<double>(3, 0) << " " << param.at<double>(4, 0) << " " << param.at<double>(5, 0) << std::endl;*/
		return true;
	}

private:
	void _calcDeriv(const cv::Mat &err1, const cv::Mat &err2, double h, cv::Mat &res) const
	{
		for (int i = 0; i < err1.rows; ++i)
			res.at<double>(i, 0) = (err2.at<double>(i, 0) - err1.at<double>(i, 0)) / h;
	}

	bool _calcError(cv::Mat &err) const
	{
		int pairNum = mpModelData->mcount;
		//err.create(pairNum * 3, 1, CV_64F);
		err.setTo(0);
		bool valid = true;

		for (size_t i = 0, idx = 0; i < pairNum; i++, idx += 3)
		{
			cv::Point2d &imgPt1 = mpModelData->mvImgPt1[i], &imgPt2 = mpModelData->mvImgPt2[i];
			//cv::Point3d &spherePt1 = mpModelData->mvSpherePt1[i], &spherePt2 = mpModelData->mvSpherePt2[i];
			cv::Point3d spherePt1, spherePt2;

			valid &= mpModel->mapI2S(imgPt1, spherePt1);
			valid &= mpModel->mapI2S(imgPt2, spherePt2);

			if (!valid)return false;

			cv::Point3d spherePt1ByRot = RotatePoint(spherePt1, *(mpRot));
			err.at<double>(idx, 0) = spherePt1ByRot.x - spherePt2.x;
			err.at<double>(idx + 1, 0) = spherePt1ByRot.y - spherePt2.y;
			err.at<double>(idx + 2, 0) = spherePt1ByRot.z - spherePt2.z;
		}

		return true;
	}

	bool _calcJacobian(cv::Mat &jac) const
	{
		int pairNum = mpModelData->mcount;

		//jac.create(pairNum * 3, activeParamNum, CV_64F);
		jac.setTo(0);

		const double step = 1e-6;
		cv::Mat err1, err2;
		err1.create(pairNum * 3, 1, CV_64F);
		err2.create(pairNum * 3, 1, CV_64F);
		bool valid = true;

		for (size_t i = 0; i < mvpParameter.size(); i++)
		{
			double originValue = *(mvpParameter[i]);

			*(mvpParameter[i]) = originValue - step;
			_updateParameters(i);

			valid &= _calcError(err1);

			*(mvpParameter[i]) = originValue + step;
			_updateParameters(i);

			valid &= _calcError(err2);

			_calcDeriv(err1, err2, 2 * step, jac.col(i));

			*(mvpParameter[i]) = originValue;
			_updateParameters(i);

			if (!valid)return false;
		}

		return true;
	}

	void _updateParameters(const int& idx) const
	{
		if (mvRotMask[idx])
		{
			mpRot->updataRotation(mpRot->axisAngle);
		}
		else
		{
			mpModel->updateFov();
		}
	}

	std::shared_ptr<ModelDataProducer> mpModelData;
	std::shared_ptr<CameraModel> mpModel;
	std::shared_ptr<Rotation> mpRot;
	std::vector<double *> mvpParameter;
	std::vector<bool> mvRotMask;

};


inline void SaveErrorsToFileOld(std::map<std::string, std::vector<std::vector<double>>> &vErrors,
							double ratio, double base, const std::string &dir, const std::string &subName)
{
	for (auto iter = vErrors.begin(); iter != vErrors.end(); iter++)
	{
		std::cout << iter->first << " : " << std::endl;
		std::vector<double> meanErrors, medianErrors;
		for (size_t i = 0; i < iter->second.size(); i++)
		{
			auto errsTmp = iter->second[i];
			std::sort(errsTmp.begin(), errsTmp.end());
			medianErrors.push_back(errsTmp[errsTmp.size() / 2]);
			double sum = 0;
			for (size_t k = 0; k < errsTmp.size(); k++)
			{
				sum += errsTmp[k];
			}
			meanErrors.push_back(sum / errsTmp.size());
		}

		std::string fName = dir + iter->first + "_" + subName + "_meanErrors.txt";
		std::ofstream fs(fName.c_str(), std::ios::out);
		std::cout << "meanErrors : ";
		for (size_t i = 0; i < meanErrors.size(); i++)
		{
			std::cout << meanErrors[i] << " ";
			fs << i * ratio + base << " " << meanErrors[i] << std::endl;
		}
		std::cout << std::endl;
		fs.close();

		fName = dir + iter->first + "_" + subName + "_medianErrors.txt";
		fs.open(fName.c_str(), std::ios::out);
		std::cout << "medianErrors : ";
		for (size_t i = 0; i < medianErrors.size(); i++)
		{
			std::cout << medianErrors[i] << " ";
			fs << i * ratio + base << " " << medianErrors[i] << std::endl;
		}
		std::cout << std::endl;
		fs.close();
	}
}

inline void SaveErrorsToFile(const std::vector<double> &vInput, const std::string &fName,
							 double ratio = 1.0, double base = 0.0)
{
	std::ofstream fs(fName.c_str(), std::ios::out);
	if (!fs.is_open())
		HL_CERR("Failed to open the file " << fName);

	for (size_t i = 0; i < vInput.size(); i++)
	{
		std::cout << vInput[i] << " ";
		fs << i * ratio + base << " " << vInput[i] << std::endl;
	}
	std::cout << std::endl;
	fs.close();
}

template< class T>
inline void GetMeanAndMedian(const std::vector<std::vector<T>> &vvInput, std::vector<T> &vMean, std::vector<T> &vMedian)
{
	if (!vMean.empty()) vMean.clear();
	if (!vMedian.empty()) vMedian.clear();

	for (size_t i = 0; i < vvInput.size(); i++)
	{
		auto errsTmp = vvInput[i];
		std::sort(errsTmp.begin(), errsTmp.end());
		vMedian.push_back(errsTmp[errsTmp.size() / 2]);
		double sum = 0;
		for (size_t k = 0; k < errsTmp.size(); k++)
		{
			sum += errsTmp[k];
		}
		vMean.push_back(sum / errsTmp.size());
	}
}