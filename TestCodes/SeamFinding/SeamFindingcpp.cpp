#include <OpencvCommon.h>
#include <opencv2/stitching/detail/seam_finders.hpp>

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <sstream>

bool LoadWarpedInfos(std::vector<cv::Mat>& blend_warpeds, std::vector<cv::Mat>& blend_warped_masks,
					 std::vector<cv::Point> &blend_corners, const std::string &fName)
{
	if (!blend_corners.empty())blend_corners.clear();
	if (!blend_warpeds.empty())blend_warpeds.clear();
	if (!blend_warped_masks.empty())blend_warped_masks.clear();

	std::ifstream fs(fName, std::ios::in);
	if (!fs.is_open())return false;
	
	std::string dir = fName.substr(0, fName.rfind('/') + 1);

	
	for ( ;!fs.eof(); )
	{
		std::string name;
		fs >> name;
		if (name.empty()) break;
		cv::Mat blend_warped, blend_warped_mask;
		blend_warped = cv::imread(dir + name);
		blend_warpeds.push_back(blend_warped);

		fs >> name;
		blend_warped_mask = cv::imread(dir + name, cv::IMREAD_GRAYSCALE);
		blend_warped_masks.push_back(blend_warped_mask);

		cv::Point corner;
		fs >> corner.x >> corner.y;
		blend_corners.push_back(corner);
	}

	fs.close();

	return true;
}

int main(int argc, char *argv[])
{
	std::vector<cv::Mat> blend_warpeds, blend_warped_masks;
	std::vector<cv::Point> blend_corners;
	std::string fName = "D:/Academic-Research/My Papers/FishEyeCodeMaterials-Stitcher/FishEyeStitcher/build_x64_vs15/FishEyeStitcherTest/warpedInfos.txt";
	
	LoadWarpedInfos(blend_warpeds, blend_warped_masks, blend_corners, fName);
	std::vector<double> areas;
	std::for_each(blend_warpeds.begin(), blend_warpeds.end(), [&](cv::Mat &warped) {
		areas.push_back(warped.size().area());
	});

	std::sort(areas.begin(), areas.end());

	double seam_megapix = 0.1, seam_scale = std::min(1.0, sqrt(seam_megapix * 1e6 / areas[areas.size() / 2]));

	int num_images = blend_warpeds.size();
	std::vector<cv::UMat> seam_warped(num_images);
	std::vector<cv::UMat> seam_warped_f(num_images);
	std::vector<cv::UMat> seam_masks_warped(num_images);
	std::vector<cv::Point> seam_corners;

	for (size_t i = 0; i < num_images; i++)
	{
		cv::Size seam_size = cv::Size(blend_warpeds[i].cols*seam_scale, blend_warpeds[i].rows*seam_scale);

		cv::resize(blend_warpeds[i], seam_warped[i], seam_size);
		cv::resize(blend_warped_masks[i], seam_masks_warped[i], seam_size);
		seam_warped[i].convertTo(seam_warped_f[i], CV_32F);
		seam_corners.push_back(blend_corners[i] * seam_scale);
	}
	cv::Ptr<cv::detail::SeamFinder> seam_finder = cv::makePtr<cv::detail::GraphCutSeamFinder>(cv::detail::GraphCutSeamFinderBase::COST_COLOR_GRAD);
	seam_finder->find(seam_warped_f, seam_corners, seam_masks_warped);
	
	for (size_t i = 0; i < num_images; i++)
	{
		std::stringstream ioStr;
		ioStr << "maskSeam_" << i << ".jpg";
		cv::Mat mask = seam_masks_warped[i].getMat(cv::ACCESS_READ);
		cv::imwrite(ioStr.str(), mask);
	}

	return 0;
}