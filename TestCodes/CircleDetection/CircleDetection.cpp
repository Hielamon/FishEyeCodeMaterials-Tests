#include <Ransac.h>
#include <OpencvCommon.h>
#include <memory>
#include <ctime>
#define MAIN_FILE
#include <commonMacro.h>


enum Method
{
	RASTER_SCAN, THRESHOLD_SEG
};

class CircleDetector
{
public:
	struct Circle
	{
		cv::Point2d center;
		double radius;
	};
	CircleDetector() {}
	~CircleDetector() {}

	virtual Circle detect(const cv::Mat &img) = 0;
};

class RasterScanDetector : public CircleDetector
{
public:
	RasterScanDetector() {}
	~RasterScanDetector() {}

	virtual Circle detect(const cv::Mat &img)
	{
		cv::Point2d circle_center;
		double radius;
		_getCircleRegion(img, circle_center, radius);
		return { circle_center, radius };
	}
	
private:

	void _getCircleRegion(const cv::Mat &img, cv::Point2d &center, double &radius)
	{
		std::vector<std::vector<int> > circle_points;
		_getCircleEdgePoints(img, circle_points);

		RansacCircle ransaccircle(10, 0.99, 2000, false);
		std::vector<char> inlier_mask;
		double interior_rate = ransaccircle.run(circle_points, inlier_mask);

		double x, y, radius_s;
		ransaccircle.getCircle(x, y, radius_s);

		radius = sqrt(radius_s);
		center = cv::Point2d(x, y);
		circle_points.clear();
	}

	void _getCircleEdgePoints(const cv::Mat &img, std::vector<std::vector<int> >&circle_points)
	{
		if (!circle_points.empty())circle_points.clear();
		double megapix = 0.5;
		int edge_black = 25;
		int big_black = 100;
		int shift_w_ratio = 50;
		int shift_h_ratio = 60;
		int window_windth_raio = 6;
		int grad_threshold_ratio = 20;
		int minX_threshold_ratio = 100;
		int m_black = 15;

		double work_scale = std::min(1.0, sqrt(megapix * 1e6 / img.size().area()));
		int step = 1.0 / work_scale;

		int W = img.cols;
		int H = img.rows;
		int half_H = H / 2;
		int half_W = W / 2;

		int shift_w = half_W / shift_w_ratio;
		int shift_h = half_H / shift_h_ratio;
		int window_width = half_W / window_windth_raio;
		int grad_threshold = grad_threshold_ratio * window_width;
		int minX_threshold = half_W / minX_threshold_ratio;

		int x_l, x_h, y_l, y_h;

		int channel_step = img.type() == CV_8UC3 ? 3 : 1;

		int c_half_W = channel_step * half_W;

		//设置大阈值找到边界框,和中心边界
		{
			//the coefficient that convert the bgr to grey value
			double coeffs[3] = { 0.114f, 0.587f, 0.299f };

			//得到最大最小X：minX ，maxX
			int minX = -1, maxX = W;
			const uchar * median_row_ptr = img.ptr(half_H);
			int temp_value;

			for (int i = 0; i < c_half_W; i += channel_step)
			{
				if (channel_step == 1)
				{
					temp_value = median_row_ptr[i];
				}
				else temp_value = coeffs[0] * median_row_ptr[i] + coeffs[1] * median_row_ptr[i + 1] + coeffs[2] * median_row_ptr[i + 2];
				if (temp_value <= edge_black)
					minX++;
				else break;
			}
			for (int i = channel_step * (img.cols - 1); i >= c_half_W; i -= channel_step)
			{
				if (channel_step == 1)
				{
					temp_value = median_row_ptr[i];
				}
				else temp_value = coeffs[0] * median_row_ptr[i] + coeffs[1] * median_row_ptr[i + 1] + coeffs[2] * median_row_ptr[i + 2];
				if (temp_value <= edge_black)
					maxX--;
				else break;
			}

			//得到最大最小Y：minY ，maxY
			int minY = -1, maxY = img.rows;
			for (size_t i = 0; i < half_H; i++)
			{
				const uchar * row_ptr = img.ptr(i);
				if (channel_step == 1)
				{
					temp_value = row_ptr[c_half_W];
				}
				else temp_value = coeffs[0] * row_ptr[c_half_W] + coeffs[1] * row_ptr[c_half_W + 1] + coeffs[2] * row_ptr[c_half_W + 2];
				if (temp_value <= edge_black)
					minY++;
			}
			for (size_t i = img.rows - 1; i > half_H; i--)
			{
				const uchar * row_ptr = img.ptr(i);
				if (channel_step == 1)
				{
					temp_value = row_ptr[c_half_W];
				}
				else temp_value = coeffs[0] * row_ptr[c_half_W] + coeffs[1] * row_ptr[c_half_W + 1] + coeffs[2] * row_ptr[c_half_W + 2];
				if (temp_value <= edge_black)
					maxY--;
			}

			x_l = minX == -1 ? 0 : minX;
			x_h = maxX == img.cols ? img.cols - 1 : maxX;
			y_l = minY == -1 ? 0 : minY;
			y_h = maxY == img.rows ? img.rows - 1 : maxY;

			y_l += shift_h;
			y_h -= shift_h;

			assert(x_l < half_W && x_h > half_W && y_l < half_H && y_h > half_H);
		}

		std::vector<int> row_integral(half_W + window_width + 1, 0);

		{
			double coeffs[3] = { 0.114f, 0.587f, 0.299f };
			assert(img.type() == CV_8UC3 || img.type() == CV_8UC1);


			//对下半图进行操作
			for (int j = half_H; j < y_h; j += step)
			{
				const uchar * down_row_ptr = img.ptr(j);

				//对左半图进行操作
				int max_length = 0;
				int minX1 = 0;
				row_integral[0] = 0;
				bool mini_clock = false;
				int temp_value;
				for (int i = x_l * channel_step; i < c_half_W; i += channel_step, max_length++)
				{
					if (channel_step == 1)
					{
						temp_value = down_row_ptr[i];
					}
					else temp_value = coeffs[0] * down_row_ptr[i] + coeffs[1] * down_row_ptr[i + 1] + coeffs[2] * down_row_ptr[i + 2];

					row_integral[max_length + 1] = row_integral[max_length] + temp_value;
					if (temp_value <= m_black && !mini_clock)minX1++;
					if (temp_value > m_black)mini_clock = true;
					if (temp_value >= big_black)
					{
						max_length++;
						break;
					}
				}
				if (max_length != 1 && minX1 != half_W - x_l && minX1 > minX_threshold)
				{
					int start = max_length + x_l;
					int end = start + window_width;
					start *= channel_step;
					end *= channel_step;
					int max_length_temp = max_length;
					for (int i = start; i < end; i += channel_step, max_length_temp++)
					{
						if (channel_step == 1)
						{
							temp_value = down_row_ptr[i];
						}
						else temp_value = coeffs[0] * down_row_ptr[i] + coeffs[1] * down_row_ptr[i + 1] + coeffs[2] * down_row_ptr[i + 2];

						row_integral[max_length_temp + 1] = row_integral[max_length_temp] + temp_value;
					}

					int max_diff = 0;
					int max_index = minX1;
					for (int i = minX1; i <= max_length; i++)
					{
						int left_value, right_value;
						if (i > window_width)
						{
							left_value = row_integral[i] - row_integral[i - window_width];
						}
						else left_value = row_integral[i];

						right_value = row_integral[i + window_width] - row_integral[i];

						double diff_temp = right_value - left_value;
						if (diff_temp > max_diff)
						{
							max_diff = diff_temp;
							max_index = i;
						}
						else if (max_diff > grad_threshold)break;
					}

					max_index += x_l;
					if (max_index >= shift_w)
					{
						max_index -= shift_w;
						std::vector<int> point(2);
						point[0] = max_index;
						point[1] = j;
						circle_points.push_back(point);
					}
				}

				//对右半图进行操作
				max_length = 0;
				row_integral[0] = 0;
				int minX2 = 0;
				mini_clock = false;
				for (int i = x_h * channel_step; i > c_half_W; i -= channel_step, max_length++)
				{
					if (channel_step == 1)
					{
						temp_value = down_row_ptr[i];
					}
					else temp_value = coeffs[0] * down_row_ptr[i] + coeffs[1] * down_row_ptr[i + 1] + coeffs[2] * down_row_ptr[i + 2];

					row_integral[max_length + 1] = row_integral[max_length] + temp_value;
					if (temp_value <= m_black && !mini_clock)minX2++;
					if (temp_value > m_black)mini_clock = true;
					if (temp_value >= big_black)
					{
						max_length++;
						break;
					}
				}
				if (max_length != 1 && minX2 != x_h - half_W && minX2 > minX_threshold)
				{
					int start = x_h - max_length;
					int end = start - window_width;
					start *= channel_step;
					end *= channel_step;
					int max_length_temp = max_length;
					for (int i = start; i > end; i -= channel_step, max_length_temp++)
					{
						if (channel_step == 1)
						{
							temp_value = down_row_ptr[i];
						}
						else temp_value = coeffs[0] * down_row_ptr[i] + coeffs[1] * down_row_ptr[i + 1] + coeffs[2] * down_row_ptr[i + 2];
						row_integral[max_length_temp + 1] = row_integral[max_length_temp] + temp_value;
					}

					int max_diff = 0;
					int max_index = minX2;
					for (int i = minX2; i <= max_length; i++)
					{
						int left_value, right_value;
						if (i > window_width)
						{
							left_value = row_integral[i] - row_integral[i - window_width];
						}
						else left_value = row_integral[i];

						right_value = row_integral[i + window_width] - row_integral[i];

						double diff_temp = right_value - left_value;
						if (diff_temp > max_diff)
						{
							max_diff = diff_temp;
							max_index = i;
						}
						else if (max_diff > grad_threshold)break;
					}

					max_index = x_h - max_index;
					if (W - max_index > shift_w)
					{
						max_index += shift_w;
						std::vector<int> point(2);
						point[0] = max_index;
						point[1] = j;
						circle_points.push_back(point);
					}
				}

				if (minX1 == half_W - x_l && minX2 == x_h - half_W)
					break;
			}

			//对上半图进行操作
			for (int j = half_H; j > y_l; j -= step)
			{
				const uchar * up_row_ptr = img.ptr(j);

				//对左半图进行操作
				int max_length = 0;
				int minX1 = 0;
				row_integral[0] = 0;
				int temp_value;
				bool mini_clock = false;
				for (int i = x_l * channel_step; i < c_half_W; i += channel_step, max_length++)
				{
					if (channel_step == 1)
					{
						temp_value = up_row_ptr[i];
					}
					else temp_value = coeffs[0] * up_row_ptr[i] + coeffs[1] * up_row_ptr[i + 1] + coeffs[2] * up_row_ptr[i + 2];
					row_integral[max_length + 1] = row_integral[max_length] + temp_value;
					if (temp_value <= m_black && !mini_clock)minX1++;
					if (temp_value > m_black)mini_clock = true;
					if (temp_value >= big_black)
					{
						max_length++;
						break;
					}
				}
				if (max_length != 1 && minX1 != half_W - x_l&& minX1 > minX_threshold)
				{
					int start = max_length + x_l;
					int end = start + window_width;
					start *= channel_step;
					end *= channel_step;
					int max_length_temp = max_length;
					for (int i = start; i < end; i += channel_step, max_length_temp++)
					{
						if (channel_step == 1)
						{
							temp_value = up_row_ptr[i];
						}
						else temp_value = coeffs[0] * up_row_ptr[i] + coeffs[1] * up_row_ptr[i + 1] + coeffs[2] * up_row_ptr[i + 2];
						row_integral[max_length_temp + 1] = row_integral[max_length_temp] + temp_value;
					}

					int max_diff = 0;
					int max_index = minX1;
					for (int i = minX1; i <= max_length; i++)
					{
						int left_value, right_value;
						if (i > window_width)
						{
							left_value = row_integral[i] - row_integral[i - window_width];
						}
						else left_value = row_integral[i];

						right_value = row_integral[i + window_width] - row_integral[i];

						double diff_temp = right_value - left_value;
						if (diff_temp > max_diff)
						{
							max_diff = diff_temp;
							max_index = i;
						}
						else if (max_diff > grad_threshold)break;
					}

					max_index += x_l;
					if (max_index >= shift_w)
					{
						max_index -= shift_w;
						std::vector<int> point(2);
						point[0] = max_index;
						point[1] = j;
						circle_points.push_back(point);
					}
				}

				//对右半图进行操作
				max_length = 0;
				row_integral[0] = 0;
				int minX2 = 0;
				mini_clock = false;
				for (int i = x_h * channel_step; i > c_half_W; i -= channel_step, max_length++)
				{
					if (channel_step == 1)
					{
						temp_value = up_row_ptr[i];
					}
					else temp_value = coeffs[0] * up_row_ptr[i] + coeffs[1] * up_row_ptr[i + 1] + coeffs[2] * up_row_ptr[i + 2];
					row_integral[max_length + 1] = row_integral[max_length] + temp_value;
					if (temp_value <= m_black && !mini_clock)minX2++;
					if (temp_value > m_black)mini_clock = true;
					if (temp_value >= big_black)
					{
						max_length++;
						break;
					}
				}
				if (max_length != 1 && minX2 != x_h - half_W && minX2 > minX_threshold)
				{
					int start = x_h - max_length;
					int end = start - window_width;
					start *= channel_step;
					end *= channel_step;
					int max_length_temp = max_length;
					for (int i = start; i > end; i -= channel_step, max_length_temp++)
					{
						if (channel_step == 1)
						{
							temp_value = up_row_ptr[i];
						}
						else temp_value = coeffs[0] * up_row_ptr[i] + coeffs[1] * up_row_ptr[i + 1] + coeffs[2] * up_row_ptr[i + 2];
						row_integral[max_length_temp + 1] = row_integral[max_length_temp] + temp_value;
					}

					int max_diff = 0;
					int max_index = minX2;
					for (int i = minX2; i <= max_length; i++)
					{
						int left_value, right_value;
						if (i > window_width)
						{
							left_value = row_integral[i] - row_integral[i - window_width];
						}
						else left_value = row_integral[i];

						right_value = row_integral[i + window_width] - row_integral[i];

						double diff_temp = right_value - left_value;
						if (diff_temp > max_diff)
						{
							max_diff = diff_temp;
							max_index = i;
						}
						else if (max_diff > grad_threshold)break;
					}

					max_index = x_h - max_index;
					if (W - max_index > shift_w)
					{
						max_index += shift_w;
						std::vector<int> point(2);
						point[0] = max_index;
						point[1] = j;
						circle_points.push_back(point);
					}
				}

				if (minX1 == half_W - x_l && minX2 == x_h - half_W)
					break;
			}
		}
	}
};




int main(int argc, char *argv[])
{
	std::string filename = "2S7A7011.jpg";
	std::shared_ptr<CircleDetector> pCDetector = std::make_shared<RasterScanDetector>();

	cv::Mat img = cv::imread(filename);
	CircleDetector::Circle circle;
	IntevalTime(circle = pCDetector->detect(img));
	IntevalTime(cv::circle(img, circle.center, circle.radius, cv::Scalar(0, 0, 255), 5));
	cv::imwrite("circleResult.jpg", img);
	
	return 0;
}

