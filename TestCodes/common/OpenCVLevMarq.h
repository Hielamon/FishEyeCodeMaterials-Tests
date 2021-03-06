#pragma once

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/utility.hpp>


#include <opencv2/core/ocl.hpp>

namespace cv
{

class CV_EXPORTS LMSolver : public Algorithm
{
public:
    class CV_EXPORTS Callback
    {
    public:
        virtual ~Callback() {}
        virtual bool compute(InputArray param, OutputArray err, OutputArray J) const = 0;
    };

    virtual void setCallback(const Ptr<LMSolver::Callback>& cb) = 0;
    virtual int run(InputOutputArray _param0) const = 0;
};

class LMSolverImpl : public LMSolver
{
public:
	LMSolverImpl() : maxIters(100) { init(); }
	LMSolverImpl(const Ptr<LMSolver::Callback>& _cb, int _maxIters) : cb(_cb), maxIters(_maxIters) { init(); }
	LMSolverImpl(const Ptr<LMSolver::Callback>& _cb, int _maxIters, double _epsx, double _epsf, std::string _logFileName) :
		cb(_cb), maxIters(_maxIters), epsx(_epsx), epsf(_epsf), logFileName(_logFileName)
	{
		printInterval = 0;
	}

	void init()
	{
		epsx = epsf = FLT_EPSILON;
		printInterval = 0;
	}

	int run(InputOutputArray _param0) const
	{
		Mat param0 = _param0.getMat(), x, xd, r, rd, J, A, Ap, v, temp_d, d;
		int ptype = param0.type();

		CV_Assert((param0.cols == 1 || param0.rows == 1) && (ptype == CV_32F || ptype == CV_64F));
		CV_Assert(cb);

		int lx = param0.rows + param0.cols - 1;
		param0.convertTo(x, CV_64F);

		if (x.cols != 1)
			transpose(x, x);

		if (!cb->compute(x, r, J))
			return -1;
		double S = norm(r, NORM_L2SQR);
		int nfJ = 2;

		mulTransposed(J, A, true);
		gemm(J, r, 1, noArray(), 0, v, GEMM_1_T);

		Mat D = A.diag().clone();

		const double Rlo = 0.25, Rhi = 0.75;
		double lambda = 1, lc = 0.75;
		int i, iter = 0;

		if (printInterval != 0)
		{
			printf("************************************************************************************\n");
			printf("\titr\tnfJ\t\tSUM(r^2)\t\tx\t\tdx\t\tl\t\tlc\n");
			printf("************************************************************************************\n");
		}

		//printf("************************************************************************************\n");
		std::ofstream fs(logFileName, std::ios::out);
		bool isLog = fs.is_open();

		if (isLog)
		{
			fs << iter << " " << norm(r) << std::endl;
		}

		for (;; )
		{
			CV_Assert(A.type() == CV_64F && A.rows == lx);
			A.copyTo(Ap);
			for (i = 0; i < lx; i++)
				Ap.at<double>(i, i) += lambda*D.at<double>(i);
			solve(Ap, v, d, DECOMP_EIG);
			subtract(x, d, xd);
			if (!cb->compute(xd, rd, noArray()))
			{
				rd = r.clone() * 10;
			}

			nfJ++;
			double Sd = norm(rd, NORM_L2SQR);
			gemm(A, d, -1, v, 2, temp_d);
			double dS = d.dot(temp_d);
			double R = (S - Sd) / (fabs(dS) > DBL_EPSILON ? dS : 1);

			if (R > Rhi)
			{
				lambda *= 0.5;
				if (lambda < lc)
					lambda = 0;
			}
			else if (R < Rlo)
			{
				// find new nu if R too low
				double t = d.dot(v);
				double nu = (Sd - S) / (fabs(t) > DBL_EPSILON ? t : 1) + 2;
				nu = std::min(std::max(nu, 2.), 10.);
				if (lambda == 0)
				{
					invert(A, Ap, DECOMP_EIG);
					double maxval = DBL_EPSILON;
					for (i = 0; i < lx; i++)
						maxval = std::max(maxval, std::abs(Ap.at<double>(i, i)));
					lambda = lc = std::max(1. / maxval, 0.01);
					//lambda = lc = 1./maxval;
					nu *= 0.5;
				}
				lambda *= nu;
			}

			if (Sd < S)
			{
				nfJ++;
				S = Sd;
				std::swap(x, xd);
				if (!cb->compute(x, r, J))
					return -1;
				mulTransposed(J, A, true);
				gemm(J, r, 1, noArray(), 0, v, GEMM_1_T);
			}

			iter++;
			bool proceed = iter < maxIters && norm(d, NORM_INF) >= epsx && norm(r, NORM_INF) >= epsf;

			/*printf("iter=%d    error=%f    params=", iter, norm(r));
			for (size_t i = 0; i < x.rows; i++)
			{
			printf("%f ", x.at<double>(i, 0));
			}
			printf("\n");*/

			if (isLog)
			{
				fs << iter << " " << norm(r) << std::endl;
			}

			if (printInterval != 0 && (iter % printInterval == 0 || iter == 1 || !proceed))
			{
				printf("%c%10d %10d %15.4e %16.4e %17.4e %16.4e %17.4e\n",
					(proceed ? ' ' : '*'), iter, nfJ, S, x.at<double>(0), d.at<double>(0), lambda, lc);
			}

			if (!proceed)
				break;
		}

		if (param0.size != x.size)
			transpose(x, x);

		x.convertTo(param0, ptype);
		if (iter == maxIters)
			iter = -iter;

		return iter;
	}

	void setCallback(const Ptr<LMSolver::Callback>& _cb) { cb = _cb; }

	Ptr<LMSolver::Callback> cb;

	double epsx;
	double epsf;
	int maxIters;
	int printInterval;
	std::string logFileName;
};

Ptr<LMSolver> customCreateLMSolver(const Ptr<LMSolver::Callback>& cb, int maxIters, double _epsx, double _epsf, std::string _logFileName)
{
	return makePtr<LMSolverImpl>(cb, maxIters, _epsx, _epsf, _logFileName);
}

} // namespace cv

