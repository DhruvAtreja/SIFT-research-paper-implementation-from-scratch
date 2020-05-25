#include <opencv2\core\core.hpp>
#include <list>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <iostream>
using namespace std;
using namespace cv;

/** default sigma for initial gaussian smoothing */
#define SIFT_FIRST_SIGMA 1.6
#define SIFT_OCTAVE 9
#define SIFT_SCALE_DEGREE 3 
#define SIFT_HESSIAN_EDGE_THR 3
#define SIFT_DoG_THR 3
/* default number of bins in histogram for orientation assignment */
#define SIFT_ORI_HIST_BINS 36
/* determines gaussian sigma for orientation assignment */
#define SIFT_ORI_SIG_FCTR 1.5

/* determines the radius of the region used in orientation assignment */
#define SIFT_ORI_RADIUS 3.0 * SIFT_ORI_SIG_FCTR
#define FEATURE_MAX_D 128
#define EXTREMA_DIST_THR 0.1
struct feature
{
	double x;                      /**< x coord */
	double y;                      /**< y coord */
	double scl;                    /**< scale of a Lowe-style feature */
	double ori_vec_angle;                    /**< orientation of a Lowe-style feature */
	double ori_vec_mag;
	double descr[FEATURE_MAX_D];   /**< descriptor */
	double im_size;
	feature(const double& x, const double& y,
		const double& scl, const double im_size) :x(x), y(y), scl(scl), im_size(im_size)
	{

	}
};

int get_sift_descriptor(Mat& img, list<feature>& feature_list);
bool show_sift_desc(Mat& img, list<feature>& feature_list, const char out_file_name[]);

#define PI 3.1415927
#define LITTLE 0.000001

int row = 5;
bool _create_gaus_pyr(Mat& base_img, Mat* Octaves[SIFT_OCTAVE][SIFT_SCALE_DEGREE + 3], int Octave_num, double* sigmas);
bool _get_extrema(Mat* DoG_octaves[SIFT_OCTAVE][SIFT_SCALE_DEGREE + 2], Mat* Gaussian_Octaves[SIFT_OCTAVE][SIFT_SCALE_DEGREE + 3], list<feature>& feature_list, double* sigmas);
Mat* get_dog(Mat* Gaus_img, Mat* Next_Gaus_img);
bool _erase_dog_octaves(Mat* Octaves[SIFT_OCTAVE][SIFT_SCALE_DEGREE + 2], int octave_num, int scale_num);
bool _erase_gaus_octaves(Mat* Octaves[SIFT_OCTAVE][SIFT_SCALE_DEGREE + 3], int octave_num, int scale_num);
bool _is_extremum(Mat* p_forward_DoG, Mat* p_this_DoG, Mat* p_Next_DoG, int r, int c);
int _calc_mag_angle(int bins_size, int x, int y, Mat* pGaussian_img, double& tmp_mag);
bool _get_dog_pyr(Mat* Gaussian_Octaves[SIFT_OCTAVE][SIFT_SCALE_DEGREE + 3], Mat* DoG_octaves[SIFT_OCTAVE][SIFT_SCALE_DEGREE + 2]);
bool _get_accurate_pos(Mat* p_forward_DoG, Mat* p_this_DoG, Mat* p_Next_DoG, int& x, int& y, double sigmas[3]);
bool _get_feature_ori(feature& new_feature, Mat* pGaussian_img);


bool show_sift_desc(Mat& img, list<feature>& feature_list, const char out_file_name[])
{
	double p2_x = 0, p2_y = 0;
	double x = 0, y = 0;
	auto it = feature_list.begin();
	for (; it != feature_list.end(); ++it)
	{
		x = it->x * it->im_size;
		y = it->y * it->im_size;
		p2_x = x + it->ori_vec_mag / 10 * cos(it->ori_vec_angle);
		p2_y = y + it->ori_vec_mag / 10 * sin(it->ori_vec_angle);
		arrowedLine(img, Point(x, y), Point(p2_x, p2_y), Scalar(255, 0, 0), 1);
		circle(img, Point(it->x, it->y), 2, Scalar(0, 255, 0), 3, LINE_AA);
	}
	
	namedWindow("Display window", WINDOW_AUTOSIZE);// Create a window for display.
	imshow("Display window", img);
	return true;
}
int get_sift_descriptor(Mat& img, list<feature>& feature_list)
{
	Mat* Gaussian_Octaves[SIFT_OCTAVE][SIFT_SCALE_DEGREE + 3];
	Mat* DoG_octaves[SIFT_OCTAVE][SIFT_SCALE_DEGREE + 2];
	double sigmas[SIFT_SCALE_DEGREE + 3];
	double* p_sigmas = sigmas;
	double k = pow(2.0, 1.0 / SIFT_SCALE_DEGREE);
	for (int i = 0; i < SIFT_SCALE_DEGREE + 3; ++i)
	{
		sigmas[i] = SIFT_FIRST_SIGMA * pow(k, i - 1);
	}

	_create_gaus_pyr(img, Gaussian_Octaves, SIFT_OCTAVE, p_sigmas);
	_get_dog_pyr(Gaussian_Octaves, DoG_octaves);
	_get_extrema(DoG_octaves, Gaussian_Octaves, feature_list, sigmas);

	_erase_gaus_octaves(Gaussian_Octaves, SIFT_OCTAVE, SIFT_SCALE_DEGREE + 3);
	_erase_dog_octaves(DoG_octaves, SIFT_OCTAVE, SIFT_SCALE_DEGREE + 2);
	return 0;
}

bool _get_extrema(Mat* DoG_octaves[SIFT_OCTAVE][SIFT_SCALE_DEGREE + 2], Mat* Gaussian_Octaves[SIFT_OCTAVE][SIFT_SCALE_DEGREE + 3], list<feature>& feature_list, double* sigmas)
{
	int x_extr = 0, y_extr = 0;
	double im_size = 0;
	for (int oct = 0; oct < SIFT_OCTAVE; ++oct)
	{
		for (int i = 1; i < SIFT_SCALE_DEGREE + 1; ++i)
		{
			for (int r = 0; r < DoG_octaves[oct][i]->rows; ++r)
			{
				for (int c = 0; c < DoG_octaves[oct][i]->cols; ++c)
				{
					if (_is_extremum(DoG_octaves[oct][i - 1], DoG_octaves[oct][i], DoG_octaves[oct][i + 1], r, c))
					{
						//Got an extrema
						x_extr = c;
						y_extr = r;
						if (false == _get_accurate_pos(DoG_octaves[oct][i - 1], DoG_octaves[oct][i], \
							DoG_octaves[oct][i + 1], x_extr, y_extr, sigmas + i - 1))
						{
							// D(x,y) is too low.
							continue;
						}
						else
						{
							// record this new feature point.
							im_size = double(DoG_octaves[0][0]->cols) / double(DoG_octaves[oct][i]->cols);
							feature new_feature(x_extr, y_extr, sigmas[i], im_size);
							// get the hist of this new feature.
							if (_get_feature_ori(new_feature, Gaussian_Octaves[oct][i]))
							{
								feature_list.push_back(new_feature);
							}
						}

					}
				}
			}
		}
	}
	return true;
}
bool _get_feature_ori(feature& new_feature, Mat* pGaussian_img)
{
	double x = new_feature.x, y = new_feature.y;
	double weight_sigma = SIFT_ORI_SIG_FCTR * new_feature.scl;
	double tmp_mag = 0;
	double exp_denom = 2.0 * weight_sigma * weight_sigma;
	double w = 0;
	int range_radius = SIFT_ORI_RADIUS * new_feature.scl;
	int bin_id = 0;
	int i = -range_radius;
	int j = -range_radius;

	double p_hist_bins[SIFT_ORI_HIST_BINS] = { 0 };
	if (x < range_radius + 1 || x > pGaussian_img->cols - range_radius - 1 || y < range_radius + 1 || y > pGaussian_img->rows - range_radius - 1)
	{
		return false; // this point is too close to the boundary of the image.
	}
	for (i = -range_radius; i < range_radius; ++i)
	{
		for (j = -range_radius; j < range_radius; ++j)
		{
			if (i > 10000)
			{
				throw;
			}
			bin_id = _calc_mag_angle(SIFT_ORI_HIST_BINS, int(x) + i, int(y) + j, pGaussian_img, tmp_mag);
			w = exp(-(i * i + j * j) / exp_denom);
			p_hist_bins[bin_id] += tmp_mag * w;
		}
	}
	// find the max orientation.
	tmp_mag = p_hist_bins[0];
	bin_id = 0;
	for (int i = 1; i < SIFT_ORI_HIST_BINS; ++i)
	{
		if (p_hist_bins[i] > tmp_mag)
		{
			tmp_mag = p_hist_bins[i];
			bin_id = i;
		}
	}
	new_feature.ori_vec_angle = (bin_id + 1) * 2 * PI / SIFT_ORI_HIST_BINS;
	new_feature.ori_vec_mag = tmp_mag;
	return true;
}
int _calc_mag_angle(int bins_size, int x, int y, Mat* pGaussian_img, double& tmp_mag)
{
	double dx = 2.0 * pGaussian_img->at<float>(y, x + 1) + pGaussian_img->at<float>(y + 1, x + 1) + pGaussian_img->at<float>(y - 1, x + 1);
	dx -= 2.0 * pGaussian_img->at<float>(y, x - 1) + pGaussian_img->at<float>(y + 1, x - 1) + pGaussian_img->at<float>(y - 1, x - 1);
	double dy = 2.0 * pGaussian_img->at<float>(y + 1, x) + pGaussian_img->at<float>(y + 1, x + 1) + pGaussian_img->at<float>(y + 1, x - 1);
	dy -= 2.0 * pGaussian_img->at<float>(y - 1, x) + pGaussian_img->at<float>(y - 1, x + 1) + pGaussian_img->at<float>(y - 1, x - 1);
	tmp_mag = sqrt(dx * dx + dy * dy);
	tmp_mag += LITTLE;
	// get the angle of (dx, dy) vector.
	double angle = 0;
	int bin_id = 0;
	if (dy > 0)
	{
		angle = acos(dx / tmp_mag);
	}
	else
	{
		angle = 2 * PI - acos(dx / tmp_mag);
	}
	bin_id = angle / (2 * PI / bins_size);
	if (bin_id > bins_size || bin_id < 0)
	{
		throw;
	}
	return bin_id;
}
/*
* Calculate the accurate position fo the extrema. And eliminate it if the D(x, y) is too low.
*/
bool _get_accurate_pos(Mat* p_forward_DoG, Mat* p_this_DoG, Mat* p_Next_DoG, int& x, int& y, double sigmas[3])
{
	double dx = (p_this_DoG->at<float>(y, x + 1) - p_this_DoG->at<float>(y, x - 1)) / 2.0;
	double dy = (p_this_DoG->at<float>(y + 1, x) - p_this_DoG->at<float>(y - 1, x)) / 2.0;
	double d_theta = (p_forward_DoG->at<float>(y, x) - p_Next_DoG->at<float>(y, x)) / 2.0;
	float D_1diff[3][1] = { { dx }, { dy }, { d_theta } };
	Mat D1_Mat(3, 1, CV_32FC1, D_1diff);
	double this_val = p_this_DoG->at<float>(y, x);
	double dxx = p_this_DoG->at<float>(y, x + 1) + p_this_DoG->at<float>(y, x - 1) - 2 * this_val;
	double dxy = p_this_DoG->at<float>(y + 1, x + 1) - p_this_DoG->at<float>(y + 1, x - 1) - \
		p_this_DoG->at<float>(y - 1, x + 1) + p_this_DoG->at<float>(y - 1, x - 1);
	dxy /= 4.0;
	double dx_theta = (p_forward_DoG->at<float>(y, x + 1) - p_forward_DoG->at<float>(y, x - 1) - \
		p_Next_DoG->at<float>(y, x + 1) + p_forward_DoG->at<float>(y, x - 1)) / 2 / (*sigmas - *(sigmas + 2));
	double dyy = p_this_DoG->at<float>(y + 1, x) + p_this_DoG->at<float>(y - 1, x) - 2 * this_val;
	double dy_theta = (p_forward_DoG->at<float>(y + 1, x) - p_forward_DoG->at<float>(y - 1, x) - \
		p_Next_DoG->at<float>(y + 1, x) + p_Next_DoG->at<float>(y - 1, x)) / 2 / (*sigmas - *(sigmas + 2));
	double d_theta_theta = (p_forward_DoG->at<float>(y, x) - this_val) / (*sigmas - *(sigmas + 1)) - \
		(this_val - p_Next_DoG->at<float>(y, x)) / (*(sigmas + 2) - *(sigmas + 1));
	d_theta_theta /= (*sigmas - *(sigmas + 2)) / 2;

	float D_2diff[3][3] = { { dxx, dxy, dx_theta }, { dxy, dyy, dy_theta }, {dx_theta, dy_theta, d_theta_theta} };
	Mat D2_Mat(3, 3, CV_32FC1, D_2diff);
	Mat deta = -D2_Mat.inv() * D1_Mat;
	if (norm(deta) > 0.5)
	{
		this_val = this_val + 0.5 * D1_Mat.dot(deta);
	}
	if (this_val < SIFT_DoG_THR)
	{
		return false;// discard this point.
	}
	// Change the coordination to the accurate one.
	x += deta.at<float>(0, 0);
	y += deta.at<float>(1, 0);
	// transfer (x,y) from Pyramid coordination to image coordination.
	//x *= im_size;
	//y *= im_size;
	return true;
}

bool _get_dog_pyr(Mat* Gaussian_Octaves[SIFT_OCTAVE][SIFT_SCALE_DEGREE + 3], Mat* DoG_octaves[SIFT_OCTAVE][SIFT_SCALE_DEGREE + 2])
{
	for (int i = 0; i < SIFT_OCTAVE; ++i)
	{
		for (int j = 1; j < SIFT_SCALE_DEGREE + 3; ++j)
		{
			DoG_octaves[i][j - 1] = get_dog(Gaussian_Octaves[i][j - 1], Gaussian_Octaves[i][j]);
		}
	}
	return true;
}

Mat* get_dog(Mat* Gaus_img, Mat* Next_Gaus_img)
{
	Mat* pDoG = new Mat(Gaus_img->size(), CV_32FC1);
	if (pDoG == nullptr)
	{
		throw;
	}
	//MatIterator_<float> it1 = Gaus_img->begin<float>(), end1 = Gaus_img->end<float>();
	//MatIterator_<float> it2 = Next_Gaus_img->begin<float>(), end2 = Gaus_img->endl<float>();
	//MatIterator_<float> it_dot = pDoG->begin<float>();
	//while (it1 != end1)
	//{

	//}
	subtract(*Gaus_img, *Next_Gaus_img, *pDoG);
	return pDoG;
}

bool _is_extremum(Mat* p_forward_DoG, Mat* p_this_DoG, Mat* p_Next_DoG, int r, int c)
{
	int dx = 0;
	int dy = 0;
	double thr = 0;
	double index = 0;
	double max_abs = 0;
	if (r <= 0 || r >= p_this_DoG->rows - 1)
	{
		return false;
	}
	if (c <= 0 || c >= p_this_DoG->cols - 1)
	{
		return false;
	}
	bool is_max = true, is_min = true;
	float this_val = p_this_DoG->at<float>(r, c);
	// check if it is a maximum.
	if (this_val >= 0)
	{
		for (dx = -1; dx <= 1; ++dx)
		{
			for (dy = -1; dy <= 1; ++dy)
			{
				if (abs(p_forward_DoG->at<float>(r + dy, c + dx) - this_val) > max_abs)
				{
					max_abs = p_forward_DoG->at<float>(r + dy, c + dx) - this_val;
				}
				if (p_forward_DoG->at<float>(r + dy, c + dx) > this_val)
				{
					return false;
				}
			}

		}
		for (dx = -1; dx <= 1; ++dx)
		{
			for (dy = -1; dy <= 1; ++dy)
			{
				if (dx == 0 && dy == 0)
				{
					continue;
				}
				if (abs(p_forward_DoG->at<float>(r + dy, c + dx) - this_val) > max_abs)
				{
					max_abs = p_forward_DoG->at<float>(r + dy, c + dx) - this_val;
				}
				if (p_forward_DoG->at<float>(r + dy, c + dx) > this_val)
				{
					return false;
				}
			}
		}

	}
	// check if it is a minimum.
	else if (this_val < 0)
	{
		for (int dx = -1; dx <= 1; ++dx)
			for (int dy = -1; dy <= 1; ++dy)
			{
				if (abs(p_forward_DoG->at<float>(r + dy, c + dx) - this_val) > max_abs)
				{
					max_abs = p_forward_DoG->at<float>(r + dy, c + dx) - this_val;
				}
				if (p_forward_DoG->at<float>(r + dy, c + dx) < this_val)
				{
					return false;
				}
			}

		for (int dx = -1; dx <= 1; ++dx)
			for (int dy = -1; dy <= 1; ++dy)
			{
				if (dx == 0 && dy == 0)
				{
					continue;
				}
				if (abs(p_forward_DoG->at<float>(r + dy, c + dx) - this_val) > max_abs)
				{
					max_abs = p_forward_DoG->at<float>(r + dy, c + dx) - this_val;
				}
				if (p_forward_DoG->at<float>(r + dy, c + dx) < this_val)
				{
					return false;
				}
			}
	}
	//if (max_abs < EXTREMA_DIST_THR)
	//{
	//	return false;
	//}
	// Check if it is a point in edge.
	double dxx = p_this_DoG->at<float>(r, c + 1) + p_this_DoG->at<float>(r, c - 1) - 2 * this_val;
	double dyy = p_this_DoG->at<float>(r + 1, c) + p_this_DoG->at<float>(r - 1, c) - 2 * this_val;
	double dxy = p_this_DoG->at<float>(r + 1, c + 1) - p_this_DoG->at<float>(r + 1, c - 1) + \
		p_this_DoG->at<float>(r - 1, c + 1) - p_this_DoG->at<float>(r - 1, c - 1);
	dxy /= 4.0;
	double tr = dxx + dyy;
	double det = dxx * dyy - pow(dxy, 2);
	if (det < 0)
	{
		return false;
	}
	thr = pow(SIFT_HESSIAN_EDGE_THR + 1, 2) / SIFT_HESSIAN_EDGE_THR;
	index = pow(tr, 2) / det;

	if (index > thr)
	{
		return false;
	}
	else
	{
		return true;
	}

}

bool _erase_dog_octaves(Mat* Octaves[SIFT_OCTAVE][SIFT_SCALE_DEGREE + 2], int octave_num, int scale_num)
{
	for (int i = 0; i < octave_num; ++i)
	{
		for (int j = 0; j < scale_num; ++j)
		{
			delete Octaves[i][j];
		}
	}
	return true;
}
bool _erase_gaus_octaves(Mat* Octaves[SIFT_OCTAVE][SIFT_SCALE_DEGREE + 3], int octave_num, int scale_num)
{
	for (int i = 0; i < octave_num; ++i)
	{
		for (int j = 0; j < scale_num; ++j)
		{
			delete Octaves[i][j];
		}
	}
	return true;
}

bool _create_gaus_pyr(Mat& base_img, Mat* Octaves[SIFT_OCTAVE][SIFT_SCALE_DEGREE + 3], int Octave_num, double* sigmas)
{

	Mat copy_base_im(base_img);

	for (int i = 0; i < Octave_num; i++)
	{
		if (i > 0)
		{
			copy_base_im.resize(copy_base_im.rows / 2);
		}
		for (int j = 0; j < SIFT_SCALE_DEGREE + 3; ++j)
		{
			Octaves[i][j] = new Mat(copy_base_im.rows, copy_base_im.cols, CV_32FC1);
			if (Octaves[i][j] == nullptr)
			{
				throw;
			}
			GaussianBlur(copy_base_im, *Octaves[i][j], Size(0, 0), sigmas[j], 0);
		}
	}
	return true;
}


int main(int argc, char** argv)
{

	Mat img = imread("C:\\Users\\dhruv\\Desktop\\corner\\index1.jpg", IMREAD_GRAYSCALE);
	Mat f_img(img.size(), CV_32FC1);
	img.convertTo(f_img, CV_32FC1);
	//imwrite("f_img.jpg", f_img);

	list<feature> feature_list;
	get_sift_descriptor(f_img, feature_list);
	show_sift_desc(f_img, feature_list, "C:\\Users\\dhruv\\Desktop\\corner");
}
