#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <ctime>

#ifndef __linux
#include <io.h> 
#define access _access_s
#else
#include <unistd.h>
#include <memory>
#endif

#define POSITIVE_TRAINING_SET_PATH "DATASET\\POSITIVE\\"
#define NEGATIVE_TRAINING_SET_PATH "DATASET\\NEGATIVE\\"
#define WINDOW_NAME "WINDOW"
#define TRAFFIC_VIDEO_FILE "video.mp4"
#define TRAINED_SVM "vehicle_detector.yml"
#define	IMAGE_SIZE Size(40, 40) 
#define IS_VIDEO 1
using namespace cv;
using namespace cv::ml;
using namespace std;

bool file_exists(const string &file);
void load_images(string directory, vector<Mat>& image_list);
vector<string> files_in_directory(string directory);

void get_svm_detector(const Ptr<SVM>& svm, vector< float > & hog_detector);
void convert_to_ml(const std::vector< cv::Mat > & train_samples, cv::Mat& trainData);
void sample_neg(const vector< Mat > & full_neg_lst, vector< Mat > & neg_lst, const Size & size);
Mat get_hogdescriptor_visu(const Mat& color_origImg, vector<float>& descriptorValues, const Size & size);
void compute_hog(const vector< Mat > & img_lst, vector< Mat > & gradient_lst, const Size & size);
void train_svm(const vector< Mat > & gradient_lst, const vector< int > & labels);
void draw_locations(Mat & img, const vector< Rect > & locations, const Scalar & color);
void test_it(const Size & size);

int main(int argc, char** argv)
{
	if (!file_exists(TRAINED_SVM)) {

		vector< Mat > pos_lst;
		vector< Mat > full_neg_lst;
		vector< Mat > neg_lst;
		vector< Mat > gradient_lst;
		vector< int > labels;

		std::cout << "Loading Positive data..." << endl << endl;
		load_images(POSITIVE_TRAINING_SET_PATH, pos_lst);
		labels.assign(pos_lst.size(), +1);

		std::cout << "Loading Negative data..." << endl << endl;
		load_images(NEGATIVE_TRAINING_SET_PATH, full_neg_lst);
		labels.insert(labels.end(), full_neg_lst.size(), -1);

		std::cout << "Feature Extraction..." << endl << endl;

		compute_hog(pos_lst, gradient_lst, IMAGE_SIZE);
		compute_hog(full_neg_lst, gradient_lst, IMAGE_SIZE);

		train_svm(gradient_lst, labels);
	}

	test_it(IMAGE_SIZE);
	getchar();
	return 0;
}

bool file_exists(const string &file)
{
	return access(file.c_str(), 0) == 0;
}

vector<string> files_in_directory(string directory)
{
	vector<string> files;
	char buf[256];
	string command;

#ifdef __linux__ 
	command = "ls " + directory;
	shared_ptr<FILE> pipe(popen(command.c_str(), "r"), pclose);

	char cwd[256];
	getcwd(cwd, sizeof(cwd));

	while (!feof(pipe.get()))
		if (fgets(buf, 256, pipe.get()) != NULL) {
			string file(cwd);
			file.append("/");
			file.append(buf);
			file.pop_back();
			files.push_back(file);
		}
#else
	command = "dir /b /s " + directory;
	FILE* pipe = NULL;

	if (pipe = _popen(command.c_str(), "rt"))
		while (!feof(pipe))
			if (fgets(buf, 256, pipe) != NULL) {
				string file(buf);
				file.pop_back();
				files.push_back(file);
			}
	_pclose(pipe);
#endif

	return files;
}

void load_images(string directory, vector<Mat>& image_list) {

	Mat img;
	vector<string> files;
	files = files_in_directory(directory);

	for (int i = 0; i < files.size(); ++i) {

		img = imread(files.at(i));
		if (img.empty())
			continue;
#ifdef _DEBUG
		imshow("image", img);
		waitKey(10);
#endif
		resize(img, img, IMAGE_SIZE);
		image_list.push_back(img.clone());
	}
}

void get_svm_detector(const Ptr<SVM>& svm, vector< float > & hog_detector)
{
	Mat sv = svm->getSupportVectors();
	const int sv_total = sv.rows;
	
	Mat alpha, svidx;
	double rho = svm->getDecisionFunction(0, alpha, svidx);

	CV_Assert(alpha.total() == 1 && svidx.total() == 1 && sv_total == 1);
	CV_Assert((alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||
		(alpha.type() == CV_32F && alpha.at<float>(0) == 1.f));
	CV_Assert(sv.type() == CV_32F);
	hog_detector.clear();

	hog_detector.resize(sv.cols + 1);
	memcpy(&hog_detector[0], sv.ptr(), sv.cols * sizeof(hog_detector[0]));
	hog_detector[sv.cols] = (float)-rho;
}


void convert_to_ml(const std::vector< cv::Mat > & train_samples, cv::Mat& trainData)
{
	const int rows = (int)train_samples.size();
	const int cols = (int)std::max(train_samples[0].cols, train_samples[0].rows);
	cv::Mat tmp(1, cols, CV_32FC1);
	trainData = cv::Mat(rows, cols, CV_32FC1);
	vector< Mat >::const_iterator itr = train_samples.begin();
	vector< Mat >::const_iterator end = train_samples.end();
	for (int i = 0; itr != end; ++itr, ++i)
	{
		CV_Assert(itr->cols == 1 ||
			itr->rows == 1);
		if (itr->cols == 1)
		{
			transpose(*(itr), tmp);
			tmp.copyTo(trainData.row(i));
		}
		else if (itr->rows == 1)
		{
			itr->copyTo(trainData.row(i));
		}
	}
}

void sample_neg(const vector< Mat > & full_neg_lst, vector< Mat > & neg_lst, const Size & size)
{
	Rect box;
	box.width = size.width;
	box.height = size.height;

	const int size_x = box.width;
	const int size_y = box.height;

	srand((unsigned int)time(NULL));

	vector< Mat >::const_iterator img = full_neg_lst.begin();
	vector< Mat >::const_iterator end = full_neg_lst.end();
	for (; img != end; ++img)
	{
		box.x = rand() % (img->cols - size_x);
		box.y = rand() % (img->rows - size_y);
		Mat roi = (*img)(box);
		neg_lst.push_back(roi.clone());
#ifdef _DEBUG
		imshow("img", roi.clone());
		waitKey(10);
#endif
	}
}

Mat get_hogdescriptor_visu(const Mat& color_origImg, vector<float>& descriptorValues, const Size & size)
{
	const int DIMX = size.width;
	const int DIMY = size.height;
	float zoomFac = 3;
	Mat visu;
	resize(color_origImg, visu, Size((int)(color_origImg.cols*zoomFac), (int)(color_origImg.rows*zoomFac)));

	int cellSize = 8;
	int gradientBinSize = 9;
	float radRangeForOneBin = (float)(CV_PI / (float)gradientBinSize); 

															
	int cells_in_x_dir = DIMX / cellSize;
	int cells_in_y_dir = DIMY / cellSize;
	float*** gradientStrengths = new float**[cells_in_y_dir];
	int** cellUpdateCounter = new int*[cells_in_y_dir];
	for (int y = 0; y<cells_in_y_dir; y++)
	{
		gradientStrengths[y] = new float*[cells_in_x_dir];
		cellUpdateCounter[y] = new int[cells_in_x_dir];
		for (int x = 0; x<cells_in_x_dir; x++)
		{
			gradientStrengths[y][x] = new float[gradientBinSize];
			cellUpdateCounter[y][x] = 0;

			for (int bin = 0; bin<gradientBinSize; bin++)
				gradientStrengths[y][x][bin] = 0.0;
		}
	}

	int blocks_in_x_dir = cells_in_x_dir - 1;
	int blocks_in_y_dir = cells_in_y_dir - 1;

	int descriptorDataIdx = 0;
	int cellx = 0;
	int celly = 0;

	for (int blockx = 0; blockx<blocks_in_x_dir; blockx++)
	{
		for (int blocky = 0; blocky<blocks_in_y_dir; blocky++)
		{
			for (int cellNr = 0; cellNr<4; cellNr++)
			{
				cellx = blockx;
				celly = blocky;
				if (cellNr == 1) celly++;
				if (cellNr == 2) cellx++;
				if (cellNr == 3)
				{
					cellx++;
					celly++;
				}

				for (int bin = 0; bin<gradientBinSize; bin++)
				{
					float gradientStrength = descriptorValues[descriptorDataIdx];
					descriptorDataIdx++;

					gradientStrengths[celly][cellx][bin] += gradientStrength;

				} 


				cellUpdateCounter[celly][cellx]++;

			} 


		}
	}

	for (celly = 0; celly<cells_in_y_dir; celly++)
	{
		for (cellx = 0; cellx<cells_in_x_dir; cellx++)
		{

			float NrUpdatesForThisCell = (float)cellUpdateCounter[celly][cellx];

			for (int bin = 0; bin<gradientBinSize; bin++)
			{
				gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
			}
		}
	}

	for (celly = 0; celly<cells_in_y_dir; celly++)
	{
		for (cellx = 0; cellx<cells_in_x_dir; cellx++)
		{
			int drawX = cellx * cellSize;
			int drawY = celly * cellSize;

			int mx = drawX + cellSize / 2;
			int my = drawY + cellSize / 2;

			rectangle(visu, Point((int)(drawX*zoomFac), (int)(drawY*zoomFac)), Point((int)((drawX + cellSize)*zoomFac), (int)((drawY + cellSize)*zoomFac)), Scalar(100, 100, 100), 1);

			for (int bin = 0; bin<gradientBinSize; bin++)
			{
				float currentGradStrength = gradientStrengths[celly][cellx][bin];

				if (currentGradStrength == 0)
					continue;

				float currRad = bin * radRangeForOneBin + radRangeForOneBin / 2;

				float dirVecX = cos(currRad);
				float dirVecY = sin(currRad);
				float maxVecLen = (float)(cellSize / 2.f);
				float scale = 2.5;


				float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
				float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
				float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
				float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;

				line(visu, Point((int)(x1*zoomFac), (int)(y1*zoomFac)), Point((int)(x2*zoomFac), (int)(y2*zoomFac)), Scalar(0, 255, 0), 1);

			} 
		} 
	} 


	for (int y = 0; y<cells_in_y_dir; y++)
	{
		for (int x = 0; x<cells_in_x_dir; x++)
		{
			delete[] gradientStrengths[y][x];
		}
		delete[] gradientStrengths[y];
		delete[] cellUpdateCounter[y];
	}
	delete[] gradientStrengths;
	delete[] cellUpdateCounter;

	return visu;

}

void compute_hog(const vector< Mat > & img_lst, vector< Mat > & gradient_lst, const Size & size)
{
	HOGDescriptor hog;
	hog.winSize = size;
	Mat gray;
	vector< Point > location;
	vector< float > descriptors;

	vector< Mat >::const_iterator img = img_lst.begin();
	vector< Mat >::const_iterator end = img_lst.end();
	for (; img != end; ++img)
	{
		cvtColor(*img, gray, COLOR_BGR2GRAY);
		hog.compute(gray, descriptors, Size(8, 8), Size(0, 0), location);
		gradient_lst.push_back(Mat(descriptors).clone());
#ifdef _DEBUG
		imshow("gradient", get_hogdescriptor_visu(img->clone(), descriptors, size));
		waitKey(10);
#endif
	}
}

void train_svm(const vector< Mat > & gradient_lst, const vector< int > & labels)
{
	Ptr<SVM> svm = SVM::create();
	svm->setCoef0(0.0);
	svm->setDegree(3);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-3));
	svm->setGamma(0);
	svm->setKernel(SVM::LINEAR_CV);
	svm->setNu(0.5);
	svm->setP(0.1); 
	svm->setC(0.01); 
	svm->setType(SVM::EPS_SVR); 

	Mat train_data;
	convert_to_ml(gradient_lst, train_data);

	clog << "Start training...";
	svm->train(train_data, ROW_SAMPLE, Mat(labels));
	clog << "...[done]" << endl;

	svm->save(TRAINED_SVM);
}

void draw_locations(Mat & img, const vector< Rect > & locations, const Scalar & color)
{
	if (!locations.empty())
	{
		vector< Rect >::const_iterator loc = locations.begin();
		vector< Rect >::const_iterator end = locations.end();
		for (; loc != end; ++loc)
		{
			rectangle(img, *loc, color, 2);
		}
	}
}

void test_it(const Size & size)
{
	char key = 27;
	Mat img, draw;
	Ptr<SVM> svm;
	HOGDescriptor hog;
	hog.winSize = size;
	VideoCapture video;
	vector< Rect > locations;

	svm = StatModel::load<SVM>(TRAINED_SVM);
	vector< float > hog_detector;
	get_svm_detector(svm, hog_detector);
	hog.setSVMDetector(hog_detector);

	video.open(TRAFFIC_VIDEO_FILE);
	if (!video.isOpened())
	{
		cerr << "Unable to open the device" << endl;
	}

	int num_of_vehicles = 0;

	bool end_of_process = false;
#if IS_VIDEO
	while (!end_of_process)
	{
		video >> img;
		if (img.empty())
			break;
#else
	img = imread("test.PNG");
#endif
		draw = img.clone();

		for (int pi = 0; pi < img.rows; ++pi)
			for (int pj = 0; pj < img.cols; ++pj)
				if (pj > img.cols) {
					img.at<Vec3b>(pi, pj)[0] = 0;
					img.at<Vec3b>(pi, pj)[1] = 0;
					img.at<Vec3b>(pi, pj)[2] = 0;
				}

		locations.clear();
		hog.detectMultiScale(img, locations);
		draw_locations(draw, locations, Scalar(0, 255, 0));


		imshow(WINDOW_NAME, draw);
#if IS_VIDEO
		key = (char)waitKey(10);
		if (27 == key)
			end_of_process = true;
	}
#else
		waitKey();
#endif
}
