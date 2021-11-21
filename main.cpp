#include <opencv2/opencv.hpp>
#include <iostream>
#include <numeric>

#include "capacity_dimension.h"

using namespace cv;
using namespace std;

void task1(Mat image);
void task2(Mat image);
void task3(Mat image);
void task4(Mat image);
void task5(Mat image);
void task6(Mat image);
void task7(Mat image);

int main(int argc, char** argv)
{
	Mat image;
	string imgPath;

	if (argc < 2)
	{
		cout << "Input path to image" << endl;
		cin >> imgPath;
	}
	else
	{
		imgPath = argv[1];
	}

	image = imread(imgPath);

	if (image.empty())
	{
		cout << "Image Not Found!" << endl;
		cin.get();
		return -1;
	}

	imshow("Original image", image);
	waitKey(0);

	//task1(image);
	task2(image);
	//task3(image);
	//task4(image);
	//task5(image);
	//task6(image);
	//task7(image);
	
	return 0;
}


// ~~~~ task 1 ~~~~

void task1(Mat image)
{
	cout << "Capacity dimension: " << cap_dim(image) << endl;
}


// ~~~~ task 2 ~~~~

void reduceColor(const Mat& src, Mat& dst)
{
	int K = 8;
	int n = src.rows * src.cols;
	Mat data = src.reshape(1, n);
	data.convertTo(data, CV_32F);

	vector<int> labels;
	Mat1f colors;
	kmeans(data, K, labels, cv::TermCriteria(), 1, cv::KMEANS_PP_CENTERS, colors);

	for (int i = 0; i < n; ++i)
	{
		data.at<float>(i, 0) = colors(labels[i], 0);
		data.at<float>(i, 1) = colors(labels[i], 1);
		data.at<float>(i, 2) = colors(labels[i], 2);
	}

	Mat reduced = data.reshape(3, src.rows);
	reduced.convertTo(dst, CV_8U);
}

struct lessVec3b
{
	bool operator()(const Vec3b& lhs, const Vec3b& rhs) const
	{
		return (lhs[0] != rhs[0]) ? (lhs[0] < rhs[0]) : ((lhs[1] != rhs[1]) ? (lhs[1] < rhs[1]) : (lhs[2] < rhs[2]));
	}
};

map<Vec3b, int, lessVec3b> getPalette(const Mat3b& src)
{
    map<Vec3b, int, lessVec3b> palette;
    for (int r = 0; r < src.rows; ++r)
    {
        for (int c = 0; c < src.cols; ++c)
        {
            Vec3b color = src(r, c);
            if (palette.count(color) == 0)
            {
                palette[color] = 1;
            }
            else
            {
                palette[color] = palette[color] + 1;
            }
        }
    }

    return palette;
}

void task2(Mat image)
{
	// 1) Уменьшение количества цветов в изображении
	Mat reduced;
	reduceColor(image, reduced);

	imshow("With reduced palette", reduced);
	waitKey(0);

	// 2) Получение палитры получившегося изображения
	map<Vec3b, int, lessVec3b> palette = getPalette(reduced);
	int area = reduced.rows * reduced.cols;

	Vec3b* colors = new Vec3b [palette.size()];
	size_t idx = 0;

	for (auto color : palette)
	{
		cout << idx + 1 << " Color: " << color.first
			<< " \t - Area: " << 100.f * float(color.second) / float(area) << "%" << endl;
		colors[idx] = color.first;
		++idx;
	}

	// 3) Выбор цвета
	cout << "Choose color (put number): ";
	cin >> idx;
	Vec3b& chosenColor = colors[idx - 1];

	// 4) Оставить только выбранный цвет
	for (size_t j = 0; j < reduced.rows; ++j)
	{
		for (size_t i = 0; i < reduced.cols; ++i)
		{
			Vec3b& color = reduced.at<Vec3b>((int)j, (int)i);
			if (color != chosenColor)
			{
				color[0] = color[1] = color[2] = 255;
			}
		}
	}

	imshow("Only one chosen color", reduced);
	waitKey(0);
}


// ~~~~ task 3 ~~~~

void task3(Mat image)
{
	cvtColor(image, image, COLOR_BGR2GRAY);

	imshow("In gray scale", image);
	waitKey(0);
}


// ~~~~ task 4 ~~~~

Mat getU(const Mat& predU)
{
	Mat newU = Mat(predU.rows, predU.cols, CV_8UC1, Scalar(0));

	for (size_t j = 0; j < predU.rows; ++j)
	{
		for (size_t i = 0; i < predU.cols; ++i)
		{
			std::vector<uchar> neighbours;

			if (j > 0)
			{
				neighbours.push_back(predU.at<uchar>(j - 1, i));
			}
			if (j < predU.rows - 1)
			{
				neighbours.push_back(predU.at<uchar>(j + 1, i));
			}
			if (i > 0)
			{
				neighbours.push_back(predU.at<uchar>(j, i - 1));
			}
			if (i < predU.cols - 1)
			{
				neighbours.push_back(predU.at<uchar>(j, i + 1));
			}

			newU.at<uchar>(j, i) = std::max(
				static_cast<uchar>(predU.at<uchar>(j, i) + 1),
				*std::max_element(neighbours.begin(), neighbours.end()));
		}
	}

	return newU;
}

Mat getB(const Mat& predB)
{
	Mat newB = Mat(predB.rows, predB.cols, CV_8UC1, Scalar(0));

	for (size_t j = 0; j < predB.rows; ++j)
	{
		for (size_t i = 0; i < predB.cols; ++i)
		{
			std::vector<uchar> neighbours;

			if (j > 0)
			{
				neighbours.push_back(predB.at<uchar>(j - 1, i));
			}
			if (j < predB.rows - 1)
			{
				neighbours.push_back(predB.at<uchar>(j + 1, i));
			}
			if (i > 0)
			{
				neighbours.push_back(predB.at<uchar>(j, i - 1));
			}
			if (i < predB.cols - 1)
			{
				neighbours.push_back(predB.at<uchar>(j, i + 1));
			}

			newB.at<uchar>(j, i) = std::min(
				static_cast<uchar>(predB.at<uchar>(j, i) - 1),
				*std::min_element(neighbours.begin(), neighbours.end()));
		}
	}

	return newB;
}

int getVol(const Mat& u, const Mat& b)
{
	int res = 0;

	for (size_t j = 0; j < u.rows; ++j)
	{
		for (size_t i = 0; i < u.cols; ++i)
		{
			res += (int)u.at<uchar>(j, i) - (int)b.at<uchar>(j, i);
		}
	}

	return res;
}

double getA(int volCurr, int volPred)
{
	return (static_cast<double>(volCurr) - static_cast<double>(volPred)) / 2.0;
}

void task4(Mat image)
{
	vector<double> X; // -log2(delta)
	vector<double> Y; // log2(A_n)
	Mat u;
	Mat b;
	int volPred = 0;
	int volCurr = 0;
	double a_n = 0.0;

	cvtColor(image, image, COLOR_BGR2GRAY);
	image.copyTo(u);
	image.copyTo(b);

	const int numDelta = 2;

	for (size_t d = 1; d <= numDelta; ++d)
	{
		u = getU(u);
		b = getB(b);
		volCurr = getVol(u, b);
		a_n = getA(volCurr, volPred);
		
		X.push_back(-log2(d));
		Y.push_back(log2(a_n));
		volPred = volCurr;
	}

	cout << "D = " << 2.0 - computeCoefficient(X, Y) << endl;

	cout << "\nX = ln(delta)" << " : " << "Y = ln(A_n)" << endl;
	for (size_t i = 0; i < X.size(); ++i)
	{
		cout << X[i] << " : " << Y[i] << endl;
	}
}


// ~~~~ task 5 ~~~~

void task5(Mat image)
{
	const int cellSize = 10; // При таких ячейках фон имеет площадь ~100
	double aThreshold = 100;

	Mat img = Mat(image.rows + (cellSize - image.rows % cellSize),
				  image.cols + (cellSize - image.cols % cellSize),
				  CV_8UC3, Scalar(255, 255, 255));
	image.copyTo(img(cv::Rect(0, 0, image.cols, image.rows)));

	vector<double> A_n;

	for (size_t j = 0; j < img.rows / cellSize; ++j)
	{
		for (size_t i = 0; i < img.cols / cellSize; ++i)
		{
			Mat roi(img(Rect(i * cellSize, j * cellSize, cellSize, cellSize)));

			Mat u;
			Mat b;
			int volPred = 0;
			int volCurr = 0;
			double a_n = 0.0;

			cvtColor(roi, roi, COLOR_BGR2GRAY);
			roi.copyTo(u);
			roi.copyTo(b);

			const int numDelta = 2;

			for (size_t d = 1; d <= numDelta; ++d)
			{
				u = getU(u);
				b = getB(b);
				volCurr = getVol(u, b);
				a_n = getA(volCurr, volPred);

				volPred = volCurr;
			}

			A_n.push_back(a_n);
		}
	}

	for (size_t j = 0; j < img.rows / cellSize; ++j)
	{
		for (size_t i = 0; i < img.cols / cellSize; ++i)
		{
			if (A_n[i + j * img.cols / cellSize] >= aThreshold)
			{
				rectangle(img, Rect(i * cellSize, j * cellSize, cellSize, cellSize), Scalar(0, 0, 0), FILLED);
			}
			else
			{
				rectangle(img, Rect(i * cellSize, j * cellSize, cellSize, cellSize), Scalar(255, 255, 255), FILLED);
			}
		}
	}

	imshow("Segmented image", img);
	waitKey();
}


// ~~~~ task 6 ~~~~

void task6(Mat image)
{
	vector<double> X; // -log2(delta)
	vector<double> Y; // log2(A_n)
	Mat u;
	Mat b;
	int volPred = 0;
	int volCurr = 0;
	double a_n = 0.0;

	cvtColor(image, image, COLOR_BGR2GRAY);
	image.copyTo(u);
	image.copyTo(b);

	const int numDelta = 10;

	for (size_t d = 1; d <= numDelta; ++d)
	{
		u = getU(u);
		b = getB(b);
		volCurr = getVol(u, b);
		a_n = getA(volCurr, volPred);

		X.push_back(-log(d));
		Y.push_back(log(a_n));
		volPred = volCurr;
	}

	cout << "\nY/X = ln(A)/ln(delta)" << endl;
	for (size_t i = 1; i < X.size(); ++i)
	{
		cout << -Y[i]/X[i] << endl;
	}
}


// ~~~~ task 7 ~~~~

void task7(Mat image)
{
	vector<size_t> cellSizes = { 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 };
	vector<double> D;

	for (size_t cs : cellSizes)
	{
		Mat img = Mat(image.rows + (cs - image.rows % cs),
					  image.cols + (cs - image.cols % cs),
					  CV_8UC3, Scalar(0, 0, 0));
		image.copyTo(img(cv::Rect(0, 0, image.cols, image.rows)));

		double A1 = 0.0;
		double A2 = 0.0;

		for (size_t j = 0; j < img.rows / cs; ++j)
		{
			for (size_t i = 0; i < img.cols / cs; ++i)
			{
				Mat roi(img(Rect(i * cs, j * cs, cs, cs)));

				Mat u;
				Mat b;
				int volPred = 0;
				int volCurr = 0;
				double a_n = 0.0;

				cvtColor(roi, roi, COLOR_BGR2GRAY);
				roi.copyTo(u);
				roi.copyTo(b);

				u = getU(u);
				b = getB(b);
				volCurr = getVol(u, b);

				A1 += getA(volCurr, volPred);
				volPred = volCurr;

				u = getU(u);
				b = getB(b);
				volCurr = getVol(u, b);

				A2 += getA(volCurr, volPred);
			}
		}

		D.push_back( 2.0 - computeCoefficient({ -log(1), -log(2) }, { log(A1), log(A2) }) );
	}

	for (size_t i = 0; i < cellSizes.size(); ++i)
	{
		cout << cellSizes[i] << " : " << D[i] << endl;
	}
}
