#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

double cap_dim(Mat& img);

double computeCoefficient(const std::vector<double>& X, const std::vector<double>& Y);
