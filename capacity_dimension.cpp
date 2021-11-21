#include "capacity_dimension.h"


constexpr uint8_t BLACK = 0;
constexpr uint8_t WHITE = 255;
constexpr uint8_t BLACK_THRESHOLD = 127;

constexpr uint8_t CELL_NUMBER = 5;


void preprocessImage(Mat& img, uint32_t& newSize);
void convertToBW(Mat& img);


double cap_dim(Mat& img)
{
	uint32_t newSize = 0;
	preprocessImage(img, newSize);

	std::vector<double> X; // log2(1/e)
	std::vector<double> Y; // log2(N(e))

	// цикл по размеру ячейки (2,4,...,2^(CELL_NUMBER-1))
	for (size_t k = 1; k < CELL_NUMBER; ++k)
	{
		size_t cellSize = static_cast<size_t>(std::pow(2, k));	// размер ячейки
		size_t nCell = newSize / cellSize;						// количество ячеек по одной оси
		size_t nCoverage = 0;									// количество ячеек, имеющих чёрную точку

		// цикл по ячейкам
		for (size_t j = 0; j < nCell; ++j) //rows
		{
			for (size_t i = 0; i < nCell; ++i) //cols
			{
				//поиск чёрной точки внутри ячейки
				bool foundBlack = false;

				for (size_t jCell = 0; jCell < cellSize; ++jCell)
				{
					for (size_t iCell = 0; iCell < cellSize; ++iCell)
					{
						Vec3b& color = img.at<Vec3b>(cellSize * j + iCell, cellSize * i + jCell);

						if (color[0] == BLACK)
						{
							foundBlack = true;
							nCoverage++;
							break;
						}
					}

					if (foundBlack)
					{
						break;
					}
				}
			}
		}

		X.push_back(-static_cast<double>(k));
		Y.push_back(std::log2(nCoverage));
	}


	cout << "X = ln(1/e)" << " : " << "Y = ln(N(e))" << endl;
	for (size_t i = 0; i < X.size(); ++i)
	{
		cout << X[i] << " : " << Y[i] << endl;
	}

	return computeCoefficient(X, Y);
}

void preprocessImage(Mat& img, uint32_t& newSize)
{
	newSize = 1024U;
	Size cvSize(static_cast<int>(newSize), static_cast<int>(newSize));

	resize(img, img, cvSize);
	convertToBW(img);

	imshow("Preporcessed image", img);
	waitKey(0);
}

void convertToBW(Mat& img)
{
	for (size_t j = 0; j < img.rows; ++j)
	{
		for (size_t i = 0; i < img.cols; ++i)
		{
			Vec3b& color = img.at<Vec3b>((int)j, (int)i);
			if (color[0] <= BLACK_THRESHOLD)
			{
				color[0] = color[1] = color[2] = BLACK;
			}
			else
			{
				color[0] = color[1] = color[2] = WHITE;
			}
		}
	}
}

double computeCoefficient(const std::vector<double>& X, const std::vector<double>& Y)
{
	double A = 0.0;
	double B = 0.0;
	double C = 0.0;
	double D = 0.0;
	double a = 0.0;
	double b = 0.0;

	for (size_t i = 0; i < X.size(); ++i)
	{
		A += X[i] * Y[i];
		B += X[i] * X[i];
		C += X[i];
		D += Y[i];
	}

	b = (D * B - A * C) / (X.size() * B - C * C);
	a = (A - b * C) / B;

	return a;
}
