//
// Created by pavel on 25.08.21.
//

#include <opencv2/opencv.hpp>
#include <stdio.h>

int main() {
    cv::Mat test = cv::Mat::zeros(10, 10, CV_16UC1);

    for (int y = 0; y < test.rows; y++) {
        for (int x = 0; x < test.cols; x++) {
            printf("%d", test.at<uchar>(y, x));

        }
    }


    return 0;
}