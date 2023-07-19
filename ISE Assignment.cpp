// ISE Assignment.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include"highgui\highgui.hpp"
#include"opencv2\opencv.hpp"
#include"core\core.hpp"
#include <opencv2/imgproc/types_c.h>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <opencv2/opencv.hpp>


using namespace cv;
using namespace std;

// Covert images from RGB to grey
Mat RGBtoGrey(Mat RGB) {
    Mat GreyImg = Mat::zeros(RGB.size(), CV_8UC1);
    for (int i = 0; i < RGB.rows; i++) {
        for (int j = 0; j < RGB.cols * 3; j = j + 3) {
            GreyImg.at<uchar>(i, j / 3) = (RGB.at<uchar>(i, j)+ RGB.at<uchar>(i, j + 1)+ RGB.at<uchar>(i, j + 2)) / 3;
           
        }
    }
    return GreyImg;
}

// Convert images from grey to binary
Mat GreytoBinary(Mat Grey, int Th) {
    Mat BinaryImg = Mat::zeros(Grey.size(), CV_8UC1);
    for (int i = 0; i < Grey.rows; i++) {
        for (int j = 0; j < Grey.cols; j++) {
            if (Grey.at<uchar>(i, j) >= Th) {
                BinaryImg.at<uchar>(i, j) = 255;
            }
        }
    }
    return BinaryImg;
}

// Invert the images
Mat InversionMath(Mat Grey) {
    Mat InversionImg = Mat::zeros(Grey.size(), CV_8UC1);
    for (int i = 0; i < Grey.rows; i++) {
        for (int j = 0; j < Grey.cols; j++) {
            InversionImg.at<uchar>(i, j) = 255 - Grey.at<uchar>(i, j);
        }
    }
    return InversionImg;
}

// Produce proper illuminated images
Mat EqualizeHistogram(Mat Grey) {
    Mat EHImg = Mat::zeros(Grey.size(), CV_8UC1);
    // Counting
    int count[256] = { 0 };
    for (int i = 0; i < Grey.rows; i++) {
        for (int j = 0; j < Grey.cols; j++) {
            count[Grey.at<uchar>(i, j)]++;
        }
    }
    // Probability
    float prob[256] = { 0.0 };
    for (int i = 0; i < 256; i++) {
        prob[i] = (float)count[i] / (float)(Grey.rows * Grey.cols);
    }
    // Accumulate probability
    float accprob[256] = { 0.0 };
    accprob[0] = prob[0];
    for (int i = 1; i < 256; i++) {
        accprob[i] = prob[i] + accprob[i - 1];
    }
    // New pixel
    int newpixel[256] = { 0 };
    for (int i = 0; i < 256; i++) {
        newpixel[i] = 255 * accprob[i];
    }
    // Create the EHImg
    for (int i = 0; i < Grey.rows; i++) {
        for (int j = 0; j < Grey.cols; j++) {
            EHImg.at<uchar>(i, j) = newpixel[Grey.at<uchar>(i, j)];
        }
    }
    return EHImg;
}

// Blur the images
Mat AverageBlur(Mat Grey, int neighbor) {
    Mat AvgImg = Mat::zeros(Grey.size(), CV_8UC1);
    for (int i = neighbor; i < Grey.rows - neighbor; i++) {
        for (int j = neighbor; j < Grey.cols - neighbor; j++) {
            int totalVal = 0;
            for (int ii = -neighbor; ii <= neighbor; ii++) {
                for (int jj = -neighbor; jj <= neighbor; jj++) {
                    totalVal = totalVal + Grey.at<uchar>(i + ii, j + jj);
                }
            }
            AvgImg.at<uchar>(i, j) = totalVal / ((neighbor * 2 + 1) * (neighbor * 2 + 1));
        }
    }
    return AvgImg;
}

// Remove the border of the images
Mat ExcludeBorder(Mat BinaryPlate, int up, int down, int left, int right) {
    Mat RemovedBorderImg = Mat::zeros(BinaryPlate.size(), CV_8UC1);
    for (int i = up; i < BinaryPlate.rows - down; i++) {
        for (int j = left; j < BinaryPlate.cols - right; j++) {
            RemovedBorderImg.at<uchar>(i, j) = BinaryPlate.at<uchar>(i, j);
        }
    }
    return RemovedBorderImg;
};

// Find the edge in the images
Mat EdgeDetection(Mat Blur, int threshold) {
    Mat EDImg = Mat::zeros(Blur.size(), CV_8UC1);
    for (int i = 1; i < Blur.rows - 1; i++) {
        for (int j = 1; j < Blur.cols - 1; j++) {
            int avgL = (Blur.at<uchar>(i - 1, j - 1) + Blur.at<uchar>(i, j - 1) + Blur.at<uchar>(i + 1, j - 1)) / 3;
            int avgR = (Blur.at<uchar>(i - 1, j + 1) + Blur.at<uchar>(i, j + 1) + Blur.at<uchar>(i + 1, j + 1)) / 3;
            if (abs(avgL - avgR) > threshold) {
                EDImg.at<uchar>(i, j) = 255;
            }
        }
    }
    return EDImg;
}

// Split the connected segments
Mat Erosion(Mat Edge, int neighbor) {
    Mat ErosionImg = Mat::zeros(Edge.size(), CV_8UC1);
    for (int i = neighbor; i < Edge.rows - neighbor; i++) {
        for (int j = neighbor; j < Edge.cols - neighbor; j++) {
            ErosionImg.at<uchar>(i, j) = Edge.at<uchar>(i, j);
            for (int ii = -neighbor; ii <= neighbor; ii++) {
                for (int jj = -neighbor; jj <= neighbor; jj++) {
                    if (Edge.at<uchar>(i + ii, j + jj) == 0) {
                        ErosionImg.at<uchar>(i, j) = 0;
                        break;
                    }
                }
            }
        }
    }
    return ErosionImg;
}

// Bridging the gap
Mat Dilation(Mat Edge, int neighbor) {
    Mat DilationImg = Mat::zeros(Edge.size(), CV_8UC1);
    for (int i = neighbor; i < Edge.rows - neighbor; i++) {
        for (int j = neighbor; j < Edge.cols - neighbor; j++) {
            for (int ii = -neighbor; ii <= neighbor; ii++) {
                for (int jj = -neighbor; jj <= neighbor; jj++) {
                    if (Edge.at<uchar>(i + ii, j + jj) == 255) {
                        DilationImg.at<uchar>(i, j) = 255;
                        break;
                    }
                }
            }
        }
    }
    return DilationImg;
}

// Find the best threshold value to binarize the grey images
int OTSU (Mat Grey,int bias) {
    int count[256] = { 0 };
    float prob[256] = { 0.0 };
    float accProb[256] = { 0.0 };
    float meu[256] = { 0.0 };
    float sigma[256] = { 0.0 };

    for (int i = 0; i < Grey.rows; i++) {
        for (int j = 0; j < Grey.cols; j++) {
            count[Grey.at<uchar>(i, j)]++; //Count
        }
    }
    for (int c = 0; c < 256; c++) {
        prob[c] = (float)count[c] / (float)(Grey.cols * Grey.rows); // Probability
    }
    accProb[0] = prob[0];
    for (int p = 1; p < 256; p++) {
        accProb[p] = prob[p] + accProb[p - 1];
    } // =theta

    meu[0] = 0;
    for (int i = 1; i < 256; i++) {
        meu[i] = meu[i - 1] + prob[i] * i;
    }

    // Calculate meu acc i * acc prob i
    for (int i = 0; i < 256; i++) {
        sigma[i] = pow((meu[255] * accProb[i] - meu[i]), 2) / (accProb[i] * (1 - accProb[i]));
    }
    // Find i which maximize sigma
    int OtsuVal = 0;
    int MaxSigma = -1;
    for (int i = 0; i < 256; i++)
    {
        if (sigma[i] > MaxSigma)
        {
            MaxSigma = sigma[i];
            OtsuVal = i;

        }
    }
    return OtsuVal+bias;
}


int main()
{
    vector<String> images;
    string outText = "null";
    String dataset = "C:\\Users\\angel\\Desktop\\ISE assignment\\Dataset";
    glob(dataset, images);
    for (int i = 0; i < images.size(); i++) 
    {
        int epoch = 0;
        int epoch2 = 0;
        int BLURvalue = 1;
        int EDvalue = 48;
        int DILvalue = 4;
        int maxHeight = 50;
        int minHeight = 20;
        int maxWidth = 160;
        int minWidth = 70;

        // Read and display the car images
        Mat img;
        img = imread(images[i]);
        cout << images[i];
        imshow("Car image", img);
        waitKey();

        // Preprocess the car images
        Mat GreyImg = RGBtoGrey(img);
        Mat EHImg = EqualizeHistogram(GreyImg);

    startPD:
        Mat AvgBlurImg = AverageBlur(EHImg, BLURvalue);
        Mat EDImg = EdgeDetection(AvgBlurImg, EDvalue);
        Mat DilationImg = Dilation(EDImg, DILvalue);

        // Segment the image and color the segmented blocks
        int blobCount = 0;
        Mat blob;
        blob = DilationImg.clone();
        vector<vector<Point>>countours1;
        vector<Vec4i>hierarchy1;
        findContours(DilationImg, countours1, hierarchy1, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));
        Mat dst = Mat::zeros(DilationImg.size(), CV_8UC3);
        if (!countours1.empty()) {
            for (int i = 0; i < countours1.size(); i++) {
                Scalar colour((rand() & 255), (rand() & 255), (rand() & 255));
                drawContours(dst, countours1, i, colour, -1, 8, hierarchy1);
            }
        }

       // Filter the segment based on the features of bounding box that surround the segment
        Mat Plate;
        Rect BlobRect;
        Scalar black = CV_RGB(0, 0, 0);
        for (int j = 0; j < countours1.size(); j++) {
            BlobRect = boundingRect(countours1[j]);
            if (BlobRect.width / BlobRect.height < 1 || BlobRect.width < minWidth || BlobRect.width > maxWidth || BlobRect.height < minHeight || BlobRect.height> maxHeight || BlobRect.y < 50 || BlobRect.y>540) {
                drawContours(blob, countours1, j, black, -1, 8, hierarchy1);
            }
            else {
                Plate = GreyImg(BlobRect);
                blobCount++;
            }
        }

        // Change the parameters if the car plates cannot be detected
        epoch++;
        if (blobCount==0||blobCount>1) { //low resolution pic
            if (epoch == 1) {
                BLURvalue = 0;
                EDvalue = 44;
                DILvalue = 8;
                goto startPD;
            }
            else if (epoch == 2) { //double row car plate
                BLURvalue = 1;
                EDvalue = 60;
                DILvalue = 7;
                minWidth = 55;
                maxWidth = 160;
                minHeight = 35;
                maxHeight = 60;
                goto startPD;
            }
            else if (epoch==3) { //car plates with very obvious logo
                BLURvalue = 1;
                EDvalue = 50;
                DILvalue = 5;
                minWidth = 95;
                maxWidth = 160;
                minHeight = 33;
                maxHeight = 45;
                goto startPD;
            }

        }
       
            // Display cropped plate images
            imshow("Plate", Plate);
            waitKey();
            destroyWindow("Plate");

            // Rescaling the cropped plate images
            Size size(Plate.cols * 2, Plate.rows * 2);
            Mat EnlargedPlate;
            resize(Plate, EnlargedPlate, size);

            int bias = 47;
            int borderUp = 5;
            int borderDown = 8;
            int borderLeft = 12;
            int borderRight = 10;
            int erosionVal = 1;
            int dilationVal = 2;

        startPR: 
            // Preprocess the cropped plate images
            int otsuVal = OTSU(EnlargedPlate,bias);
            Mat binaryPlate = GreytoBinary(EnlargedPlate, otsuVal);
            Mat RemovedImage = ExcludeBorder(binaryPlate,borderUp,borderDown,borderLeft,borderRight);
            Mat ErodeImg = Erosion(RemovedImage,erosionVal);
            Mat DilatedImg = Dilation(ErodeImg,dilationVal);
            Mat InversedImg = InversionMath(DilatedImg);
            imshow("Inverse", InversedImg);
            
            cv::cvtColor(InversedImg, InversedImg, COLOR_BGR2BGRA);
            tesseract::TessBaseAPI* api = new tesseract::TessBaseAPI();

            // Initialize tesseract-ocr with English, without specifying tessdata path
            if (api->Init(NULL, "eng", tesseract::OEM_LSTM_ONLY)) {
                fprintf(stderr, "Could not initialize tesseract.\n");
                exit(1);
            }
            char* outText;
            api->SetImage(InversedImg.data, InversedImg.cols, InversedImg.rows, 4, 4 * InversedImg.cols);
            api->SetVariable("tessedit_char_whitelist", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ");
            api->SetVariable("user_defined_dpi", "300");

            // Get OCR result
            outText = api->GetUTF8Text();
            String output = outText;
            String finaloutput = "";
            for (int i = 0; i < output.length(); i++) {
                if (!isspace(output[i])) {
                    finaloutput = finaloutput + output[i];
                }
            }
            int outputSize= finaloutput.length();

            // Destroy used object and release memory
            api->End();
            delete api;
            delete[] outText;
            waitKey();

            // Change the parameter if the car plates cannot be recognized
            epoch2++;
            if (outputSize==0||outputSize < 6 || outputSize>7) {
                if (epoch2 == 1) {
                    bias = 20;
                    borderUp = 0;
                    borderDown = 0;
                    borderLeft = 13;
                    borderRight = 13;
                    erosionVal = 0;
                    dilationVal = 0;
                    goto startPR;
                }

                if (epoch2 == 2) {
                    bias = 80;
                    borderUp = 8;
                    borderDown = 8;
                    borderLeft = 30;
                    borderRight = 20;
                    erosionVal = 1;
                    dilationVal = 1;
                    goto startPR;
                }

            }

            // Display the car plates number
            cout << "\nCar plate:" << finaloutput<<"\n\n";
            waitKey();
    }
    return 0;
}

     

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
