//https://github.com/spmallick/learnopencv/tree/master/FacialLandmarkDetection
//https://www.learnopencv.com/facemark-facial-landmark-detection-using-opencv/
// initialize build folder: cmake -G "Visual Studio 15 2017 Win64" ..
// build: cmake --build . --config Release

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include "drawLandmarks.hpp"

using namespace std;
using namespace cv;
using namespace cv::face;

bool imgResizeKeepRatio(Mat &im, int width=0, int height=0) {
    Size s = im.size();
    float ratio;

    if (width == 0 && height != 0)
    {
        ratio = (float)height/s.height;
    }
    else if (height == 0 && width != 0)
    {
        ratio = (float)width/s.width;
    }
    else
    {
        cout << "imgResizeKeepRatio(Mat, width, height) needs either width or height to be 0" << endl;
        return false;
    }

    resize(im, im, Size(), ratio, ratio);
    return true;
}

int main(int argc, char** argv)
{
    if (argc != 2) 
    {
        cout << "Usage: make_trollface IMAGE" << endl;
        return -1;
    }

    // Load face detector and facial landmark detector
    CascadeClassifier faceDetector("haarcascade_frontalface_alt.xml");

    Ptr<Facemark> facemark = FacemarkLBF::create();

    facemark->loadModel("lbfmodel.yaml");

    // Detect faces and landmarks
    Mat frame, gray;
    frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    
    vector<Rect> faces;
    cvtColor(frame, gray, COLOR_BGR2GRAY);

    faceDetector.detectMultiScale(gray, faces);
    cout << "Detected " << faces.size() << " faces." << endl;

    // Detect landmarks
    vector< vector<Point2f> > landmarks;
    bool success = facemark->fit(frame, faces, landmarks);

    if (success) 
    {
        for (int i = 0; i < landmarks.size(); i++) {
            drawLandmarks(frame, landmarks[i]);
        }
    }

    imgResizeKeepRatio(frame, 1000);
    imshow("Facial Landmarks Prediction", frame);
    waitKey(0);

    return 0;
}