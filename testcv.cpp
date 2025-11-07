
#include <opencv2/opencv.hpp>
#include <atomic>
#include <thread>
#include <iostream>
#include <string>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <queue>


int main(void) 
{  
    std::string inPath  = "/home/orangepi/Desktop/video.mp4";

    // 打开输入视频
    cv::VideoCapture cap(inPath);
    if(!cap.isOpened()) 
    {
        std::cerr << "Fail to open input video: " << inPath << "\n";
        return -1;
    }

    // 获取属性
    int width  = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int fps = cap.get(cv::CAP_PROP_FRAME_COUNT);

    if(fps < 1.0) 
    {
        fps = 25.0; // 避免某些视频元数据不完整
    }
    std::cout<<"输入视频的总帧数是"<<fps<<std::endl;

    std::string output = "/home/orangepi/Desktop/build/output.avi";
    cv::VideoCapture cap1(output);
    if(!cap1.isOpened()) 
    {
        std::cerr << "Fail to open input video: " << inPath << "\n";
        return -1;
    }

    // 获取属性
    int asdf = cap1.get(cv::CAP_PROP_FRAME_COUNT);
    std::cout<<"输出视频的总帧数是"<<asdf<<std::endl;
   
    return 0;
}