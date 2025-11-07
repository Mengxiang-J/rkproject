#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <queue>
#include <mutex>
#include <thread>
#include <map>
#include <future>

#include "SafeQueue.h"
#include "thread_poll.h"
#include "yolov5s.h"
#include "post_process.h"

using namespace std;
using namespace cv;

ThreadPoll gthreadpool("/home/orangepi/Desktop/model/yolov5s.rknn", 12);

struct FrameData
{
    Mat frame;
    int index;
};

int image_index = 0;
mutex image_index_mutex;
SafeQueue<FrameData> Read_Queue(50);
SafeQueue<FrameData> Write_Queue(50);


mutex video_mutex;
std::atomic<bool> r_finished{false};  // 生产者完成标志
std::atomic<bool> p_finished{false};  // 处理完成标志

void enque(int thread_index, VideoCapture &cap)
{
    while (true)
    {
        FrameData temp_frame;
        /* 从视频读出每一帧图片 */
        {
            lock_guard<mutex> lock(video_mutex);
            if(!cap.read(temp_frame.frame))
            {
                r_finished = true;
                cout<<"EOF break"<<endl;
                break;
            }
        }

        {
            lock_guard<mutex> lock(image_index_mutex);
            /* 将读出的图片入队 */
            temp_frame.index = image_index;
            image_index++;
            std::cout << "Before enqueue, queue.size()=" << Read_Queue.size() << std::endl;

            Read_Queue.enqueue(temp_frame);

            std::cout << "After enqueue\n";
        }
    }    
}

// 定义一个用于处理帧的缓冲区，使用互斥锁保护
// std::map<int, cv::Mat> ProcessFrameBuffer;
// mutex bufferMutex;
// 新增结构体定义

void Thread_ProcessVideo(SafeQueue<FrameData>& Read_Queue, SafeQueue<FrameData>& Write_Queue)
{
    std::map<int, std::future<ProcessResult>> processing_futures;
    std::map<int, ProcessResult> completed_cache;
    int expected_index = 0;

    while (!(r_finished && Read_Queue.empty() && processing_futures.empty())) 
    {
        // 提交新任务
        if (!Read_Queue.empty()) 
        {
            FrameData frame_temp;
            Read_Queue.dequeue(frame_temp);
            
            auto future = gthreadpool.submit_task_async(
                frame_temp.index, 
                frame_temp.frame.clone()
            );
            processing_futures.emplace(frame_temp.index, std::move(future));
        }

        // 检查所有处理中的任务
        for (auto it = processing_futures.begin(); it != processing_futures.end(); ) 
        {
            if (it->second.wait_for(std::chrono::seconds(0)) == std::future_status::ready) 
            {
                // 获取处理结果并存入缓存
                ProcessResult result = it->second.get();
                completed_cache.emplace(it->first, std::move(result));
                it = processing_futures.erase(it);
            } 
            else 
            {
                ++it;
            }
        }

        // 按顺序处理缓存中的结果
        auto cache_it = completed_cache.find(expected_index);
        while (cache_it != completed_cache.end()) 
        {
            const ProcessResult& result = cache_it->second;
            
            if (result.success) 
            {
                Write_Queue.enqueue({result.processed_img, expected_index});
                if (expected_index % 100 == 0) 
                {
                    printf("Processed frame %d\n", expected_index);
                }
            } 
            else 
            {
                std::cerr << "Processing failed for frame " << expected_index
                        << ": " << result.error_msg << std::endl;
            }

            completed_cache.erase(cache_it);
            expected_index++;
            cache_it = completed_cache.find(expected_index);
        }

        // 退出条件检查
        if (r_finished && Read_Queue.empty() && processing_futures.empty()) 
        {
            // 在处理线程循环中添加
            std::cout << "[Processing] futures: " << processing_futures.size() 
            << " | cache: " << completed_cache.size() 
            << " | expected: " << expected_index << std::endl;
            p_finished = true;
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

void dequeue(VideoWriter &writer)
{
    
    while(true)
    {
        if(!Write_Queue.empty())
        {
            FrameData temp_frame;
            Write_Queue.dequeue(temp_frame);
            if(!temp_frame.frame.empty())
            {
                writer.write(temp_frame.frame);
            }
        }
        else if(p_finished && Write_Queue.empty())
        {
            break;
        } 
        else
        {
            // 如果写取队列为空，则等待一段时间
            this_thread::sleep_for(chrono::milliseconds(5));
        }
    }
}
// void dequeue(VideoWriter &writer)
// {
//     FrameData temp_frame;
//     while (true)
//     {
//         {
//             if(!Write_Queue.empty())
//             {
//                 FrameData temp_frame;
//                 Write_Queue.dequeue(temp_frame);
//                 if(!temp_frame.frame.empty())
//                 {
//                     auto start = chrono::high_resolution_clock::now();
//                     writer.write(temp_frame.frame);
//                     auto end = chrono::high_resolution_clock::now();
//                     cout << "Write time: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;               
//                 }
//             }
//             else if(r_finished && p_finished )
//             {
//                 printf("write finished!\n");
//                 break;
//             }
//         }
//     } 
// }
int main() {

    // //  使用绝对路径进行验证
    // const char *image_path = "/home/orangepi/Desktop/image.png";

    // // 加载图片
    // Mat img0 = imread(image_path, IMREAD_GRAYSCALE);
    // if (img0.empty()) {
    //     cout << "Failed to load image!" << endl;
    //     return -1;
    // }
    // cout << "Image loaded successfully!" << endl;

	// imwrite("img0.png",img0);

    /* 获取视频信息 */
	char video_path[] = "/home/orangepi/Desktop/video.mp4";
	VideoCapture cap(video_path);
	if(!cap.isOpened())
	{
		printf("video open failed!");
        return (int)-1;
	}
    int width     = cap.get(CAP_PROP_FRAME_WIDTH);
    int higth     = cap.get(CAP_PROP_FRAME_HEIGHT);
    double fps    = cap.get(CAP_PROP_FPS);
    int frame_num = cap.get(CAP_PROP_FRAME_COUNT);

    printf("heigth:%d,width:%d\n fps:%f,frame_sum:%d\n",higth,width,fps,frame_num);
//测试yolov5模型及其后处理的代码
    // const char *image_path = "/home/orangepi/Desktop/person.jpg";
    // Mat img_tmp  = imread(image_path, IMREAD_COLOR);
    // detect_result_group_t result_group;
    // Yolov5s test("/home/orangepi/Desktop/model/yolov5s.rknn", 0);
    // test.inference_image(img_tmp, result_group);
    // test.draw_result(img_tmp, result_group);
    // imwrite("asdf.jpg", img_tmp);
    // return 0;
    // while (1)
    // {
    //     /* code */
    // }
    

    /* 获取视频所有帧，存放入读取队列 */
    vector<thread> read_threads;
    read_threads.emplace_back(enque, 0, ref(cap)); 

    //读取队列从读取队列取出视频，处理，放入写入队列
    thread process_video(Thread_ProcessVideo,ref(Read_Queue),ref(Write_Queue) );
    //从去写入队列取出，保存到本地
    Size framesize(width,higth);
    const char* video_save_path = "test.avi";
    VideoWriter writer(video_save_path,VideoWriter::fourcc('I','4','2','0'),fps,framesize);
    thread write_thread(dequeue,ref(writer));
    /* 逐帧处理视频 */
    // thread process_video(Thread_ProcessVideo,ref(Read_Queue),ref(Write_Queue) );

    // /* 处理后的视频重新写入队列，并保存 */
    // Size framesize(width,higth);
    // const String video_save_path = "video_post_processing.avi";
    // VideoWriter writer(video_save_path,VideoWriter::fourcc('I','4','2','0'),fps,framesize);
    // thread write_thread(dequeue,ref(writer));

    for(thread&t : read_threads)
    {
        t.join();
    }
    process_video.join();
    // process_video.join();
    write_thread.join();
    // auto t1 = chrono::high_resolution_clock::now();
    writer.release();
    // auto t2 = chrono::high_resolution_clock::now();
    // cout << "writer.release() took " 
    //  << chrono::duration_cast<chrono::milliseconds>(t2 - t1).count() 
    //  << " ms" << endl;
    return 0;
}
