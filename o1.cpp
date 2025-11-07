下面给你演示一种“单一聚合线程 + 线程池 N 个 worker 并行”的改进方案，以实现真正的并行推理并且严格按照原帧顺序输出到视频文件。核心思路是：

1) 保留原来的读线程 (readThread) 与写线程 (writeThread)，它们的功能不变。
2) 取消之前的“处理线程”或只开一个“处理线程”，改用一个“聚合线程”(aggregatorThread) 来对接线程池：
   • 从 g_readQueue 中批量获取帧，调用 thread pool 的 submit_task_async(...) 提交推理任务。
   • 不立即 future.get() 阻塞，而是把 (index → future) 存在一个映射中。
   • 轮询映射里“下一个待写帧”是否推理完成；若完成，则将结果按顺序放入 g_writeQueue；否则继续提交/等待其他帧。
3) 这样就会同时提交多个帧到线程池，利用线程池的并行 (例如 12 worker)，当有多帧排队时，线程池可同时处理多个任务，从而真正发挥并行优势。
4) 聚合线程确保只有当下标为 nextWriteIndex 的帧完成后才写入，从而保持帧顺序正确。

下面给出完整示例，着重展示关键修改部分(特别是 aggregatorThreadFunc 与 main)。你可以在自己已有的工程中替换相应部分。请确保你已实现 ThreadPoll, SafeQueue, Yolov5s 等类，并包含相关头文件。

----------------------------------------------------------------------------------------
// main.cpp
#include <opencv2/opencv.hpp>
#include <atomic>
#include <thread>
#include <iostream>
#include <string>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <chrono>
#include <future>
#include <map>

#include "SafeQueue.h"
#include "yolov5s.h"
#include "thread_poll.h"

//-----------------------------------
// 1) 定义一个存放帧和下标的结构
//-----------------------------------
struct FrameData {
    cv::Mat frame;
    int index;
};

// 全局队列 & 全局标志
SafeQueue<FrameData> g_readQueue(100);
SafeQueue<FrameData> g_writeQueue(100);
std::atomic<bool> g_readFinish(false);
std::atomic<bool> g_processFinish(false);

//-----------------------------------
// 2) 读线程：不断从视频文件读取放入 g_readQueue
//-----------------------------------
void readThreadFunc(cv::VideoCapture &cap) {
    int idx = 0;
    while(true)
    {
        cv::Mat frame;
        if(!cap.read(frame))
        {
            // 读不到帧了（到视频末尾或出错）
            std::cerr << "[ReadThread] read failed or EOF.\n";
            break;
        }
        FrameData data{ frame.clone(), idx++ };
        g_readQueue.enqueue(data);
    }
    // 通知后续不再有新帧
    g_readFinish = true;
    std::cerr << "[ReadThread] finished.\n";
}

//-----------------------------------
// 3) 聚合线程：既提交多帧到线程池并行处理，也按顺序收集结果
//-----------------------------------
void aggregatorThreadFunc(ThreadPoll &npu_pool)
{
    // 用于按正确顺序写入的下标
    int nextWriteIndex = 0;

    // 存储“帧下标 -> future”的映射，实现无阻塞并行提交与按序收集
    std::map<int, std::future<ProcessResult>> tasks_inflight;

    while(true)
    {
        // 步骤A：批量尝试从 g_readQueue 获取新帧并提交到线程池
        //       (如果想控制并行度，可在这里做判断 tasks_inflight.size() 是否过大。)
        FrameData inputFD;
        while(g_readQueue.dequeue(inputFD))  // 每次循环能弹尽量多的帧
        {
            // 提交异步推理任务
            auto fut = npu_pool.submit_task_async(inputFD.index, inputFD.frame);
            // 将 (index -> future) 存到映射
            tasks_inflight[inputFD.index] = std::move(fut);
        }

        // 步骤B：检查是否有“下一个待写帧(nextWriteIndex)”已经推理完成
        //        如果完成，就把其结果按顺序放到 g_writeQueue
        auto it = tasks_inflight.find(nextWriteIndex);
        while(it != tasks_inflight.end())
        {
            // 不阻塞，先检查这条 future 是否ready
            auto status = it->second.wait_for(std::chrono::milliseconds(0));
            if(status == std::future_status::ready)
            {
                // 获取推理结果
                ProcessResult result = it->second.get();

                // 将推理后图像放到 g_writeQueue
                FrameData outputFD;
                outputFD.index = nextWriteIndex;
                outputFD.frame = result.processed_img.clone();
                g_writeQueue.enqueue(outputFD);

                // 移除映射并递增下一个待写index
                tasks_inflight.erase(it);
                nextWriteIndex++;

                // 继续尝试下一个
                it = tasks_inflight.find(nextWriteIndex);
            }
            else
            {
                // 下一个还没完成，就先退出等待，后面再检测
                break;
            }
        }

        // 步骤C：判断退出条件
        //   若读完了 && 读队列空了 && 当前映射也空了，就说明都处理完了
        if(g_readFinish && g_readQueue.empty() && tasks_inflight.empty())
        {
            break;
        }

        // 为避免CPU空转过高，可稍微睡一下
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // 设置处理完成标志
    g_processFinish = true;
    std::cerr << "[AggregatorThread] finished.\n";
}

//-----------------------------------
// 4) 写线程：从 g_writeQueue 中取出图像写到文件
//-----------------------------------
void writeThreadFunc(cv::VideoWriter &writer)
{
    while(true)
    {
        if(g_processFinish && g_writeQueue.empty())
        {
            // 没有更多帧了
            break;
        }

        FrameData outputFD;
        if(!g_writeQueue.dequeue(outputFD)) {
            // 即使没取到，也要尝试继续，直到所有任务结束
            continue;
        }

        if(!outputFD.frame.empty())
        {
            writer.write(outputFD.frame);
        }
    }
    std::cerr << "[WriteThread] finished.\n";
}

//-----------------------------------
// 5) main 函数，把上述线程和线程池串起来
//-----------------------------------
int main()
{
    auto start = std::chrono::high_resolution_clock::now();

    std::string inPath  = "/home/orangepi/Desktop/video.mp4"; // 你的输入视频
    std::string outPath = "output.avi";                       // 输出文件

    // 打开输入视频
    cv::VideoCapture cap(inPath);
    if(!cap.isOpened())
    {
        std::cerr << "Fail to open input video: " << inPath << "\n";
        return -1;
    }

    // 获取视频属性
    int width  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    if(fps < 1.0)  // 避免某些视频元数据不完整
        fps = 25.0;

    int fourcc = cv::VideoWriter::fourcc('M','J','P','G');

    // 打开输出视频
    cv::VideoWriter writer(outPath, fourcc, fps, cv::Size(width, height));
    if(!writer.isOpened())
    {
        std::cerr << "Fail to create output video: " << outPath << "\n";
        return -1;
    }

    // 创建 thread pool，让它开足核数（例如 12 worker）
    ThreadPoll npu_pool("/home/orangepi/Desktop/model/yolov5s.rknn", 12);

    // 启动：1) 读线程, 2) 聚合线程, 3) 写线程
    std::thread tRead(readThreadFunc, std::ref(cap));
    std::thread tAggregator(aggregatorThreadFunc, std::ref(npu_pool));
    std::thread tWrite(writeThreadFunc, std::ref(writer));

    // 等3个线程退出
    tRead.join();
    tAggregator.join();
    tWrite.join();

    // 给队列发 stop 信号（好习惯，但此时往往都空了）
    g_readQueue.stop();
    g_writeQueue.stop();

    writer.release();

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "处理总用时：" << elapsed_ms.count() << " ms\n";
    std::cerr << "[Main] All done.\n";
    return 0;
}
----------------------------------------------------------------------------------------

核心亮点说明：
• 有且仅有一个“聚合线程”aggregatorThreadFunc()，它不断地从 g_readQueue 获取原始帧并一口气提交给线程池，拿到 future，但并不立刻 .get()。而是先把 (index, future) 存在一个映射 tasks_inflight。
• 然后按顺序 (nextWriteIndex) 检查是否有推理完成的帧可写，如果完成，就 .get() 并将结果排到 g_writeQueue。这样保证写文件时的顺序正确。
• 一次可以提交多个帧给线程池，让 npu_pool 里多个 worker 线程并发推理。只要 tasks_inflight 里有足够的帧，线程池就能最大化利用并行能力。
• 写线程和读线程都与聚合线程并行工作，读线程不断往 g_readQueue 塞帧；聚合线程抓取帧->提交任务->按序收集结果->排队待写；写线程就从 g_writeQueue 中按队列顺序写入文件。

这样，你就可以得到“真正的并发推理 + 保证输出视频顺序正确 + 三个线程可并行工作” 的流式处理流程。如果还有更多定制需求，如对并行度进行限流、使用更智能的数据结构、减少空转等待等，都可以在同样的思路上再做扩展。希望对你有所帮助，祝调试顺利!