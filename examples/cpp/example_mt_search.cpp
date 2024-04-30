#include "../../hnswlib/hnswlib.h"
#include <thread>

#include <chrono>


// Multithreaded executor
// The helper function copied from python_bindings/bindings.cpp (and that itself is copied from nmslib)
// An alternative is using #pragme omp parallel for or any other C++ threading
template<class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }

    if (numThreads == 1) {
        for (size_t id = start; id < end; id++) {
            fn(id, 0);
        }
    } else {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);

        // keep track of exceptions in threads
        // https://stackoverflow.com/a/32428427/1713196
        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;

        for (size_t threadId = 0; threadId < numThreads; ++threadId) {
            threads.push_back(std::thread([&, threadId] {
                while (true) {
                    size_t id = current.fetch_add(1);

                    if (id >= end) {
                        break;
                    }

                    try {
                        fn(id, threadId);
                    } catch (...) {
                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                        lastException = std::current_exception();
                        /*
                         * This will work even when current is the largest value that
                         * size_t can fit, because fetch_add returns the previous value
                         * before the increment (what will result in overflow
                         * and produce 0 instead of current + 1).
                         */
                        current = end;
                        break;
                    }
                }
            }));
        }
        for (auto &thread : threads) {
            thread.join();
        }
        if (lastException) {
            std::rethrow_exception(lastException);
        }
    }
}


int main() {
    int dim = 128;               // Dimension of the elements
    int max_elements = 128000;   // Maximum number of elements, should be known beforehand
    int M = 90;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff
    int num_threads = 20;       // Number of threads for operations with index

    // Initing index
    hnswlib::InnerProductSpace space(dim);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, 1000000, M, ef_construction);
    alg_hnsw->setEf(100);

    // Generate random data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    float* data = new float[dim * max_elements];
    for (int i = 0; i < dim * max_elements; i++) {
        data[i] = distrib_real(rng);
    }

    auto start2 = std::chrono::high_resolution_clock::now();
    // Add data to index
    alg_hnsw->addPoint(data, 0);
    auto end2 = std::chrono::high_resolution_clock::now();
    // 计算时长
    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2);
    std::cout << "第一次单插入执行时长: " << duration2.count() << " 毫秒" << std::endl;


    auto start = std::chrono::high_resolution_clock::now();
    // Add data to index
    ParallelFor(1, max_elements, num_threads, [&](size_t row, size_t threadId) {
        alg_hnsw->addPoint((void*)(data + dim * row), row);
    });
    auto end = std::chrono::high_resolution_clock::now();
    // 计算时长
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "代码执行时长: " << duration.count() << " 毫秒" << std::endl;



    float* data3 = new float[dim];
    for (int i = 0; i < dim * 1; i++) {
        data3[i] = distrib_real(rng);
    }

    auto start3 = std::chrono::high_resolution_clock::now();
    // Add data to index
    alg_hnsw->addPoint(data3, 0);
    auto end3 = std::chrono::high_resolution_clock::now();
    // 计算时长
    auto duration3 = std::chrono::duration_cast<std::chrono::milliseconds>(end3 - start3);
    std::cout << "第二次执行时长: " << duration3.count() << " 毫秒" << std::endl;

    start3 = std::chrono::high_resolution_clock::now();
    // Add data to index
    alg_hnsw->addPoint(data3, 0);
    end3 = std::chrono::high_resolution_clock::now();
    // 计算时长
    duration3 = std::chrono::duration_cast<std::chrono::milliseconds>(end3 - start3);
    std::cout << "第二次执行时长: " << duration3.count() << " 毫秒" << std::endl;


    // Query the elements for themselves and measure recall
    std::vector<hnswlib::labeltype> neighbors(max_elements);
    ParallelFor(0, max_elements, num_threads, [&](size_t row, size_t threadId) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data + dim * row, 1);
        hnswlib::labeltype label = result.top().second;
        neighbors[row] = label;
    });
    float correct = 0;
    for (int i = 0; i < max_elements; i++) {
        hnswlib::labeltype label = neighbors[i];
        if (label == i) correct++;
    }
    float recall = correct / max_elements;
    std::cout << "Recall: " << recall << "\n";

    delete[] data;
    delete alg_hnsw;
    return 0;
}
