#include <algorithm>
#include <chrono>
#include <exception>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

class background_task {
public:
    void operator()() const {
        do_something("task 1");
        do_something("task 2");
    }

private:
    void do_something(const char *task_name) const {
        std::cout << __func__ << ": " << task_name << std::endl;
    }
};

struct func {
    int &i;
    func(int &i_) : i(i_) {}
    void operator()() {
        for (unsigned j = 0; j < 1000000; ++j) { std::cout << i << ", "; }
        std::cout << std::endl;
    }
};

void oops() {
    {
        int some_local_state = 1;
        func my_func{some_local_state};
        std::thread my_thread{my_func};
        my_thread.detach();
    }
    std::cout << "\n out of scope" << std::endl;
    std::this_thread::sleep_for(500ms);
}

// detach 线程很简单；但是如果需要join，则需要仔细考虑join的时机。如线程可能抛出异常
void throw_exception(const char *eMsg = "Some Exception") { throw std::runtime_error{eMsg}; }

void another_thread() {
    std::this_thread::sleep_for(200ms);
    std::cout << __func__ << std::endl;
}

void test_exception_case() {
    std::thread t(another_thread);
#if 0
    throw_exception();
#else
    try {
        throw_exception();
    } catch (...) {
        std::cout << "[ExceptionCaught] in " << __func__ << std::endl;
        // 如果此处没有join，异常发生时该线程结束，既没有detach也没有join，则会调用std::terminate
        t.join();
        return;
    }
#endif
    std::cout << "[NormalCase] in " << __func__ << std::endl;
    t.join();
}

// try-catch 会使代码冗余且作用域混乱，更好的方法是使用RAII
class thread_guard {
public:
    explicit thread_guard(std::thread &thread) : mThread(thread) {}
    virtual ~thread_guard() {
        if (mThread.joinable()) { mThread.join(); }
    }

private:
    thread_guard(const thread_guard &) = delete;
    thread_guard &operator=(const thread_guard &) = delete;

    std::thread &mThread;
};

void test_exception_case_raii() {
    std::cout << "[Test] in " << __func__ << std::endl;
    std::thread t(another_thread);
    thread_guard guard{t};
    try {
        throw_exception();
    } catch (...) { return; }
}

// pass argument - pointer to automatic variable
void pass_argument_by_reference(const std::string &str) {
    std::cout << __func__ << " | " << str << std::endl;
}
void test_pass_argument() {
    char buffer[1024];
    sprintf(buffer, "hello, test pass argument");
    // buffer: const char * -> std::string 转换发生在 thread 内部，在转换完成前 buffer 可能就被改写
    // std::thread thread{pass_argument_by_reference, buffer};
    // 显式转换，在构建thread前就转换，以免buffer被改写
    std::thread thread{pass_argument_by_reference, std::string{buffer}};
    buffer[0] = 'H';
    thread.detach();
}

struct Data {
    Data() { std::cout << __func__ << ": ctor" << std::endl; }
    ~Data() { std::cout << __func__ << ": dtor" << std::endl; }
    Data(const Data &) { std::cout << __func__ << ": cpy-ctor" << std::endl; }
    Data &operator=(const Data &) {
        std::cout << __func__ << ": cpy-assign" << std::endl;
        return *this;
    }
};
void update_data(Data &data) { std::cout << __func__ << std::endl; }
void test_update_data() {
    Data data;
    // 编译失败！thread构造无视函数需要ref参数，将会盲目拷贝。随后在内部尝试move，但这里函数要求非常量引用，
    // 非常量引用是无法引用右值的，因此编译失败。
    // std::thread thread{update_data, data};
    // 解决办法是使用 std::ref/cref 包装，
    std::thread thread{update_data, std::ref(data)};
    thread.join();
}

// test class
struct Object {
    void say(std::string something = "?") { std::cout << "hello " << something << std::endl; }
};

// 另一个传递参数的情况是不能copy只能move，比如std::unique_ptr
void test_move(std::unique_ptr<Object> obj) { obj->say(__func__); }
// NOTE: std::thread 也有类似的ownership语义，不可copy可move

// example of moving thread ownership
class scoped_thread final {
public:
    explicit scoped_thread(std::thread thread) : mThread(std::move(thread)) {
        if (!mThread.joinable()) { throw std::logic_error("Invalid thread"); }
    }
    ~scoped_thread() { mThread.join(); }

private:
    scoped_thread(const scoped_thread &) = delete;
    scoped_thread &operator=(const scoped_thread &) = delete;

    std::thread mThread;
};

void test_move_thread() {
    Object obj;
    std::thread thread{&Object::say, &obj, std::string{"move thread"}};
    scoped_thread sThread1{std::move(thread)};

    try {
        // thread has been moved
        scoped_thread sThread2{std::move(thread)};
    } catch (std::exception &e) { std::cout << e.what() << std::endl; }
}

// example of parralleled std::accumulate
template <typename Iterator, typename T>
struct accumulate_block {
    void operator()(Iterator first, Iterator last, T *result) {
        *result = std::accumulate(first, last, *result /* init val */);
    }
};

template <typename Iterator, typename T>
T parallel_accumulate(Iterator first, Iterator last, T init = 0) {
    const unsigned long length = std::distance(first, last);
    if (length == 0) { return init; }
    const unsigned long min_blocks = 25;
    const unsigned long max_threads = (length + min_blocks - 1) / min_blocks;
    const unsigned long hardware_limits = std::thread::hardware_concurrency();
    const unsigned long thread_count =
        std::min(hardware_limits != 0 ? hardware_limits : 2, max_threads);
    const unsigned long block_size = length / thread_count;

    std::vector<T> results(thread_count);
    std::vector<std::thread> threads(thread_count - 1);
    auto block_start = first;
    for (auto i = 0; i < (thread_count - 1); ++i) {
        auto block_end = block_start;
        std::advance(block_end, block_size);
        threads[i] =
            std::thread{accumulate_block<Iterator, T>(), block_start, block_end, &(results[i])};
        block_start = block_end;
    }
    accumulate_block<Iterator, T>()(block_start, last, &(results[thread_count - 1]));

    for (auto &entry : threads) { entry.join(); }
    return std::accumulate(results.begin(), results.end(), init);
}

int main() {
    background_task backgroundTask;
    std::thread backgroundThread{backgroundTask};

    // 如果在thread对象销毁前没有指定join/detach，std::thread 析构会调用 std::terminate() 中断
    backgroundThread.join();

    // std::thread tOops(oops);
    // tOops.join();

    std::thread exception_test_thread(test_exception_case);
    exception_test_thread.join();
    std::thread exception_test_raii_thread(test_exception_case_raii);
    exception_test_raii_thread.join();

    test_pass_argument();
    std::this_thread::sleep_for(50ms);

    // test class
    Object object;
    std::thread clsThread{&Object::say, &object, std::string{"world"}};
    // std::thread clsThread{&Object::say, std::ref(object), std::string{"earth"}};
    // invoke object.say();
    clsThread.join();

    // move test
    auto pObj = std::make_unique<Object>();
    std::thread moveThread{test_move, std::move(pObj)};
    moveThread.join();

    // move thread test
    test_move_thread();

    std::cout << "Hardware support thread count: " << std::thread::hardware_concurrency()
              << std::endl;

    std::vector<uint32_t> nums;
    for (uint32_t i = 0; i <= 100 /* 0000000 */; ++i) { nums.emplace_back(i); }
    auto tp_0 = std::chrono::system_clock::now();
    std::cout << parallel_accumulate<std::vector<uint32_t>::iterator, uint32_t>(nums.begin(),
                                                                                nums.end())
              << std::endl;
    auto tp_1 = std::chrono::system_clock::now();
    std::cout << std::accumulate<std::vector<uint32_t>::iterator, uint32_t>(nums.begin(),
                                                                            nums.end(), 0)
              << std::endl;
    auto tp_2 = std::chrono::system_clock::now();
    std::chrono::duration<double> diff_10 = tp_1 - tp_0;
    std::chrono::duration<double> diff_21 = tp_2 - tp_1;
    std::cout << "Duration 1: " << diff_10.count() << "s" << std::endl;
    std::cout << "Duration 2: " << diff_21.count() << "s" << std::endl;

    std::thread thread_1{backgroundTask};
    std::cout << thread_1.get_id() << std::endl;
    std::thread thread_2 = std::move(thread_1);
    std::cout << thread_2.get_id() << std::endl;
    std::cout << thread_1.get_id() << std::endl;

    if (thread_1.joinable()) { thread_1.join(); }
    if (thread_2.joinable()) { thread_2.join(); }
}