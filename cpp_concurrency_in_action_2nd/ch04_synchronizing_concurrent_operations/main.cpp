//  condition variables, futures, latches, and barriers

#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <exception>
#include <future>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>

using namespace std::chrono_literals;

#define BREAK_LINE(str) std::cout << "===== " << str << " =====" << std::endl;

// condition variable
// std::condition_variable 只能与 std::mutex 配合使用
// std::condition_variable_any 可以与任何 BasicLockable 锁配合使用
std::mutex basicTestMutex;
std::condition_variable basicTestCV;
std::atomic<bool> basicTestAlreadyDone = false;

void basic_sender() {
    // std::this_thread::sleep_for(1s);  // A
    std::cout << "[Enter] " << __func__ << std::endl;
    {
        std::scoped_lock lock(basicTestMutex);
        std::cout << "[..ING] " << __func__ << " processing..." << std::endl;
        std::this_thread::sleep_for(5s);
        basicTestAlreadyDone.store(true);
        std::cout << "[..ING] " << __func__ << " ...done" << std::endl;
    }
    basicTestCV.notify_one();
    std::cout << "[Exit ] " << __func__ << std::endl;
}
void basic_receiver() {
    std::cout << "[Enter] " << __func__ << std::endl;
    std::unique_lock lock(basicTestMutex);
    std::cout << "[..ING] " << __func__ << " waiting..." << std::endl;
    basicTestCV.wait(lock, [] { return basicTestAlreadyDone.load(); });  // B
    // basicTestCV.wait(lock);                                           // B'
    std::cout << "[..ING] " << __func__ << " received notification..." << std::endl;
    std::cout << "[Exit ] " << __func__ << std::endl;
}
/**
 * 注意：std::condition_variable::wait() 有两个重载
 * void wait( std::unique_lock<std::mutex>& lock );                 (1)
 * template< class Predicate >
 * void wait( std::unique_lock<std::mutex>& lock, Predicate pred ); (2)
 *
 * 重载(1)必须先获得锁，然后wait先unlock锁并阻塞线程等待通知。在上面的例子里如果B处使用重载（1），则必须去
 * 掉A处的注释以保证receiver先获得锁。否则如果sender先获得锁则receiver会一直阻塞在尝试获得锁，
 * 并且当sender结束后receiver获得锁并开始wait可能永远收不到notify了。
 * 如果使用重载（2），即使发生上面的情况，但由于更新了basicTestAlreadyDone，则wait不会继续阻塞，因为它只有
 * pred不成立时才会阻塞。相当于：
 * while (!pred()) {
 *     wait(lock);
 * }
 */

// thread safe queue
template <typename T>
class threadsafe_queue {
public:
    threadsafe_queue() = default;
    threadsafe_queue(const threadsafe_queue &otherQueue) {
        std::scoped_lock lock(otherQueue.mMutex);
        this->mQueue = otherQueue.mQueue;
    }

    void push(T new_value) {
        std::scoped_lock lock(mMutex);
        mQueue.push(new_value);
        std::cout << "Queue: pushed " << new_value << std::endl;
        mCond.notify_one();
    }

    bool try_pop(T &value) {
        std::scoped_lock lock(mMutex);
        if (mQueue.empty) {
            std::cout << "Queue: try_pop : NO_INIT" << std::endl;
            return false;
        }
        value = mQueue.front();
        mQueue.pop();
        std::cout << "Queue: try_pop " << value << std::endl;
        return true;
    }

    std::shared_ptr<T> try_pop() {
        std::scoped_lock lock(mMutex);
        if (mQueue.empty()) {
            std::cout << "Queue: try_pop : NO_INIT" << std::endl;
            return nullptr;
        }
        auto value = std::make_shared<T>(mQueue.front());
        mQueue.pop();
        std::cout << "Queue: try_pop " << *value << std::endl;
        return value;
    }

    void wait_and_pop(T &value) {
        std::unique_lock lock(mMutex);
        mCond.wait(lock, [this] { return !mQueue.empty(); });
        value = mQueue.front();
        mQueue.pop();
        std::cout << "Queue: wait_and_pop " << value << std::endl;
    }

    std::shared_ptr<T> wait_and_pop() {
        std::unique_lock lock(mMutex);
        mCond.wait(lock, [this] { return !mQueue.empty(); });
        auto value = std::make_shared<T>(mQueue.front());
        mQueue.pop();
        std::cout << "Queue: wait_and_pop " << *value << std::endl;
        return value;
    }

    bool empty() const {
        std::scoped_lock lock(mMutex);
        std::cout << "Queue: empty " << (mQueue.empty() ? "true" : "false") << std::endl;
        return mQueue.empty();
    }

private:
    threadsafe_queue &operator=(const threadsafe_queue &) = delete;

    mutable std::mutex mMutex;
    std::queue<T> mQueue;
    std::condition_variable mCond;
};

threadsafe_queue<int32_t> testThreadsafeQueue;
void data_preparation_thread() {
    std::cout << "[Enter] " << __func__ << std::endl;
    std::this_thread::sleep_for(3s);
    testThreadsafeQueue.push(100);
    std::cout << "[Exit ] " << __func__ << std::endl;
}
void data_processing_thread() {
    std::cout << "[Enter] " << __func__ << std::endl;
    auto v = testThreadsafeQueue.wait_and_pop();
    std::cout << "[..ING] " << __func__ << " got value: " << *v << std::endl;
    testThreadsafeQueue.empty();
    std::cout << "[Exit ] " << __func__ << std::endl;
}

void testQueueThread() {
    std::thread feedThread{data_preparation_thread};
    std::thread handleThread{data_processing_thread};
    feedThread.join();
    handleThread.join();
}

// futures
int32_t find_the_answer_to_ltuae() {
    std::cout << "[Enter] " << __func__ << std::endl;
    std::cout << "[..ING] " << __func__ << " calculating in thread " << std::this_thread::get_id()
              << std::endl;
    std::this_thread::sleep_for(5s);
    std::cout << "[Exit ] " << __func__ << std::endl;
    return 42;
}
void test_basic_future() {
    std::cout << "[Enter] " << __func__ << std::endl;
    auto answer = std::async(find_the_answer_to_ltuae);
    std::cout << "[..ING] " << __func__ << " doing other things in thread "
              << std::this_thread::get_id() << std::endl;
    std::this_thread::sleep_for(2s);
    std::cout << "[..ING] " << __func__ << " I got the answer is " << answer.get() << std::endl;
    std::cout << "[Exit ] " << __func__ << std::endl;
}

// packaged_task
std::mutex taskMutex;
std::deque<std::packaged_task<void()>> tasks;
auto TP_NO_INIT = std::chrono::time_point<std::chrono::system_clock>();
auto anchor = TP_NO_INIT;
bool mock_gui_shutdown_message_received() {
    // mock of receiving shutdown message after 5s
    auto now = std::chrono::system_clock::now();
    if (anchor == TP_NO_INIT) { anchor = now; }
    return (now - anchor) > 5s;
}
void mock_get_and_process_gui_message() {
    std::cout << "Starting processing gui message." << std::endl;
    std::this_thread::sleep_for(1s);
    std::cout << "Gui message process done." << std::endl;
}
void mock_gui_thread() {
    while (!mock_gui_shutdown_message_received()) {
        mock_get_and_process_gui_message();
        std::packaged_task<void()> task;
        {
            std::scoped_lock lock(taskMutex);
            if (tasks.empty()) { continue; }
            task = std::move(tasks.front());
            tasks.pop_front();
        }
        task();
    }
    std::scoped_lock lock(taskMutex);
    std::cout << "Shutting down... unprocessed task count " << tasks.size() << std::endl;
}

template <typename Func>
std::future<void> mock_post_task_for_gui_thread(Func &&f) {
    std::packaged_task<void()> task{f};
    auto fut = task.get_future();
    std::scoped_lock lock(taskMutex);
    tasks.emplace_back(std::move(task));
    return fut;
}

void test_packaged_task() {
    std::thread mock_gui_bg_thread{mock_gui_thread};
    mock_post_task_for_gui_thread([]() { std::cout << "task - 1" << std::endl; });
    mock_post_task_for_gui_thread([]() { std::cout << "task - 2" << std::endl; });
    mock_post_task_for_gui_thread([]() { std::cout << "task - 3" << std::endl; });
    mock_post_task_for_gui_thread([]() { std::cout << "task - 4" << std::endl; });
    mock_post_task_for_gui_thread([]() { std::cout << "task - 5" << std::endl; });
    mock_post_task_for_gui_thread([]() { std::cout << "task - 6" << std::endl; });
    mock_post_task_for_gui_thread([]() { std::cout << "task - 7" << std::endl; });
    mock_gui_bg_thread.join();
}

// test exception
double square_root(double x) {
    if (x < 0) { throw std::out_of_range("x < 0"); }
    return sqrt(x);
}
void test_exception() {
    std::future<double> fut = std::async(square_root, -1);
    try {
        auto result = fut.get();
        std::cout << "Result: " << result << std::endl;
    } catch (std::exception &e) { std::cout << e.what() << std::endl; }

    std::promise<double> p;
    auto res = p.get_future();
    try {
        p.set_value(square_root(-2));
    } catch (...) { p.set_exception(std::current_exception()); }
    try {
        auto result = res.get();
        std::cout << "Result: " << result << std::endl;
    } catch (std::exception &e) { std::cout << e.what() << std::endl; }
}

// a sequential impl of quicksort
template <typename T>
std::list<T> sequential_quick_sort(std::list<T> input, int round = 0) {
    auto round_print = [&](const std::list<T> &list, const char *name) {
        std::cout << "[Round " << round << "] " << name << ": [";
        for (const auto &v : list) { std::cout << v << ", "; }
        std::cout << "]" << std::endl;
    };
    round_print(input, "input");

    if (input.empty()) { return input; }
    std::list<T> result;
    result.splice(result.begin(), input, input.begin());
    const T &pivot = *result.begin();
    round_print(result, "init-result");

    auto divide_point =
        std::partition(input.begin(), input.end(), [&](const T &t) { return t < pivot; });
    std::list<T> lower_part;
    lower_part.splice(lower_part.end(), input, input.begin(), divide_point);
    round_print(lower_part, "lower");
    auto new_lower(sequential_quick_sort(std::move(lower_part), round + 1));
    auto new_higher(sequential_quick_sort(std::move(input), round + 1));
    round_print(new_lower, "new-lower");
    round_print(new_higher, "new-higher");
    result.splice(result.end(), new_higher);
    result.splice(result.begin(), new_lower);
    round_print(result, "final-result");
    return result;
}

// parallel quicksort using futures
template <typename T>
std::list<T> parallel_quick_sort(std::list<T> input) {
    if (input.empty()) { return input; }
    std::list<T> result;
    result.splice(result.begin(), input, input.begin());
    const T &pivot = *result.begin();

    auto divide_point =
        std::partition(input.begin(), input.end(), [&](const T &t) { return t < pivot; });
    std::list<T> lower_part;
    lower_part.splice(lower_part.end(), input, input.begin(), divide_point);
    auto new_lower_fut = std::async(&parallel_quick_sort<T>, std::move(lower_part));
    auto new_higher = parallel_quick_sort(std::move(input));
    result.splice(result.end(), new_higher);
    result.splice(result.begin(), new_lower_fut.get());
    return result;
}

void test_quick_sort() {
    std::list<int32_t> list{5, 7, 3, 4, 1, 9, 2, 8, 10, 6};
    auto print = [](const std::list<int32_t> &lst) {
        for (const auto &v : lst) { std::cout << v << ", "; }
        std::cout << std::endl;
    };

    BREAK_LINE("sequential quick sort")
    auto seq_res = sequential_quick_sort(list);
    print(seq_res);

    BREAK_LINE("parallel quick sort")
    auto par_res = parallel_quick_sort(list);
    print(par_res);
}

int main() {
    BREAK_LINE("Basic condition variable test")
    std::thread basicSenderThread{basic_sender};
    std::thread basicReceiverThread{basic_receiver};
    basicSenderThread.join();
    basicReceiverThread.join();

    BREAK_LINE("Test threadsafe_queue")
    testQueueThread();

    BREAK_LINE("Test basic future")
    test_basic_future();

    BREAK_LINE("Test basic packaged_task")
    test_packaged_task();

    BREAK_LINE("Test exception");
    test_exception();

    BREAK_LINE("Test quick sort");
    test_quick_sort();
}