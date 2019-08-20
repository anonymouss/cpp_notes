/**
 * The Concurrency API
 * - item 35: Prefer task-based programming to thread-based.
 * - item 36: Specify `std::launch::async` if asynchronicity is essential.
 * - item 37: Make `std::thread`s unjoinable on all paths.
 * - item 38: Be aware of varying thread handle destructor behavior.
 * - item 39: Consider `void` futures for one-shot event communication.
 * - item 40: Use `std::atomic` for concurrency, `volatile` for special memeory.
 */

#include "utils.h"

#include <cstdio>
#include <functional>
#include <future>
#include <mutex>
#include <numeric>
#include <string>
#include <vector>

using namespace std::literals;

template <typename Iter>
int parallelSum(Iter beg, Iter end) {
    auto len = end - beg;
    if (len < 100) { return std::accumulate(beg, end, 0); }

    Iter mid = beg + len / 2;
    auto fut = std::async(std::launch::async, parallelSum<Iter>, mid, end);
    return parallelSum(beg, mid) + fut.get();
}

std::mutex mtx;
struct Task {
    void foo(int i, const std::string &str) {
        std::lock_guard<std::mutex> _lock(mtx);
        printf("%s - %d\n", str.c_str(), i);
    }

    void bar(const std::string &str) {
        std::lock_guard<std::mutex> _lock(mtx);
        printf("%s\n", str.c_str());
    }

    int operator()(int i) {
        std::lock_guard<std::mutex> _lock(mtx);
        printf("%d\n", i);
        return i + 10;
    }
};

void f() {
    printf("Executing... f\n");
    std::this_thread::sleep_for(1s);
    printf("Execut Done... f\n");
}

constexpr auto kTenMillion = 10000000;
bool doWork(std::function<bool(int)> filter, int max = kTenMillion) {
    std::vector<int> goodVals;
    std::thread t([&filter, max, &goodVals] {
        for (auto i = 0; i < max; ++i) {
            if (filter(i)) { goodVals.push_back(i); }
        }
    });

    auto nh = t.native_handle();
}

int main() {
    INFO("task based programming");
    {
        {
            std::vector<int> vec(10000, 2);
            printf("sum of vec is %d\n", parallelSum(vec.cbegin(), vec.cend()));
        }
        {
            Task t;
            auto fooFut = std::async(&Task::foo, &t, 42, "Hello");
            auto barFut = std::async(std::launch::deferred, &Task::bar, &t, "world");
            auto opFut = std::async(std::launch::async, Task(), 24);

            barFut.wait();
            printf("%d\n", opFut.get());
        }
    }

    INFO("policy");
    {
        auto fut = std::async(f);  // with default policy
        int try_count = 0;
        while (fut.wait_for(100ms) != std::future_status::ready) {
            printf("Trying %dst time\n", try_count + 1);
            if (try_count++ >= 20) {
                printf("Timed out...\n");
                break;
            }
        }
    }
}