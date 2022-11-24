#include <algorithm>
#include <chrono>
#include <cstdio>
#include <exception>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <stack>
#include <string>
#include <thread>

using namespace std::chrono_literals;

// race condition
// 1.
// （最简单）通过某种保护机制将数据的修改保护起来，只有修改线程能看到不变量的破坏状态，其他线程只能看到修
//          未开始或已完成的状态
// 2.
// （难实现）修改数据结构或不变量的设计，使修改成为一些列不可分割的改变，每个改变都保存不变量（无锁编程）
// 3.
// （不讨论）将数据的一些列更新当做一个事务来处理 software transactional memory (STM)

// 互斥 std::mutex
// example: passing out a ref to protected data
struct some_data {
    explicit some_data(int _a = 0) : a(_a) {}
    void do_something() { std::cout << "a = " << a << std::endl; }

    int a;
};
class data_wrapper {
public:
    template <typename Func>
    void process_data(Func func) {
        std::scoped_lock<std::mutex> lock(mtx);
        func(data);
    }

private:
    some_data data{100};
    std::mutex mtx;
};

some_data *unprotected_data = nullptr;
void malicious_function(some_data &protected_data) {
    unprotected_data = &protected_data;
    if (unprotected_data) { unprotected_data->a *= 2; }
}
data_wrapper wrapper;
void test_stray_ref() {
    wrapper.process_data(malicious_function);
    if (unprotected_data) {
        unprotected_data->do_something();
    } else {
        std::cout << "unprotected_data nullptr" << std::endl;
    }
}

/**
 * thread safe stack pop
 * 1. pass in a reference
 * 2. require a no-throw copy ctor or move ctor
 * 3. return a ptr to the popped item
 * 4. provide both 1 + 2 or 1 + 3
 */
// example
struct empty_stack : std::exception {
    const char *what() const noexcept { return "Empty stack"; }
};

template <typename T>
class threadsafe_stack {
public:
    threadsafe_stack() {}
    threadsafe_stack(const threadsafe_stack &other) {
        std::scoped_lock<std::mutex> lock(other.mMutex);
        mStack = other.mStack;
    }
    void push(T new_value) {
        std::scoped_lock<std::mutex> lock(mMutex);
        mStack.push(std::move(new_value));
    }
    std::shared_ptr<T> pop() {
        std::scoped_lock<std::mutex> lock(mMutex);
        if (mStack.empty()) throw empty_stack{};
        const auto element = std::make_shared<T>(mStack.top());
        mStack.pop();
        return element;
    }
    void pop(T &value) {
        std::scoped_lock<std::mutex> lock(mMutex);
        if (mStack.empty()) throw empty_stack{};
        value = mStack.top();
        mStack.pop();
    }
    bool empty() const {
        std::scoped_lock<std::mutex> lock(mMutex);
        return mStack.empty();
    }

private:
    threadsafe_stack &operator=(const threadsafe_stack &) = delete;
    std::stack<T> mStack;
    mutable std::mutex mMutex;
};

struct Object {
    Object() = default;
    Object(const char *k, int32_t v) : key(k), value(v) {}
    Object(const Object &obj) {
        key = obj.key;
        value = obj.value;
    }

    friend std::ostream &operator<<(std::ostream &os, const Object &obj) {
        os << "[Key]: " << obj.key << " - [Value]: " << obj.value;
        return os;
    }

    std::string key;
    int32_t value;
};

struct X {
    explicit X(const char *n = "?") : name(n) {}
    std::mutex mutex;
    std::string name;
};

void swap(X &lhs, X &rhs) {
    if (&lhs == &rhs) {
        std::cout << "Same objects" << std::endl;
        return;
    }

    {
        // deadlock
        // 以相同的顺序上锁（有可能死锁，实际上无法保证顺序）
        /*
        std::cout << std::this_thread::get_id() << " try to lock " << lhs.name << std::endl;
        lhs.mutex.lock();
        std::this_thread::sleep_for(1s);
        std::cout << std::this_thread::get_id() << " try to lock " << rhs.name << std::endl;
        rhs.mutex.lock();
        std::this_thread::sleep_for(1s);
        std::cout << std::this_thread::get_id() << " unlock " << lhs.name << std::endl;
        lhs.mutex.unlock();
        std::cout << std::this_thread::get_id() << " unlock " << rhs.name << std::endl;
        rhs.mutex.unlock();
        */
    }

    {
        // 通过std::lock 保证同时给多个锁安全上锁
        // std::lock(lhs.mutex, rhs.mutex);
        // std::lock_guard<std::mutex> ll(lhs.mutex, std::adopt_lock);
        // std::lock_guard<std::mutex> lr(rhs.mutex, std::adopt_lock);
        // defer_lock_t	    不获得互斥的所有权
        // try_to_lock_t	尝试获得互斥的所有权而不阻塞
        // adopt_lock_t	    假设调用方线程已拥有互斥的所有权
    }

    {
        // 等价方法
        // std::unique_lock<std::mutex> ll(lhs.mutex, std::defer_lock);
        // std::unique_lock<std::mutex> lr(rhs.mutex, std::defer_lock);
        // std::lock(ll, lr);
    }

    // 等价
    std::scoped_lock lock(lhs.mutex, rhs.mutex);
}

// hierarchical mutex
class hierarchical_mutex {
public:
    explicit hierarchical_mutex(uint64_t hierarchical_value)
        : mIncomingHierarchyValue(hierarchical_value), mPreviousHierarchyValue(0) {}
    void lock() {
        check_for_hierarchy_violation();
        mMutex.lock();
        update_hierarchy_value();
    }
    void unlock() {
        if (sThisThreadHierarchyValue != mIncomingHierarchyValue) {
            throw std::runtime_error("Mutex is not locked");
        }
        sThisThreadHierarchyValue = mPreviousHierarchyValue;
        mMutex.unlock();
    }
    bool try_lock() {
        check_for_hierarchy_violation();
        if (!mMutex.try_lock()) { return false; }
        update_hierarchy_value();
        return true;
    }

private:
    std::mutex mMutex;
    const uint64_t mIncomingHierarchyValue;
    uint64_t mPreviousHierarchyValue;
    static thread_local uint64_t sThisThreadHierarchyValue;

    void check_for_hierarchy_violation() {
        if (sThisThreadHierarchyValue <= mIncomingHierarchyValue) {
            throw std::runtime_error("mutex hierarchy violated");
        }
    }

    void update_hierarchy_value() {
        mPreviousHierarchyValue = sThisThreadHierarchyValue;
        sThisThreadHierarchyValue = mIncomingHierarchyValue;
        std::cout << "Updated this: " << sThisThreadHierarchyValue
                  << ", prev: " << mPreviousHierarchyValue << std::endl;
    }
};

thread_local uint64_t hierarchical_mutex::sThisThreadHierarchyValue{UINT64_MAX};

hierarchical_mutex highHierMutex(90);
hierarchical_mutex middleHierMutex(60);
hierarchical_mutex lowHierMutex(30);
int do_low() {
    std::cout << __func__ << std::endl;
    return 0;
}
int low_func() {
    std::cout << __func__ << std::endl;
    std::scoped_lock lock(lowHierMutex);
    return do_low();
}
void do_high(int param) { std::cout << __func__ << ": " << param << std::endl; }
void high_func() {
    std::cout << __func__ << std::endl;
    std::scoped_lock lock(highHierMutex);
    do_high(low_func());
}
void thread_a() {
    std::cout << __func__ << ": in" << std::endl;
    high_func();
    std::cout << __func__ << ": out" << std::endl;
}
void do_middle() { std::cout << __func__ << std::endl; }
void middle_func() {
    std::cout << __func__ << std::endl;
    high_func();
    do_middle();
}
void thread_b() {
    std::cout << __func__ << ": in" << std::endl;
    std::scoped_lock lock(middleHierMutex);
    try {
        middle_func();
    } catch (std::exception &e) { std::cout << e.what() << std::endl; }
    std::cout << __func__ << ": out" << std::endl;
}

void test_hier_mutex() {
    std::thread threadA{thread_a};
    threadA.join();
    std::this_thread::sleep_for(100ms);
    std::thread threadB{thread_b};
    threadB.join();
}

// transfer ownership
std::unique_lock<std::mutex> get_lock() {
    std::mutex some_mutex;
    std::unique_lock<std::mutex> l(some_mutex);
    // do something
    return l;
}

void test() { std::unique_lock l(get_lock()); }

int main() {
    test_stray_ref();

    threadsafe_stack<Object> ost;
    ost.push({"A", 1});
    ost.push({"B", 2});
    ost.push({"C", 3});
    std::cout << *ost.pop() << std::endl;
    Object o;
    ost.pop(o);
    std::cout << o << std::endl;
    ost.pop();
    try {
        ost.pop();
    } catch (std::exception &e) { std::cout << e.what() << std::endl; }

    // test deadlock
    X xa{"a"}, xb{"b"};
    std::thread thread_ab{swap, std::ref(xa), std::ref(xb)};
    std::thread thread_ba{swap, std::ref(xb), std::ref(xa)};
    thread_ab.join();
    thread_ba.join();

    test_hier_mutex();

    test();
}