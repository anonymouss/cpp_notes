/**
 * Moving to Modern C++
 * - item  7. Distinguish between `()` and `{}` when creating objects
 * - item  8. Prefer `nullptr` to `0` and `NULL`
 * - item  9. Prefer alias declaration to `typedef`s
 * - item 10. Prefer scoped `enum`s to unscoped `enum`s
 * - item 11. Prefer deleted functions to private undefined ones
 * - item 12. Declare overriding functions `override`
 * - item 13. Prefer `const_iterator`s to `iterator`s
 * - item 14. Declare functions `noexcept` if they won't emit exceptions
 * - item 15. Use `constexpr` whenever possible
 * - item 16. Make `const` member functions thread safe
 * - item 17. Understand special member function generation
 */

#include "utils.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <initializer_list>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

struct A {
    int x{0};
    int y = 1;
    // int z(3);  // ERROR
};

struct B {
    int x;
    int y;
    // ctor 1
    B(int ix, int iy) : x(ix), y(iy) { INFO("ctor (int, int)"); }
    // ctor 2
    B(std::initializer_list<double> li) { INFO("ctor (std::initializer_list<double>)"); }

    B(const B &b) { INFO("default copy ctor"); }  // cpy ctor
    operator double() const {
        INFO("Conver to float");
        return 0.1;
    }
};

template <typename T>
struct SimpleVector {
    SimpleVector(T v, int sz) {
        printf("Create a simple vector with size %d and initialize all elements to %f\n", sz,
               static_cast<double>(v));
    }
    SimpleVector(std::initializer_list<T> il) {
        printf("Create a simple vector and initialize it via {");
        for (auto e : il) { printf("%f, ", static_cast<double>(e)); }
        printf(")\n");
    }
};

template <typename T, typename... Ts>
void doSomeWork(Ts &&... params) {
    INFO("using ()");
    T t1(std::forward<Ts>(params)...);
    INFO("using {}");
    T t2{std::forward<Ts>(params)...};
}

struct RefQualifierDemo {
    void echo() & { INFO("echo() &"); }
    void echo() && { INFO("echo() &&"); }
};

constexpr int add_constexpr(int a, int b) { return a + b; }

class ConstexprPixel {
public:
    constexpr ConstexprPixel(int x, int y) : X(x), Y(y) {}
    void where() const { printf("I am at (%d, %d)\n", X, Y); }

private:
    int X;
    int Y;
};

constexpr ConstexprPixel createConstexprPixel(int x, int y) {
    if (x < 0 || y < 0) {
        return ConstexprPixel{0, 0};
    } else {
        return ConstexprPixel{x, y};
    }
}

class ThreadSafe {
public:
    int magicValue() const {
        std::lock_guard<std::mutex> guard(mMutex);  // for solution 3
        if (mIsCacheValid) {
            printf("already cached, return...\n");
            return mCachedValue;
        } else {
            printf("not cached yet, calculating...\n");
            // issue here: two threads call here, they both find mIsCacheValid == false....
            // auto v1 = expensiveComputation();
            // auto v2 = expensiveComputation();
            // mCachedValue = v1 + v2;
            // mIsCacheValid = true;
            // return mCachedValue;

            // issue here: two threads call here, one set mIsCacheValid to true but still
            // calculating. another finds mIsCacheValid == ture, return uninitialized cache value
            // mIsCacheValid = true; return mCachedValue = expensiveComputation() +
            // expensiveComputation();

            // solution 3, protected by mutex
            auto v1 = expensiveComputation();
            auto v2 = expensiveComputation();
            mCachedValue = v1 + v2;
            mIsCacheValid = true;
            return mCachedValue;
        }
    }

    int expensiveComputation() const {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        std::random_device rd;
        std::uniform_int_distribution<int> dist(0, 100);
        return dist(rd);
    }

private:
    // mutable std::atomic<bool> mIsCacheValid{false};
    // mutable std::atomic<int> mCachedValue;

    mutable std::mutex mMutex;
    mutable bool mIsCacheValid{false};
    mutable int mCachedValue;
};

void threadFunc(ThreadSafe *tsObj, int threadId) {
    printf("Entering thread %d\n", threadId);
    auto mgv = tsObj->magicValue();
    printf("Thread %d got magic value: %d\n", threadId, mgv);
    printf("Exiting  thread %d\n", threadId);
}

int main() {
    INFO("() and {}");
    std::atomic<int> ati1{1};
    std::atomic<int> ati2(2);
    // std::atomic<int> ati3 = 3; // ERROR
    // ==> braced initialization is uniform
    // also, it can prohibit implicit narrowing conversions
    double x = 1.0, y = 2.0, z = 3.0;
    // int sum1 {x + y + z}; // error, need static_cast<int>()
    int sum2(x + y + z);   // pass, implicit conversion
    int sum3 = x + y + z;  // pass, ditto
    // avoid ambiguity
    // A a(); // ctor or func declaration?

    // * {} drawbacks
    // auto - std::initializer_list
    B b1(1, 1);
    B b2{1, 1};  // if ctor 2 exist, match ctor 2. otherwise, match ctor 1
    B b3(b1);    // cpy ctor
    // the book says, it will call ctor 2 when converion from B to double is define.
    // but it still calls cpy ctor in my experiment. this may because the deduction role changed
    // for auto from braced init-list
    // http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n3922.html
    B b4{b1};

    // another example shows as a template designer, we MUST KEEP difference of {} () in mind.
    // std::vector has such issue.
    doSomeWork<SimpleVector<int>>(0, 10);

    INFO("ref-qualifier");
    RefQualifierDemo rq;
    rq.echo();                  // lvalue
    RefQualifierDemo().echo();  // rvalue

    INFO("constexpr");
    constexpr int res1 = add_constexpr(1, 2);
    int ia = 3, ib = 4;
    // constexpr int res2 = add_constexpr(ia, ib); // ERROR: ia, ib are known at runtime. constexpr
    // func acts like normal function
    int res2 = add_constexpr(ia, ib);

    auto px1 = createConstexprPixel(-1, -1);
    auto px2 = createConstexprPixel(10, 20);
    px1.where();
    px2.where();

    INFO("const function to be thread safe");
    ThreadSafe threadSafe;
    std::thread t1(threadFunc, &threadSafe, 1);
    std::thread t2(threadFunc, &threadSafe, 2);
    t1.join();
    t2.join();
}