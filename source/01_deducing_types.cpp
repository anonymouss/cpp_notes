/**
 * Deducing Types
 * - item 1: Understand template type deduction
 * - item 2: Understand `auto` type deduction
 * - item 3: Understand `decltype`
 * - item 4: Know how to view deduced types
 */

#include "utils.h"

#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <vector>

// basic concepts of type deducing
//   template <typename T>
//   void f(ParamType param);

// case 1. ParamType is a Reference or Pointer, but not a Universal Reference
template <typename T>
void func_case1_r(T &param) {
    DESCRIBE();
}

template <typename T>
void func_case1_cr(const T &param) {
    DESCRIBE();
}

template <typename T>
void func_case1_p(T *param) {
    DESCRIBE();
}

// case 2. ParamType is a universal reference
template <typename T>
void func_case2(T &&param) {
    DESCRIBE();
}

// case 3. ParamType is neither a pointer or a reference
template <typename T>
void func_case3(T param) {
    DESCRIBE();
}

// std::initializer_list for template deduction
template <typename T>
void func_init_list(std::initializer_list<T> &&param) {
    DESCRIBE();
}

void funcArgTest(int, double) {}

// decltype help function
// C++11 version, with trailing return type
template <typename C, typename I>
auto getIndexAt_11(const C &c, I i) -> decltype(c[i]) {
    return c[i];
}

// C++14 version
template <typename C, typename I>
auto getIndexAt_14(const C &c, I i) {
    return c[i];
}

// decltype(auto)
template <typename C, typename I>
decltype(auto) getIndexAt_dc(const C &c, I i) {
    return c[i];
}

int main() {
    int x = 27;         // x    is int
    const int cx = x;   // cx   is const int
    const int &rx = x;  // rx   is const int &
    const int *px = &x;

    // case 1
    {
        INFO("Case 1. with reference");
        func_case1_r(x);   // T should be int,        ParamType should be int &
        func_case1_r(cx);  // T should be const int,  ParamType should be const int &
        func_case1_r(rx);  // T should be const int,  ParamType should be const int &

        INFO("Case 1. with const reference");
        func_case1_cr(x);   // T int, ParamType const int &
        func_case1_cr(cx);  // T int, ParamType const int &
        func_case1_cr(rx);  // T int, ParamType const int &

        INFO("Case 1. with pointer");
        func_case1_p(&x);  // T int,        ParamType int *
        func_case1_p(px);  // T const int,  PramType const int *
    }

    // case 2
    {
        INFO("Case 2. universal reference");
        func_case2(x);   // int &,       int &
        func_case2(cx);  // const int &, const int &
        func_case2(rx);  // const int &, const int &
        func_case2(27);  // int,         int &&
    }

    // case 3
    {
        INFO("Case 3. neither a pointer nor reference");
        func_case3(x);   // int, int
        func_case3(cx);  // int, int
        func_case3(rx);  // int, int

        const char *const ptr = "Fun with pointers";
        // a const pointer points to const char
        func_case3(ptr);  // const char *, const char *
    }

    // array arguments
    {
        INFO("* Array arguments");
        const char char_array[] = "char array";
        // decays to pointer
        func_case3(char_array);    // const char *,         const char *
        func_case2(char_array);    // const cahr (&)[N],    const char (&)[N]
        func_case1_r(char_array);  // const char [N],       const char (&)[N]
    }

    // function arguments
    {
        INFO("* Function arguments");
        func_case3(funcArgTest);    // void (*)(int, double),   void (*)(int, double)
        func_case2(funcArgTest);    // void (&)(int, double),   void (&)(int, double)
        func_case1_r(funcArgTest);  // void (int, double),      void (&)(int, double)
    }

    // auto deduction
    {
        INFO("`auto` deduction");
        auto x1 = 1;       // int
        auto x2(1);        // int
        auto x3 = {1};     // std::initializer_list<int>
        auto x4{1};        // std::initializer_list<int>, but int from C++17
        auto x5 = {1, 2};  // std::initializer_list<int>
        // auto x6{1, 2};  // ERROR

        ECHO_TYPE_PARAM(x1);
        ECHO_TYPE_PARAM(x2);
        ECHO_TYPE_PARAM(x3);
        ECHO_TYPE_PARAM(x4);
        ECHO_TYPE_PARAM(x5);

        // func_case2({1, 2}); // ERROR. Requires to declare param as std::initializer_list
        // implicitly
        func_init_list({1, 2});  // int, std::initializer_list<int>&&

        auto lambda = [](const auto &v) { ECHO_TYPE_PARAM(v); };
        // lambda({1, 2});  // ERROR
    }

    // decltype deduction
    {
        INFO("`decltype` deduction");
        std::vector<int> vec{1, 2, 3};
        ECHO_TYPE_PARAM(getIndexAt_11(vec, 0));  // const int &
        ECHO_TYPE_PARAM(getIndexAt_14(vec, 0));  // int
        // see what difference? although C++14 permits using auto to deduce return type without a
        // trailing return type, it drops cv qualifier. Be careful when using auto.
        // The simpleset way is to use decltype(auto)
        ECHO_TYPE_PARAM(getIndexAt_dc(vec, 0));

        int d = 0;
        ECHO_TYPE_T(decltype(d));    // int
        ECHO_TYPE_T(decltype((d)));  // int &
    }
}