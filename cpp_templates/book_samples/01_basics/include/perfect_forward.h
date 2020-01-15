#ifndef __PERFECT_FORWARD_H__
#define __PERFECT_FORWARD_H__

#include <utility>
#include <iostream>

struct X {};

void g(X &) { std::cout << "variable" << std::endl; }
void g(const X &) { std::cout << "constant" << std::endl; }
void g(X &&) { std::cout << "movable" << std::endl; }

template <typename T>
void f (T &&o) {
    g(std::forward<T>(o));
}

#endif // __PERFECT_FORWARD_H__