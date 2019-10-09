#ifndef __VAR_PRINT_H__
#define __VAR_PRINT_H__

#include <iostream>

void print() {}

template <typename T, typename... Args>
void print(T val, Args... args) {
    std::cout << val << '\n';
    print(args...);
}

void print_size() {}

template <typename T, typename... Args>
void print_size(T val, Args... args) {
    std::cout << "Args : " << sizeof...(Args) << std::endl;
    std::cout << "args : " << sizeof...(args) << std::endl;
    print_size(args...);
}

#endif  // __VAR_PRINT_H__