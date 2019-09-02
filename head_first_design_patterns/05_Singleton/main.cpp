#include "Singleton.h"

#include <future>
#include <iostream>

using namespace std::literals;

void func(int id) {
    std::cout << "enter thread " << id << std::endl;
    auto inst = Singleton::GetInstance();
    auto v = id * 2;
    inst->setValue(v);
    std::this_thread::sleep_for(1s);
    std::cout << id << ": set value to " << v << ", but got " << inst->getValue() << std::endl;
    std::cout << "exit thread " << id << std::endl;
}

int main() {
    auto fut5 = std::async(func, 5);
    auto fut2 = std::async(func, 2);
}

/**
 * output
 * enter thread 5
 * enter thread 2
 * 5: set value to 10, but got 4
 * exit thread 5
 * 2: set value to 4, but got 4
 * exit thread 2
 */