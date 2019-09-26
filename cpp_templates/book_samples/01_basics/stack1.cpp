#include "stack1.h"

#include <iostream>
#include <string>

int main() {
    Stack<int> intStack;
    Stack<std::string> strStack;

    intStack.push(7);
    std::cout << intStack.top() << std::endl;

    strStack.push("hello");
    std::cout << strStack.top() << std::endl;
    strStack.pop();

    strStack.push("hello");
    strStack.push("world");
    std::cout << strStack << std::endl;
}