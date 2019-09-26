#include "max1.h"

#include <iostream>
#include <string>

int main() {
    std::cout << ::max(1, 2) << std::endl;
    std::cout << ::max(3.4, -6.7) << std::endl;
    std::string s1 = "mathematics", s2 = "math";
    std::cout << ::max(s1, s2) << std::endl;
}