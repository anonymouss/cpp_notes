#include "isprime.h"

#include <iostream>

#define CHECK_PRIME(v) \
    std::cout << std::boolalpha \
                << v << " is prime number : " << IsPrime<v>::value << std::endl; 

int main() {
    CHECK_PRIME(0);
    CHECK_PRIME(1);
    CHECK_PRIME(2);
    CHECK_PRIME(3);
    CHECK_PRIME(4);
    CHECK_PRIME(5);
    CHECK_PRIME(6);
    CHECK_PRIME(7);
    CHECK_PRIME(8);
    CHECK_PRIME(9);
    CHECK_PRIME(10);

    PrimeHelper<5> h5;
    PrimeHelper<10> h10;

    h5.description();
    h10.description();
}