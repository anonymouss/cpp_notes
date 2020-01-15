#ifndef __IS_PRIME_H__
#define __IS_PRIME_H__

#include <iostream>

template <unsigned p, unsigned d>
struct DoIsPrime {
    static constexpr bool value = (p % d != 0) && DoIsPrime<p, d - 1>::value;
};

template <unsigned p>
struct DoIsPrime<p, 2> {
    static constexpr bool value = (p % 2 != 0);
};

template <unsigned p>
struct IsPrime {
    static constexpr bool value = DoIsPrime<p, p / 2>::value;
};

// special cases
template <>
struct IsPrime<0> { static constexpr bool value = false; };

template <>
struct IsPrime<1> { static constexpr bool value = false; };

template <>
struct IsPrime<2> { static constexpr bool value = true; };

template <>
struct IsPrime<3> { static constexpr bool value = true; };

// select path with partial specialization
template <int SZ, bool = IsPrime<SZ>::value>
struct PrimeHelper;

template <int SZ>
struct PrimeHelper<SZ, false> {
    void description() {
        std::cout << SZ << " - I am not a prime number" << std::endl;
    }
};

template <int SZ>
struct PrimeHelper<SZ, true> {
    void description() {
        std::cout << SZ << " - I am a prime number" << std::endl;
    }
};

#endif  // __IS_PRIME_H__