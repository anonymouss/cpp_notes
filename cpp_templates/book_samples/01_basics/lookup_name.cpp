// ADL

#include <iostream>

template <typename T>
T max(T a, T b) { return b < a ? a : b; }

namespace BigMath {
    class BigNumber {
    public:
        BigNumber(int v) : value(v) {}
        int value;
    };

    bool operator < (const BigNumber &lhs, const BigNumber &rhs) {
        return lhs.value < rhs.value;
    }
}

using BigMath::BigNumber;
void g(const BigNumber &a, const BigNumber &b) {
    auto c = ::max(a, b);
}

namespace X {
    template <typename T> void f(T) { std::cout << "X::f()" << std::endl; }
}

namespace N {
    using namespace X;
    enum E { e1 };
    void f(E) { std::cout << "N::f(N::E)" << std::endl; }
}

void f(int) {
    std::cout << "::f()" << std::endl;
}

template <typename X>
struct Base {
    int basefield;
    typedef int T;  // NOTE
};

/*
template <typename T>
struct DD : public Base<T> {    // dependent base class
    void f() { basefield = 0; }
};
*/

template <typename T>
struct DD : public Base<T> {    // dependent base class
    void f() { this->basefield = 0; } // ERROR!
};

/*
template <>
struct Base<bool> {  // explicitly specialization
    enum { basefield = 42 }; // tricky
};
*/

void gg(DD<bool> &d) { d.f(); } // oops?

int main() {
    BigNumber a{1};
    BigNumber b{2};

    g(a, b);

    ::f(N::e1); // ::f() - qualified func, no ADL
    f(N::e1);   // N::f() - ADL
}