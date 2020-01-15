#include "perfect_forward.h"

int main() {
    X vx;
    const X cx;

    f(vx);
    f(cx);
    f(X());
    f(std::move(vx));
}