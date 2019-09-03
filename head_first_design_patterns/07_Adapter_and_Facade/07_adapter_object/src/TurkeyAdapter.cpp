#include "TurkeyAdapter.h"

#include <iostream>

void TurkeyAdapter::quack() {
    if (mTurkey) {
        mTurkey->gobble();
    } else {
        std::cout << "WARNING: no turkey wrapped" << std::endl;
    }
}

void TurkeyAdapter::fly() {
    if (mTurkey) {
        mTurkey->fly();
    } else {
        std::cout << "WARNING: no turkey wrapped" << std::endl;
    }
}