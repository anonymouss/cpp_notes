#include "ClamPizza.h"

#include <iostream>

void ClamPizza::prepare() {
    std::cout << " preparing clam pizza" << std::endl;
    mDough = mGredientFactory->createDough();
    mSauce = mGredientFactory->createSauce();
    mClams = mGredientFactory->createClams();
}