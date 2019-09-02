#include "CheesePizza.h"

#include <iostream>

void CheesePizza::prepare() {
    std::cout << " preparing cheese pizza" << std::endl;
    mDough = mGredientFactory->createDough();
    mSauce = mGredientFactory->createSauce();
    mCheese = mGredientFactory->createCheese();
}