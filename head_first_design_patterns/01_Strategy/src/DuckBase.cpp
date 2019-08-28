#include "DuckBase.h"

#include <iostream>
#include <string>

void DuckBase::performQuack() {
    std::cout << "[" << mName << "]" << std::endl;
    if (pFlyBehavior) {
        pFlyBehavior->fly();
    } else {
        std::cout << "!!! No fly behavior registered" << std::endl;
    }
}

void DuckBase::performFly() {
    std::cout << "[" << mName << "]" << std::endl;
    if (pQuackBehavior) {
        pQuackBehavior->quack();
    } else {
        std::cout << "!!! No quack behavior registered" << std::endl;
    }
}

void DuckBase::setFlyBehavior(std::unique_ptr<IFlyBehavior> flyBehavior) {
    pFlyBehavior = std::move(flyBehavior);
}

void DuckBase::setQuackBehavior(std::unique_ptr<IQuackBehavior> quackBehavior) {
    pQuackBehavior = std::move(quackBehavior);
}