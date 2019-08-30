#include "Milk.h"

Milk::Milk(std::shared_ptr<IBeverage> bevarage) {
    mBevarage = bevarage;
    mDescription = ", add some milk";
}

std::string Milk::getDescription() const { return mBevarage->getDescription() + mDescription; }

double Milk::calculateCost() { return mBevarage->calculateCost() + 2.56; }