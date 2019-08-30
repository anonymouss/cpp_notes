#include "Mocha.h"

Mocha::Mocha(std::shared_ptr<IBeverage> bevarage) {
    mBevarage = bevarage;
    mDescription = ", add some mocha";
}

std::string Mocha::getDescription() const { return mBevarage->getDescription() + mDescription; }

double Mocha::calculateCost() { return mBevarage->calculateCost() + 2.3; }