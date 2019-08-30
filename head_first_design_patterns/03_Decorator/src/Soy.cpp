#include "Soy.h"

Soy::Soy(std::shared_ptr<IBeverage> bevarage) {
    mBevarage = bevarage;
    mDescription = ", add some Soy";
}

std::string Soy::getDescription() const { return mBevarage->getDescription() + mDescription; }

double Soy::calculateCost() { return mBevarage->calculateCost() + 2.1; }