#include "Whip.h"

Whip::Whip(std::shared_ptr<IBeverage> bevarage) {
    mBevarage = bevarage;
    mDescription = ", add some whip";
}

std::string Whip::getDescription() const { return mBevarage->getDescription() + mDescription; }

double Whip::calculateCost() { return mBevarage->calculateCost() + 1.56; }