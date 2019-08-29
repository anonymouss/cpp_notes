#include "CurrentConditionsDisplay.h"

#include <iostream>

constexpr const char *kName = "<<-- Current Conditions Display Board -->>";

void CurrentConditionsDisplay::attach(std::shared_ptr<ISubject> subject) {
    if (mSubjectImpl) {
        std::cout << "WARNING: already registered to one station" << std::endl;
        return;
    }
    mSubjectImpl = subject;
    mSubjectImpl->registerObserver(this);
}

void CurrentConditionsDisplay::detach() {
    if (!mSubjectImpl) {
        std::cout << "WARNING: you have not resgister to any station" << std::endl;
        return;
    }
    mSubjectImpl->removeObserver(this);
}

void CurrentConditionsDisplay::update(float temp, float humidity, float pressure) {
    this->temperature = temp;
    this->humidity = humidity;
    display();
}

void CurrentConditionsDisplay::display() const {
    std::cout << kName << std::endl;
    std::cout << "  : current temperature : " << temperature << "F degrees." << std::endl;
    std::cout << "  : current    humidity : " << humidity << "% humidity." << std::endl;
    std::cout << "=========================================" << std::endl;
}