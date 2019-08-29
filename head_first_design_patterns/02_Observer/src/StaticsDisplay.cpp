#include "StatisticsDisplay.h"

#include <iostream>

constexpr const char *kName = "<<-- Statistics Display Board -->>";

void StatisticsDisplay::attach(std::shared_ptr<ISubject> subject) {
    if (mSubjectImpl) {
        std::cout << "WARNING: already registered to one station" << std::endl;
        return;
    }
    mSubjectImpl = subject;
    mSubjectImpl->registerObserver(this);
}

void StatisticsDisplay::detach() {
    if (!mSubjectImpl) {
        std::cout << "WARNING: you have not resgister to any station" << std::endl;
        return;
    }
    mSubjectImpl->removeObserver(this);
}

void StatisticsDisplay::update(float temp, float humidity, float pressure) {
    temps.emplace_back(temp);
    humidities.emplace_back(humidity);
    pressures.emplace_back(pressure);

    tmpSum += temp;
    humiSum += humidity;
    pressSum += pressure;

    aveTemperature = temp / temps.size();
    aveHumidity = humiSum / humidities.size();
    avePressure = pressSum / pressures.size();

    display();
}

void StatisticsDisplay::display() const {
    std::cout << kName << std::endl;
    std::cout << "  : average temperature : " << aveTemperature << "F degrees." << std::endl;
    std::cout << "  : average    humidity : " << aveHumidity << "% humidity." << std::endl;
    std::cout << "  : average    pressure : " << avePressure << "Kpa." << std::endl;
    std::cout << "=========================================" << std::endl;
}
