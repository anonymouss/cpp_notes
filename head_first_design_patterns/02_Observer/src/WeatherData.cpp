#include "WeatherData.h"

#include <algorithm>
#include <iostream>

void WeatherData::registerObserver(IObserver *observer) {
    if (std::find(mObservers.cbegin(), mObservers.cend(), observer) != mObservers.cend()) {
        std::cout << "WARNING: this observer is already registered!" << std::endl;
        return;
    }
    mObservers.emplace_back(observer);
    std::cout << "INFO: new observer registered. total :" << mObservers.size() << std::endl;
}

void WeatherData::removeObserver(IObserver *observer) {
    if (std::find(mObservers.cbegin(), mObservers.cend(), observer) == mObservers.cend()) {
        std::cout << "WARNING: this observer is NOT registered!" << std::endl;
        return;
    }
    mObservers.remove(observer);
    std::cout << "INFO: one observer removed. total :" << mObservers.size() << std::endl;
}

void WeatherData::notifyObserverAll() {
    for (auto *o : mObservers) {
        if (o) { o->update(temperature, humidity, pressure); }
    }
}

void WeatherData::measurementsChanged() { notifyObserverAll(); }

void WeatherData::setMeasurements(float temperature, float humidity, float pressure) {
    std::cout << "INFO: station is updating measurements... will notify observers..." << std::endl;
    this->temperature = temperature;
    this->humidity = humidity;
    this->pressure = pressure;
    measurementsChanged();
}
