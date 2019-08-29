#ifndef __WEATHER_DATA_H__
#define __WEATHER_DATA_H__

#include "IObserver.h"
#include "ISubject.h"

#include <list>

class WeatherData : public ISubject {
public:
    void registerObserver(IObserver *observer) final;
    void removeObserver(IObserver *observer) final;
    void notifyObserverAll() final;

    void measurementsChanged();
    void setMeasurements(float temperature, float humidity, float pressure);

private:
    std::list<IObserver *> mObservers;
    float temperature;
    float humidity;
    float pressure;
};

#endif  // __WEATHER_DATA_H__