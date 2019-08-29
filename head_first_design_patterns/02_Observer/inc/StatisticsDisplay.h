#ifndef __STATISTICS_DISPLAY_H__
#define __STATISTICS_DISPLAY_H__

#include "IDisplayElement.h"
#include "IObserver.h"
#include "ISubject.h"

#include <memory>
#include <vector>

class StatisticsDisplay : public IObserver, IDisplayElement {
public:
    virtual ~StatisticsDisplay() { detach(); };

    void attach(std::shared_ptr<ISubject> subject) final;
    void detach() final;
    void update(float temp, float humidity, float pressure) final;
    void display() const final;

private:
    float aveTemperature;
    float aveHumidity;
    float avePressure;

    float tmpSum = 0;
    float humiSum = 0;
    float pressSum = 0;

    std::vector<float> temps, humidities, pressures;

    std::shared_ptr<ISubject> mSubjectImpl;
};

#endif  // __STATISTICS_DISPLAY_H__