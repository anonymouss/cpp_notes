#ifndef __CURRENT_CONDITIONS_DISPLAY_H__
#define __CURRENT_CONDITIONS_DISPLAY_H__

#include "IDisplayElement.h"
#include "IObserver.h"
#include "ISubject.h"

#include <memory>

class CurrentConditionsDisplay : public IObserver, IDisplayElement {
public:
    virtual ~CurrentConditionsDisplay() { detach(); }

    void attach(std::shared_ptr<ISubject> subject) final;
    void detach() final;
    void update(float temp, float humidity, float pressure) final;
    void display() const final;

private:
    float temperature;
    float humidity;

    std::shared_ptr<ISubject> mSubjectImpl;
};

#endif  // __CURRENT_CONDITIONS_DISPLAY_H__