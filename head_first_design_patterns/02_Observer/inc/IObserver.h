#ifndef __I_OBSERVER_H__
#define __I_OBSERVER_H__

#include <memory>

class ISubject;

class IObserver {
public:
    virtual ~IObserver() = default;

    virtual void attach(std::shared_ptr<ISubject> subject) = 0;
    virtual void detach() = 0;
    virtual void update(float temp, float humidity, float pressure) = 0;
};

#endif  // __I_OBSERVER_H__