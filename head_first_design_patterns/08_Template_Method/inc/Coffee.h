#ifndef __COFEE_H__
#define __COFEE_H__

#include "ICaffeineBeverage.h"

class Coffee : public ICaffeineBeverage {
public:
    virtual ~Coffee() = default;

    virtual void brew() final;
    virtual void addCondiments() final;

    virtual bool customerWantsCondiments() final;
};

#endif  // __COFEE_H__