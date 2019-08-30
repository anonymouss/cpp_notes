#ifndef __I_CONDIMENT_DECORATOR_H__
#define __I_CONDIMENT_DECORATOR_H__

#include "IBeverage.h"

#include <string>

class ICondimentDecorator : public IBeverage {
public:
    virtual std::string getDescription() const override = 0;
    virtual double calculateCost() override = 0;

    virtual ~ICondimentDecorator() = default;
};

#endif  // __I_CONDIMENT_DECORATOR_H__