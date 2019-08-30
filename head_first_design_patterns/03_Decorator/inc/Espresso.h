#ifndef __ESPRESSO_H__
#define __ESPRESSO_H__

#include "IBeverage.h"

class Espresso : public IBeverage {
public:
    Espresso();
    virtual ~Espresso() = default;

    double calculateCost() final;
};

#endif  // __ESPRESSO_H__