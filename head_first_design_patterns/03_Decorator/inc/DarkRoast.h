#ifndef __DARK_ROAST_H__
#define __DARK_ROAST_H__

#include "IBeverage.h"

class DarkRoast : public IBeverage {
public:
    DarkRoast();
    virtual ~DarkRoast() = default;

    double calculateCost() final;
};

#endif  // __DARK_ROAST_H__