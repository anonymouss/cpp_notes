#ifndef __HOUSE_BLEND_H__
#define __HOUSE_BLEND_H__

#include "IBeverage.h"

class HouseBlend : public IBeverage {
public:
    HouseBlend();
    virtual ~HouseBlend() = default;

    double calculateCost() final;
};

#endif  // __HOUSE_BLEND_H__