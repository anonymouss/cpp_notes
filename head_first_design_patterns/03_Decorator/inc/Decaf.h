#ifndef __DECAF_H__
#define __DECAF_H__

#include "IBeverage.h"

class Decaf : public IBeverage {
public:
    Decaf();
    virtual ~Decaf() = default;

    double calculateCost() final;
};

#endif  // __DECAF_H__