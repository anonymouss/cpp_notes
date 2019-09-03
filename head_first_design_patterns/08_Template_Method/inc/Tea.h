#ifndef __TEA_H__
#define __TEA_H__

#include "ICaffeineBeverage.h"

#include <iostream>

class Tea : public ICaffeineBeverage {
public:
    virtual ~Tea() = default;

    virtual void brew() final;
    virtual void addCondiments() final;

    virtual bool customerWantsCondiments() final;
};

#endif  // __TEA_H__