#ifndef __FLY_NO_WAY_H__
#define __FLY_NO_WAY_H__

#include "IBehaviors.h"

class FlyNoWay : public IFlyBehavior {
public:
    void fly() final;
    virtual ~FlyNoWay() = default;
};

#endif  //__FLY_NO_WAY_H__