#ifndef __FLY_WITH_WINGS_H__
#define __FLY_WITH_WINGS_H__

#include "IBehaviors.h"

class FlyWithWings : public IFlyBehavior {
public:
    void fly() final;
    virtual ~FlyWithWings() = default;
};

#endif  //__FLY_WITH_WINGS_H__