#ifndef __SQUEAK_H__
#define __SQUEAK_H__

#include "IBehaviors.h"

class Squeak : public IQuackBehavior {
public:
    void quack() final;
    virtual ~Squeak() = default;
};

#endif  // __SQUEAK_H__