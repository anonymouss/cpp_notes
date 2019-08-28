#ifndef __QUACK_H__
#define __QUACK_H__

#include "IBehaviors.h"

class Quack : public IQuackBehavior {
public:
    void quack() final;
    virtual ~Quack() = default;
};

#endif  // __QUACK_H__