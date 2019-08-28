#ifndef __MUTE_QUACK_H__
#define __MUTE_QUACK_H__

#include "IBehaviors.h"

class MuteQuack : public IQuackBehavior {
public:
    void quack() final;
    virtual ~MuteQuack() = default;
};

#endif  // __MUTE_QUACK_H__