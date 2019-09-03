#ifndef __WILD_TURKEY_H__
#define __WILD_TURKEY_H__

#include "ITurkey.h"

class WildTurkey : public ITurkey {
public:
    virtual ~WildTurkey() = default;
    virtual void gobble() final;
    virtual void fly() final;
};

#endif  // __WILD_TURKEY_H__