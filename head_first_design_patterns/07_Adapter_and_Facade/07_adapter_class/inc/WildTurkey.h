#ifndef __WILD_TURKEY_H__
#define __WILD_TURKEY_H__

#include "ITurkey.h"

class WildTurkey : public ITurkey {
public:
    virtual ~WildTurkey() = default;
    virtual void gobble() override;
    virtual void fly() override;
};

#endif  // __WILD_TURKEY_H__