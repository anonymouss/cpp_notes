#ifndef __DECOY_DUCK_H__
#define __DECOY_DUCK_H__

#include "DuckBase.h"

class DecoyDuck : public DuckBase {
public:
    DecoyDuck(const char *name = "decoy duck");
    virtual ~DecoyDuck() = default;
};

#endif  // __DECOY_DUCK_H__