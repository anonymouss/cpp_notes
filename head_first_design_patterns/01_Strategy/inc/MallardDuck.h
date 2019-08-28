#ifndef __MALLARD_DUCK_H__
#define __MALLARD_DUCK_H__

#include "DuckBase.h"

class MallardDuck : public DuckBase {
public:
    MallardDuck(const char *name = "mallard duck");
    virtual ~MallardDuck() = default;
};

#endif  // __MALLARD_DUCK_H__