#ifndef __RUBBER_DUCK_H__
#define __RUBBER_DUCK_H__

#include "DuckBase.h"

class RubberDuck : public DuckBase {
public:
    RubberDuck(const char *name = "rubber duck");
    virtual ~RubberDuck() = default;
};

#endif  // __RUBBER_DUCK_H__