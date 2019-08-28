#ifndef __RED_HEAD_DUCK_H__
#define __RED_HEAD_DUCK_H__

#include "DuckBase.h"

class RedheadDuck : public DuckBase {
public:
    RedheadDuck(const char *name = "red head duck");
    virtual ~RedheadDuck() = default;
};

#endif  // __RED_HEAD_DUCK_H__