#ifndef __SIMPLE_DUCK_H__
#define __SIMPLE_DUCK_H__

#include "IDuck.h"

class SimpleDuck : public IDuck {
public:
    virtual ~SimpleDuck() = default;
    virtual void quack() override;
    virtual void fly() override;
};

#endif  // __SIMPLE_DUCK_H__