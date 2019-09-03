#ifndef __TURKEY_ADAPTER_H__
#define __TURKEY_ADAPTER_H__

#include "SimpleDuck.h"
#include "WildTurkey.h"

#include <memory>

class Adapter : public SimpleDuck, WildTurkey {
public:
    virtual ~Adapter() = default;

    virtual void quack() final { gobble(); }
    virtual void fly() final { WildTurkey::fly(); }
};

#endif  // __TURKEY_ADAPTER_H__