#ifndef __I_STATE_H__
#define __I_STATE_H__

#include <string>

struct IState {
    // explicit IState(const char *name) : mName(name) {}
    virtual ~IState() = default;

    virtual void insertQuarter() = 0;
    virtual void ejectQuarter() = 0;
    virtual void turnCrank() = 0;
    virtual void dispense() = 0;
};

#endif  // __I_STATE_H__