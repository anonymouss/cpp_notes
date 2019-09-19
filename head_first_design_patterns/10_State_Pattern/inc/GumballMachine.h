#ifndef __GUMBALL_MACHINE_H__
#define __GUMBALL_MACHINE_H__

#include "IState.h"

#include <memory>

class GumballMachine : public IState {
    struct BaseState;
    struct NoQuarterState;
    struct HasQuarterState;
    struct SoldState;
    struct SoldOutState;

public:
    virtual ~GumballMachine() = default;
    GumballMachine();
    void setGumballCount(int count);
    void changeState(std::shared_ptr<BaseState> state);
    void releaseGumball();
    bool isEmpty() const;
    virtual void insertQuarter() final;
    virtual void ejectQuarter() final;
    virtual void turnCrank() final;
    virtual void dispense() final;

    int mGumballCounts;

    std::shared_ptr<BaseState> mState;

    std::shared_ptr<BaseState> mNoQuarterState;
    std::shared_ptr<BaseState> mHasQuarterState;
    std::shared_ptr<BaseState> mSoldState;
    std::shared_ptr<BaseState> mSoldOutState;
};

#endif  // __GUMBALL_MACHINE_H__