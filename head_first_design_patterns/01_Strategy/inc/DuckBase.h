#ifndef __DUCK_BASE_H__
#define __DUCK_BASE_H__

#include "IBehaviors.h"

#include <memory>
#include <string>

class DuckBase {
public:
    explicit DuckBase(const char *name = "") : mName(name) {}
    virtual ~DuckBase() = default;

    virtual void performQuack();
    virtual void performFly();

protected:
    std::string mName;
    std::unique_ptr<IFlyBehavior> pFlyBehavior;
    std::unique_ptr<IQuackBehavior> pQuackBehavior;

    virtual void setFlyBehavior(std::unique_ptr<IFlyBehavior> flyBehavior);
    virtual void setQuackBehavior(std::unique_ptr<IQuackBehavior> quackBehavior);
};

#endif  // __DUCK_BASE_H__