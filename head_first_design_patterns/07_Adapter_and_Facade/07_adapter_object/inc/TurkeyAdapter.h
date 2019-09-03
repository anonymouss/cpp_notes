#ifndef __TURKEY_ADAPTER_H__
#define __TURKEY_ADAPTER_H__

#include "IDuck.h"
#include "ITurkey.h"

#include <memory>

class TurkeyAdapter : public IDuck {
public:
    virtual ~TurkeyAdapter() = default;
    explicit TurkeyAdapter(std::shared_ptr<ITurkey> turkey) : mTurkey(turkey) {}

    virtual void quack() final;
    virtual void fly() final;

private:
    std::shared_ptr<ITurkey> mTurkey;
};

#endif  // __TURKEY_ADAPTER_H__