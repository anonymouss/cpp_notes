#ifndef __WHIP_H__
#define __WHIP_H__

#include "IBeverage.h"
#include "ICondimentDecorator.h"

#include <memory>

class Whip : public ICondimentDecorator {
public:
    explicit Whip(std::shared_ptr<IBeverage> bevarage);
    virtual ~Whip() = default;

    virtual std::string getDescription() const final;
    virtual double calculateCost() final;

    std::shared_ptr<IBeverage> mBevarage;
};

#endif  // __WHIP_H__