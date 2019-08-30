#ifndef __MOCHA_H__
#define __MOCHA_H__

#include "IBeverage.h"
#include "ICondimentDecorator.h"

#include <memory>

class Mocha : public ICondimentDecorator {
public:
    explicit Mocha(std::shared_ptr<IBeverage> bevarage);
    virtual ~Mocha() = default;

    virtual std::string getDescription() const final;
    virtual double calculateCost() final;

    std::shared_ptr<IBeverage> mBevarage;
};

#endif  // __MOCHA_H__