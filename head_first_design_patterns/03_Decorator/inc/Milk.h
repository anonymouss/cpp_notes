#ifndef __MILK_H__
#define __MILK_H__

#include "IBeverage.h"
#include "ICondimentDecorator.h"

#include <memory>

class Milk : public ICondimentDecorator {
public:
    explicit Milk(std::shared_ptr<IBeverage> bevarage);
    virtual ~Milk() = default;

    virtual std::string getDescription() const final;
    virtual double calculateCost() final;

    std::shared_ptr<IBeverage> mBevarage;
};

#endif  // __MILK_H__