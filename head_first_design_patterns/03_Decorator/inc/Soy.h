#ifndef __SOY_H__
#define __SOY_H__

#include "IBeverage.h"
#include "ICondimentDecorator.h"

#include <memory>

class Soy : public ICondimentDecorator {
public:
    explicit Soy(std::shared_ptr<IBeverage> bevarage);
    virtual ~Soy() = default;

    virtual std::string getDescription() const final;
    virtual double calculateCost() final;

    std::shared_ptr<IBeverage> mBevarage;
};

#endif  // __SOY_H__