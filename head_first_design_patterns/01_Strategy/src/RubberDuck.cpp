#include "RubberDuck.h"
#include "FlyNoWay.h"
#include "Squeak.h"

#include <memory>

RubberDuck::RubberDuck(const char *name) : DuckBase(name) {
    setFlyBehavior(std::make_unique<FlyNoWay>());
    setQuackBehavior(std::make_unique<Squeak>());
}