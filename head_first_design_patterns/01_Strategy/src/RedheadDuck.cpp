#include "RedheadDuck.h"
#include "FlyNoWay.h"
#include "Quack.h"

#include <memory>

RedheadDuck::RedheadDuck(const char *name) : DuckBase(name) {
    setFlyBehavior(std::make_unique<FlyNoWay>());
    setQuackBehavior(std::make_unique<Quack>());
}