#include "MallardDuck.h"
#include "FlyWithWings.h"
#include "Quack.h"

#include <memory>

MallardDuck::MallardDuck(const char *name) : DuckBase(name) {
    setFlyBehavior(std::make_unique<FlyWithWings>());
    setQuackBehavior(std::make_unique<Quack>());
}