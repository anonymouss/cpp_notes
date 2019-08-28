#include "DecoyDuck.h"
#include "FlyNoWay.h"
#include "MuteQuack.h"

#include <memory>

DecoyDuck::DecoyDuck(const char *name) : DuckBase(name) {
    setFlyBehavior(std::make_unique<FlyNoWay>());
    setQuackBehavior(std::make_unique<MuteQuack>());
}