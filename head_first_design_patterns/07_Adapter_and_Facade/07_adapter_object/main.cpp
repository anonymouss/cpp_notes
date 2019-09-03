#include "IDuck.h"
#include "TurkeyAdapter.h"
#include "WildTurkey.h"

#include <memory>

int main() {
    std::shared_ptr<IDuck> duck;
    auto turkey = std::make_shared<WildTurkey>();

    duck = std::make_shared<TurkeyAdapter>(turkey);

    duck->quack();
    duck->fly();
}

/**
 * output
 * WildTurkey: gobble
 * WildTurkey: fly
 */