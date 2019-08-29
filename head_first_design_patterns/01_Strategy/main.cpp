#include "DecoyDuck.h"
#include "DuckBase.h"
#include "MallardDuck.h"
#include "RedheadDuck.h"
#include "RubberDuck.h"

#include <iostream>
#include <memory>
#include <vector>

int main() {
    std::vector<std::unique_ptr<DuckBase>> ducks;
    ducks.emplace_back(std::make_unique<DuckBase>());
    ducks.emplace_back(std::make_unique<MallardDuck>());
    ducks.emplace_back(std::make_unique<DecoyDuck>());
    ducks.emplace_back(std::make_unique<RedheadDuck>());
    ducks.emplace_back(std::make_unique<RubberDuck>());

    std::cout << "Let's see the ducks...\n";
    for (const auto &duck : ducks) {
        duck->performQuack();
        duck->performFly();
        std::cout << std::endl;  // blank line
    }
}

/**
 * outputs
 * Let's see the ducks...
 * []
 * !!! No fly behavior registered
 * []
 * !!! No quack behavior registered
 *
 * [mallard duck]
 * I am flying with my wings...
 * [mallard duck]
 * Quack! Quack! Quack!
 *
 * [decoy duck]
 * << Sorry, I can't fly... >>
 * [decoy duck]
 * << Sorry, I am dumb... >>
 *
 * [red head duck]
 * << Sorry, I can't fly... >>
 * [red head duck]
 * Quack! Quack! Quack!
 *
 * [rubber duck]
 * << Sorry, I can't fly... >>
 * [rubber duck]
 * Squeak! Squeak! Squeak!
 * */