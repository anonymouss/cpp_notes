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