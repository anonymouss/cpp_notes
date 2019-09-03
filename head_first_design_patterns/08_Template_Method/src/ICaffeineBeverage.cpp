#include "ICaffeineBeverage.h"

#include <iostream>

void ICaffeineBeverage::prepareRecipe() {
    boilWater();
    brew();
    pourInCup();
    if (customerWantsCondiments()) { addCondiments(); }
    std::cout << "done!" << std::endl;
}

void ICaffeineBeverage::boilWater() { std::cout << "Boiling water...." << std::endl; }

void ICaffeineBeverage::pourInCup() { std::cout << "Pouring in cup..." << std::endl; }

bool ICaffeineBeverage::customerWantsCondiments() { return true; }