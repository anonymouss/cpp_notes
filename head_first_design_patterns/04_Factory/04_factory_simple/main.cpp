#include "PizzaStore.h"

#include <iostream>

template <typename T>
void check_null(const T &t, bool expected) {
    if ((t == nullptr) == expected) {
        std::cout << "OK" << std::endl;
    } else {
        std::cout << "FAIL" << std::endl;
    }
}

int main() {
    PizzaStore store;

    auto pizza1 = store.orderPizza(PizzaType::CHEESE_PIZZA, "cheese 1");
    auto pizza2 = store.orderPizza(PizzaType::PEPPERON_PIZZA, "pepperon 1");
    auto pizza3 = store.orderPizza(PizzaType::UNKNOWN_PIZZA, "?");

    check_null(pizza1, false);
    check_null(pizza2, false);
    check_null(pizza3, true);
}

/**
 * output
 * STORE: sending order to our factory... please wait...
 * FACTORY: got order, one cheese pizza!
 * STORE: got pizza from fatory, will bake it...
 * >> Preparing [Cheese Pizza] - cheese 1
 * >> Baking [Cheese Pizza] - cheese 1
 * >> Cutting [Cheese Pizza] - cheese 1
 * >> Boxing [Cheese Pizza] - cheese 1
 * STORE: here is your pizza
 *
 * STORE: sending order to our factory... please wait...
 * FACTORY: got order, one pepperon pizza!
 * STORE: got pizza from fatory, will bake it...
 * >> Preparing [Pepperon Pizza] - pepperon 1
 * >> Baking [Pepperon Pizza] - pepperon 1
 * >> Cutting [Pepperon Pizza] - pepperon 1
 * >> Boxing [Pepperon Pizza] - pepperon 1
 * STORE: here is your pizza
 *
 * STORE: sending order to our factory... please wait...
 * FACTORY: got order, but we don't know how to make it!
 * STORE: sorry, our factory has not such pizza
 *
 * OK
 * OK
 * OK
 */