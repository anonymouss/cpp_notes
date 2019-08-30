#include "PizzaStore.h"

#include <iostream>

std::unique_ptr<IPizza> PizzaStore::orderPizza(PizzaType pizzaType, const char *desc) {
    std::cout << "STORE: sending order to our factory... please wait..." << std::endl;

    auto pizza = pizzaFactory.createPizza(pizzaType, desc);

    if (pizza) {
        std::cout << "STORE: got pizza from fatory, will bake it..." << std::endl;
        pizza->prepare();
        pizza->bake();
        pizza->cut();
        pizza->box();
        std::cout << "STORE: here is your pizza\n" << std::endl;
    } else {
        std::cout << "STORE: sorry, our factory has not such pizza\n" << std::endl;
    }

    return pizza;
}