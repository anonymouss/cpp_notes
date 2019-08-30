#include "NewYorkPizzaStore.h"

#include "NewYorkStyleCheesePizza.h"
#include "NewYorkStyleClamPizza.h"

#include <iostream>

std::unique_ptr<IPizza> NewYorkPizzaStore::orderPizza(PizzaType type) {
    std::cout << "--- New York Pizza Store --- is at your service" << std::endl;
    std::cout << "we received your order, please wait..." << std::endl;

    auto pizza = createPizza_l(type);

    if (pizza) {
        std::cout << "making..." << std::endl;
        pizza->prepare();
        pizza->bake();
        pizza->cut();
        pizza->box();
        std::cout << "please enjoy your pizza\n" << std::endl;
    } else {
        std::cout << "sorry, we don't provide this pizza\n" << std::endl;
    }
    return pizza;
}

std::unique_ptr<IPizza> NewYorkPizzaStore::createPizza_l(PizzaType type) {
    std::unique_ptr<IPizza> pizza = nullptr;

    switch (type) {
        case PizzaType::NEWYORK_CHEESE: {
            pizza = std::make_unique<NewYorkStyleCheesePizza>();
            break;
        }
        case PizzaType::NEWYORK_CLAM: {
            pizza = std::make_unique<NewYorkStyleClamPizza>();
            break;
        }
        default: {
            std::cout << "sorry, I don't know how to make this kind of pizza" << std::endl;
            break;
        }
    }

    return pizza;
}