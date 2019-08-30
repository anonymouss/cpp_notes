#include "ChicagoPizzaStore.h"

#include "ChicagoStyleCheesePizza.h"
#include "ChicagoStyleClamPizza.h"

#include <iostream>

std::unique_ptr<IPizza> ChicagoPizzaStore::orderPizza(PizzaType type) {
    std::cout << "--- Chicago Pizza Store --- is at your service" << std::endl;
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

std::unique_ptr<IPizza> ChicagoPizzaStore::createPizza_l(PizzaType type) {
    std::unique_ptr<IPizza> pizza = nullptr;

    switch (type) {
        case PizzaType::CHICAGO_CHEESE: {
            pizza = std::make_unique<ChicagoStyleCheesePizza>();
            break;
        }
        case PizzaType::CHICAGO_CLAM: {
            pizza = std::make_unique<ChicagoStyleClamPizza>();
            break;
        }
        default: {
            std::cout << "sorry, I don't know how to make this kind of pizza" << std::endl;
            break;
        }
    }

    return pizza;
}