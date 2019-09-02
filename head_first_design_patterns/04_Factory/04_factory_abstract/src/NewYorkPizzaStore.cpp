#include "NewYorkPizzaStore.h"
#include "CheesePizza.h"
#include "ClamPizza.h"

#include <iostream>

std::unique_ptr<IPizza> NewYorkPizzaStore::orderPizza(PizzaType type) {
    std::cout << "WELCOME TO NEW YORK PIZZA STORE" << std::endl;
    std::cout << "we recieved your order, please wait..." << std::endl;
    auto pizza = createPizza_l(type);
    if (pizza) {
        pizza->prepare();
        pizza->make();
        std::cout << "your pizza is ready." << std::endl;
    } else {
        std::cout << "sorry, we can't make your pizza" << std::endl;
    }
    return pizza;
}

std::unique_ptr<IPizza> NewYorkPizzaStore::createPizza_l(PizzaType type) {
    std::unique_ptr<IPizza> pizza = nullptr;
    switch (type) {
        case PizzaType::CHEESE: {
            pizza = std::make_unique<CheesePizza>(mIngredientFactory);
            break;
        }
        case PizzaType::CLAM: {
            pizza = std::make_unique<ClamPizza>(mIngredientFactory);
            break;
        }
        default: {
            std::cout << "sorry, we don't know how to make this kind of pizza" << std::endl;
            break;
        }
    }
    return pizza;
}