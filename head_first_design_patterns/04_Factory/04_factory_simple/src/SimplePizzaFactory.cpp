#include "SimplePizzaFactory.h"

#include "CheesePizza.h"
#include "ClamPizza.h"
#include "PepperonPizza.h"
#include "VeggiePizza.h"

#include <iostream>

std::unique_ptr<IPizza> SimplePizzaFactory::createPizza(PizzaType type, const char *desc) {
    std::unique_ptr<IPizza> pizza = nullptr;

    switch (type) {
        case PizzaType::CHEESE_PIZZA: {
            std::cout << "FACTORY: got order, one cheese pizza!" << std::endl;
            pizza = std::make_unique<CheesePizza>(desc);
            break;
        }
        case PizzaType::VEGGIE_PIZZA: {
            std::cout << "FACTORY: got order, one veggie pizza!" << std::endl;
            pizza = std::make_unique<VeggiePizza>(desc);
            break;
        }
        case PizzaType::CLAM_PIZZA: {
            std::cout << "FACTORY: got order, one clam pizza!" << std::endl;
            pizza = std::make_unique<ClamPizza>(desc);
            break;
        }
        case PizzaType::PEPPERON_PIZZA: {
            std::cout << "FACTORY: got order, one pepperon pizza!" << std::endl;
            pizza = std::make_unique<PepperonPizza>(desc);
            break;
        }
        default: {
            std::cout << "FACTORY: got order, but we don't know how to make it!" << std::endl;
            break;
        }
    }

    return pizza;
}