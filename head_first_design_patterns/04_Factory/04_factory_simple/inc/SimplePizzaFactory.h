#ifndef __SIMPLE_PIZZA_FACTORY_H__
#define __SIMPLE_PIZZA_FACTORY_H__

#include "IPizza.h"

#include <memory>

enum class PizzaType {
    CHEESE_PIZZA,
    VEGGIE_PIZZA,
    CLAM_PIZZA,
    PEPPERON_PIZZA,
    UNKNOWN_PIZZA,
};

class SimplePizzaFactory {
public:
    std::unique_ptr<IPizza> createPizza(PizzaType type, const char *desc = "");
};

#endif  // __SIMPLE_PIZZA_FACTORY_H__