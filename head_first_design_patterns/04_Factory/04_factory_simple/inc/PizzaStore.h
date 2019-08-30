#ifndef __PIZZA_STORE_H__
#define __PIZZA_STORE_H__

#include "SimplePizzaFactory.h"

#include <memory>

class PizzaStore {
public:
    std::unique_ptr<IPizza> orderPizza(PizzaType pizzaType, const char *desc);

private:
    SimplePizzaFactory pizzaFactory;
};

#endif  // __PIZZA_STORE_H__