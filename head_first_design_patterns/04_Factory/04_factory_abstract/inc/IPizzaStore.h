#ifndef __I_PIZZA_STORE_H__
#define __I_PIZZA_STORE_H__

#include "IPizza.h"

#include <memory>

enum class PizzaType {
    CLAM,
    CHEESE,
    OTHERS,
};

class IPizzaStore {
public:
    virtual ~IPizzaStore() = default;

    virtual std::unique_ptr<IPizza> orderPizza(PizzaType type) = 0;

protected:
    virtual std::unique_ptr<IPizza> createPizza_l(PizzaType type) = 0;
};

#endif  // __I_PIZZA_STORE_H__