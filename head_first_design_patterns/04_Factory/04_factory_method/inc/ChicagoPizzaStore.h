#ifndef __CHICAGO_PIZZA_STORE_H__
#define __CHICAGO_PIZZA_STORE_H__

#include "IPizzaStore.h"

#include <memory>

class ChicagoPizzaStore : public IPizzaStore {
public:
    virtual ~ChicagoPizzaStore() = default;

    virtual std::unique_ptr<IPizza> orderPizza(PizzaType type) final;

protected:
    virtual std::unique_ptr<IPizza> createPizza_l(PizzaType type) final;
};

#endif  // __CHICAGO_PIZZA_STORE_H__