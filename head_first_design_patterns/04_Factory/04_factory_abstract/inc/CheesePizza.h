#ifndef __CHEESE_PIZZA_H__
#define __CHEESE_PIZZA_H__

#include "IPizza.h"
#include "IPizzaIngredientFactory.h"

#include <memory>

class CheesePizza : public IPizza {
public:
    explicit CheesePizza(std::shared_ptr<IPizzaIngredientFactory> gredientFactory)
        : mGredientFactory(gredientFactory) {
        mName = "<cheese pizza>";
    }
    virtual void prepare() final;

private:
    std::shared_ptr<IPizzaIngredientFactory> mGredientFactory;
};

#endif  // __CHEESE_PIZZA_H__