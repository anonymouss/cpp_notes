#ifndef __NEWYORK_PIZZA_STORE_H__
#define __NEWYORK_PIZZA_STORE_H__

#include "IPizzaStore.h"
#include "NewYorkPizzaIngredientFactory.h"

#include <memory>

class NewYorkPizzaStore : public IPizzaStore {
public:
    NewYorkPizzaStore() { mIngredientFactory = std::make_shared<NewYorkPizzaIngredientFactory>(); }
    virtual ~NewYorkPizzaStore() = default;

    virtual std::unique_ptr<IPizza> orderPizza(PizzaType type) final;

protected:
    virtual std::unique_ptr<IPizza> createPizza_l(PizzaType type) final;

private:
    std::shared_ptr<NewYorkPizzaIngredientFactory> mIngredientFactory;
};

#endif  // __NEWYORK_PIZZA_STORE_H__