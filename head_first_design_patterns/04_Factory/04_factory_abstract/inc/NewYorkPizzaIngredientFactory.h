#ifndef __NEWYORK_PIZZA_INGREDIENT_FACTORY_H__
#define __NEWYORK_PIZZA_INGREDIENT_FACTORY_H__

#include "IPizzaIngredientFactory.h"

class NewYorkPizzaIngredientFactory : public IPizzaIngredientFactory {
public:
    virtual std::unique_ptr<ICheese> createCheese() final;
    virtual std::unique_ptr<IClams> createClams() final;
    virtual std::unique_ptr<IDough> createDough() final;
    virtual std::unique_ptr<ISauce> createSauce() final;
};

#endif  // __NEWYORK_PIZZA_INGREDIENT_FACTORY_H__