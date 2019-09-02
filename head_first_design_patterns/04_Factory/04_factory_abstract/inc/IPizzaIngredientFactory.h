#ifndef __I_PIZZA_INGREDIENT_FACTORY_H__
#define __I_PIZZA_INGREDIENT_FACTORY_H__

#include <memory>

class ICheese;
class IClams;
class IDough;
class ISauce;

class IPizzaIngredientFactory {
public:
    virtual ~IPizzaIngredientFactory() = default;

    virtual std::unique_ptr<ICheese> createCheese() = 0;
    virtual std::unique_ptr<IClams> createClams() = 0;
    virtual std::unique_ptr<IDough> createDough() = 0;
    virtual std::unique_ptr<ISauce> createSauce() = 0;
};

#endif  // __I_PIZZA_INGREDIENT_FACTORY_H__