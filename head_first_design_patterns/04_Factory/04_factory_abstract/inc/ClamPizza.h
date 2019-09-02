#ifndef __CLAM_PIZZA_H__
#define __CLAM_PIZZA_H__

#include "IPizza.h"
#include "IPizzaIngredientFactory.h"

#include <memory>

class ClamPizza : public IPizza {
public:
    explicit ClamPizza(std::shared_ptr<IPizzaIngredientFactory> gredientFactory)
        : mGredientFactory(gredientFactory) {
        mName = "<clam pizza>";
    }
    virtual void prepare() final;

private:
    std::shared_ptr<IPizzaIngredientFactory> mGredientFactory;
};

#endif  // __CLAM_PIZZA_H__