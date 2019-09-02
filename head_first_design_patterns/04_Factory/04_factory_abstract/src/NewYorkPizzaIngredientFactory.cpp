#include "NewYorkPizzaIngredientFactory.h"

#include "FreshClams.h"
#include "MarinaraSauce.h"
#include "RegginaoCheese.h"
#include "ThinCrustDough.h"

std::unique_ptr<ICheese> NewYorkPizzaIngredientFactory::createCheese() {
    return std::make_unique<RegginaoCheese>();
}

std::unique_ptr<IClams> NewYorkPizzaIngredientFactory::createClams() {
    return std::make_unique<FreshClams>();
}

std::unique_ptr<IDough> NewYorkPizzaIngredientFactory::createDough() {
    return std::make_unique<ThinCrustDough>();
}

std::unique_ptr<ISauce> NewYorkPizzaIngredientFactory::createSauce() {
    return std::make_unique<MarinaraSauce>();
}