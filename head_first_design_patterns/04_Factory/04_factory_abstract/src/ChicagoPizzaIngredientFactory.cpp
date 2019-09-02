#include "ChicagoPizzaIngredientFactory.h"

#include "FrozenClams.h"
#include "MozzarellaCheese.h"
#include "PlumTomatoSauce.h"
#include "ThickCrustDough.h"

std::unique_ptr<ICheese> ChicagoPizzaIngredientFactory::createCheese() {
    return std::make_unique<MozzarellaCheese>();
}

std::unique_ptr<IClams> ChicagoPizzaIngredientFactory::createClams() {
    return std::make_unique<FrozenClams>();
}

std::unique_ptr<IDough> ChicagoPizzaIngredientFactory::createDough() {
    return std::make_unique<ThickCrustDough>();
}

std::unique_ptr<ISauce> ChicagoPizzaIngredientFactory::createSauce() {
    return std::make_unique<PlumTomatoSauce>();
}