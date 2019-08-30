#include "DarkRoast.h"
#include "Decaf.h"
#include "Espresso.h"
#include "HouseBlend.h"

#include "Milk.h"
#include "Mocha.h"
#include "Soy.h"
#include "Whip.h"

#include <iostream>
#include <memory>

template <typename Coffee>
void coffeeReady(const Coffee& coffee) {
    std::cout << "<< Your coffee is ready >>" << std::endl;
    std::cout << " : Type  : " << coffee->getDescription() << std::endl;
    std::cout << " : Price : $" << coffee->calculateCost() << std::endl;
    std::cout << "--------------------------" << std::endl;
}

int main() {
    auto darkRoast = std::make_shared<DarkRoast>();
    auto decaf = std::make_shared<Decaf>();
    auto espresso = std::make_shared<Espresso>();
    auto houseBlend = std::make_shared<HouseBlend>();

    std::cout << "I want to have a cup of dark roast coffee, please add some mik and whip.\n";
    auto order1 = std::make_shared<Whip>(std::make_shared<Milk>(darkRoast));
    coffeeReady(order1);

    std::cout << "A cup of espresso, add some soy and mocha please.\n";
    auto order2 = std::make_shared<Mocha>(std::make_shared<Soy>(espresso));
    coffeeReady(order2);
}

/**
 * outputs
 * I want to have a cup of dark roast coffee, please add some mik and whip.
 * << Your coffee is ready >>
 *  : Type  : DarkRoast coffee, add some milk, add some whip
 *  : Price : $29.12
 * --------------------------
 * A cup of espresso, add some soy and mocha please.
 * << Your coffee is ready >>
 *  : Type  : Espresso coffee, add some Soy, add some mocha
 *  : Price : $39.4
 * --------------------------
 */