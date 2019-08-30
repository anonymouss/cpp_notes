#include "NewYorkPizzaStore.h"
#include "NewYorkStyleCheesePizza.h"
#include "NewYorkStyleClamPizza.h"

#include "ChicagoPizzaStore.h"
#include "ChicagoStyleCheesePizza.h"
#include "ChicagoStyleClamPizza.h"

#include <iostream>
#include <memory>

int main() {
    NewYorkPizzaStore nyStore;
    ChicagoPizzaStore ccStore;

    std::cout << "coming into new york pizza store\n" << std::endl;
    auto npizza1 = nyStore.orderPizza(PizzaType::NEWYORK_CHEESE);
    auto npizza2 = nyStore.orderPizza(PizzaType::NEWYORK_CLAM);
    auto npizza3 = nyStore.orderPizza(PizzaType::CHICAGO_CHEESE);

    std::cout << "coming into chicago pizza store\n" << std::endl;
    auto cpizza1 = ccStore.orderPizza(PizzaType::CHICAGO_CHEESE);
    auto cpizza2 = ccStore.orderPizza(PizzaType::CHICAGO_CLAM);
    auto cpizza3 = ccStore.orderPizza(PizzaType::NEWYORK_CHEESE);

    auto check_null = [](const auto &obj, bool expected) {
        if ((obj == nullptr) == expected) {
            std::cout << "OK" << std::endl;
        } else {
            std::cout << "FAIL" << std::endl;
        }
    };

    check_null(npizza1, false);
    check_null(npizza2, false);
    check_null(npizza3, true);

    check_null(cpizza1, false);
    check_null(cpizza2, false);
    check_null(cpizza3, true);
}

/**
 * output
 * coming into new york pizza store
 * --- New York Pizza Store --- is at your service
 * we received your order, please wait...
 * making...
 * >> Preparing [ New York Style Cheese Pizza ]
 * >> Baking [ New York Style Cheese Pizza ]
 * >> Cutting [ New York Style Cheese Pizza ]
 * >> Boxing [ New York Style Cheese Pizza ]
 * please enjoy your pizza
 *
 * --- New York Pizza Store --- is at your service
 * we received your order, please wait...
 * making...
 * >> Preparing [ New York Style Clam Pizza ]
 * >> Baking [ New York Style Clam Pizza ]
 * >> Cutting [ New York Style Clam Pizza ]
 * >> Boxing [ New York Style Clam Pizza ]
 * please enjoy your pizza
 *
 * --- New York Pizza Store --- is at your service
 * we received your order, please wait...
 * sorry, I don't know how to make this kind of pizza
 * sorry, we don't provide this pizza
 *
 * coming into chicago pizza store
 *
 * --- Chicago Pizza Store --- is at your service
 * we received your order, please wait...
 * making...
 * >> Preparing [ Chicago Style Cheese Pizza ]
 * >> Baking [ Chicago Style Cheese Pizza ]
 * >> Cutting [ Chicago Style Cheese Pizza ]
 * >> Boxing [ Chicago Style Cheese Pizza ]
 * please enjoy your pizza
 *
 * --- Chicago Pizza Store --- is at your service
 * we received your order, please wait...
 * making...
 * >> Preparing [ Chicago Style Clam Pizza ]
 * >> Baking [ Chicago Style Clam Pizza ]
 * >> Cutting [ Chicago Style Clam Pizza ]
 * >> Boxing [ Chicago Style Clam Pizza ]
 * please enjoy your pizza
 *
 * --- Chicago Pizza Store --- is at your service
 * we received your order, please wait...
 * sorry, I don't know how to make this kind of pizza
 * sorry, we don't provide this pizza
 */