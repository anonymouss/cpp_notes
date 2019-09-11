#include "DinnerMenu.h"

#include <iostream>

int main() {
    DinnerMenu menu;
    auto *it = menu.createIterator();

    if (it) {
        while (it->hasNext()) {
            auto *e = it->next();
            std::cout << "[NAME]: " << e->getName() << ", [DESCRIPTION]: " << e->getDescription()
                      << ", [VEGETARIAN]: " << e->isVegetarian() << ", [PRICE]: " << e->getPrice()
                      << std::endl;
        }
    }
}

/**
 * output
 * [NAME]: item 0, [DESCRIPTION]: , [VEGETARIAN]: 0, [PRICE]: 0.1
 * [NAME]: item 1, [DESCRIPTION]: , [VEGETARIAN]: 1, [PRICE]: 1.2
 * [NAME]: item 2, [DESCRIPTION]: , [VEGETARIAN]: 0, [PRICE]: 4.3
 * [NAME]: item 3, [DESCRIPTION]: , [VEGETARIAN]: 1, [PRICE]: 9.4
 * [NAME]: item 4, [DESCRIPTION]: , [VEGETARIAN]: 0, [PRICE]: 16.5
 * [NAME]: item 5, [DESCRIPTION]: , [VEGETARIAN]: 1, [PRICE]: 25.6
 */