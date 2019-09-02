#include "NewYorkPizzaStore.h"

#include <iostream>

int main() {
    NewYorkPizzaStore nyStore;

    std::cout << "----- order clam pizza ----\n";
    auto clamPizza = nyStore.orderPizza(PizzaType::CLAM);
    std::cout << "---- order cheese pizza ---\n";
    auto chezPizza = nyStore.orderPizza(PizzaType::CHEESE);
    std::cout << "---- order other pizza ----\n";
    auto othzPizza = nyStore.orderPizza(PizzaType::OTHERS);
}

/**
 * output
 * ----- order clam pizza ----
 * WELCOME TO NEW YORK PIZZA STORE
 * we recieved your order, please wait...
 *  preparing clam pizza
 *   making pizza: <clam pizza>
 *         adding some <thin crust dough>
 *         adding some <marinara sauce>
 *         adding some <fresh clams>
 * your pizza is ready.
 * ---- order cheese pizza ---
 * WELCOME TO NEW YORK PIZZA STORE
 * we recieved your order, please wait...
 *  preparing cheese pizza
 *   making pizza: <cheese pizza>
 *         adding some <thin crust dough>
 *         adding some <regginao cheese>
 *         adding some <marinara sauce>
 * your pizza is ready.
 * ---- order other pizza ----
 * WELCOME TO NEW YORK PIZZA STORE
 * we recieved your order, please wait...
 * sorry, we don't know how to make this kind of pizza
 * sorry, we can't make your pizza
 */