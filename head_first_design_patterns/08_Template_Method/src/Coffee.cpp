#include "Coffee.h"

#include <iostream>
#include <string>

void Coffee::brew() { std::cout << "Dripping coffee through filter..." << std::endl; }

void Coffee::addCondiments() { std::cout << "Adding some sugar and milk..." << std::endl; }

bool Coffee::customerWantsCondiments() {
    std::string input;
    std::cout << "Do you want to add some condiments in the coffee?(y/n)" << std::endl;
    std::cin >> input;
    if (input.length() == 1) {
        if (input[0] == 'y' || input[0] == 'Y') {
            return true;
        } else if (input[0] == 'n' || input[0] == 'N') {
            return false;
        }
    }
    std::cout << "Sorry, I can't understand your input, please try again" << std::endl;
    return customerWantsCondiments();
}