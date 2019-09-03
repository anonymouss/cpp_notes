#include "Tea.h"

#include <iostream>
#include <string>

void Tea::brew() { std::cout << "Steeping the tea..." << std::endl; }

void Tea::addCondiments() { std::cout << "Adding lemon..." << std::endl; }

bool Tea::customerWantsCondiments() {
    std::string input;
    std::cout << "Do you want to add some condiments in the Tea?(y/n)" << std::endl;
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