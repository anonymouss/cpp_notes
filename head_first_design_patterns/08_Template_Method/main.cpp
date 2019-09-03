#include "Coffee.h"
#include "Tea.h"

#include <iostream>

int main() {
    std::cout << "--- order coffee --- " << std::endl;
    Coffee coffee;
    coffee.prepareRecipe();
    std::cout << "--- order tea ------ " << std::endl;
    Tea tea;
    tea.prepareRecipe();
}

/**
 * output
 * --- order coffee ---
 * Boiling water....
 * Dripping coffee through filter...
 * Pouring in cup...
 * Do you want to add some condiments in the coffee?(y/n)
 * y
 * Adding some sugar and milk...
 * done!
 * --- order tea ------
 * Boiling water....
 * Steeping the tea...
 * Pouring in cup...
 * Do you want to add some condiments in the Tea?(y/n)
 * xxx
 * Sorry, I can't understand your input, please try again
 * Do you want to add some condiments in the Tea?(y/n)
 * n
 * done!
 */