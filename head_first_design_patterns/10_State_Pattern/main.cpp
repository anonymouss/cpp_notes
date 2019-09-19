#include "GumballMachine.h"

#include <iostream>

int main() {
    GumballMachine machine;

    auto basic_transaction = [](auto &machine) {
        machine.insertQuarter();
        machine.turnCrank();
    };

    std::cout << "--- 1st customer ---" << std::endl;
    for (auto i = 0; i < 6; ++i) {
        std::cout << "--- buy " << i << " ---" << std::endl;
        basic_transaction(machine);
    }

    std::cout << "\n--- fill goods ---" << std::endl;
    machine.setGumballCount(2);
    machine.insertQuarter();
    std::cout << "--- try clear goods during transaction ---" << std::endl;
    machine.setGumballCount(0);
    machine.insertQuarter();
    machine.turnCrank();

    std::cout << "\n--- clear goods ---" << std::endl;
    machine.setGumballCount(0);
}

/**
 * Outputs
 * [INFO] [BaseState] -> [NoQuarterState]
 * --- 1st customer ---
 * --- buy 0 ---
 * [INFO] you inserted quarter
 * [INFO] [NoQuarterState] -> [HasQuarterState]
 * [INFO] turning crank...
 * [INFO] [HasQuarterState] -> [SoldState]
 * [INFO] a gumball comes rolling out the slot...
 * [INFO] [SoldState] -> [NoQuarterState]
 * --- buy 1 ---
 * [INFO] you inserted quarter
 * [INFO] [NoQuarterState] -> [HasQuarterState]
 * [INFO] turning crank...
 * [INFO] [HasQuarterState] -> [SoldState]
 * [INFO] a gumball comes rolling out the slot...
 * [INFO] [SoldState] -> [NoQuarterState]
 * --- buy 2 ---
 * [INFO] you inserted quarter
 * [INFO] [NoQuarterState] -> [HasQuarterState]
 * [INFO] turning crank...
 * [INFO] [HasQuarterState] -> [SoldState]
 * [INFO] a gumball comes rolling out the slot...
 * [INFO] [SoldState] -> [NoQuarterState]
 * --- buy 3 ---
 * [INFO] you inserted quarter
 * [INFO] [NoQuarterState] -> [HasQuarterState]
 * [INFO] turning crank...
 * [INFO] [HasQuarterState] -> [SoldState]
 * [INFO] a gumball comes rolling out the slot...
 * [INFO] [SoldState] -> [NoQuarterState]
 * --- buy 4 ---
 * [INFO] you inserted quarter
 * [INFO] [NoQuarterState] -> [HasQuarterState]
 * [INFO] turning crank...
 * [INFO] [HasQuarterState] -> [SoldState]
 * [INFO] a gumball comes rolling out the slot...
 * [INFO] [SoldState] -> [SoldOutState]
 * --- buy 5 ---
 * [ERROR] gumball sold out... don't insert quater
 * [ERROR] gumball sold out...
 * [ERROR] gumball sold out...

 * --- fill goods ---
 * [INFO] [SoldOutState] -> [NoQuarterState]
 * [INFO] you inserted quarter
 * [INFO] [NoQuarterState] -> [HasQuarterState]
 * --- try clear goods during transaction ---
 * [ERROR] transaction in progress, invalid state to set count
 * [ERROR] you have already insert quarter
 * [INFO] turning crank...
 * [INFO] [HasQuarterState] -> [SoldState]
 * [INFO] a gumball comes rolling out the slot...
 * [INFO] [SoldState] -> [NoQuarterState]

 * --- clear goods ---
 * [INFO] [NoQuarterState] -> [SoldOutState]
 */