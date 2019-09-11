#include "DinnerMenu.h"
#include "DinnerMenuIterator.h"

#include <string>

void DinnerMenu::initDefaultMenu() {
    mRealItemNums = MAX_ITEM_NUM;
    for (auto i = 0; i < MAX_ITEM_NUM; ++i) {
        mItems[i].set(std::string("item ") + std::to_string(i), "", i % 2, i * (i + 0.1) + 0.1);
    }
}

IIterator *DinnerMenu::createIterator() {
    initDefaultMenu();
    return new DinnerMenuIterator(mItems, mRealItemNums);
}