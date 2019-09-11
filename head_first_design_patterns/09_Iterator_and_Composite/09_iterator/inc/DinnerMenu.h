#ifndef __DINNER_MENU_H__
#define __DINNER_MENU_H__

#include "DinnerMenuIterator.h"
#include "MenuItem.h"

constexpr int MAX_ITEM_NUM = 6;

class DinnerMenu {
public:
    virtual ~DinnerMenu() = default;
    void initDefaultMenu();
    IIterator *createIterator();

private:
    MenuItem mItems[MAX_ITEM_NUM];
    int mRealItemNums;
};

#endif  // __DINNER_MENU_H__