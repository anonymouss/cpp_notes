#ifndef __MENU_ITEM_H__
#define __MENU_ITEM_H__

#include <string>

class MenuItem {
public:
    virtual ~MenuItem() = default;
    MenuItem() = default;
    MenuItem(std::string name, std::string desc, bool is_veg, double price)
        : mName(name), mDescription(desc), bIsVegetarian(is_veg), mPrice(price) {}
    void set(std::string name, std::string desc, bool is_veg, double price) {
        mName =name;
        mDescription = desc;
        bIsVegetarian = is_veg;
        mPrice = price;
    }
    std::string getName() const { return mName; }
    std::string getDescription() const { return mDescription; }
    bool isVegetarian() const { return bIsVegetarian; }
    double getPrice() const { return mPrice; }

protected:
    std::string mName;
    std::string mDescription;
    bool bIsVegetarian;
    double mPrice;
};

#endif  // __MENU_ITEM_H__