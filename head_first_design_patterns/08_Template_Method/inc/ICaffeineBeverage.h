#ifndef __I_CAFFEINE_BEVERAGE_H__
#define __I_CAFFEINE_BEVERAGE_H__

class ICaffeineBeverage {
public:
    virtual ~ICaffeineBeverage() = default;

    virtual void prepareRecipe();
    virtual void boilWater();
    virtual void pourInCup();
    virtual bool customerWantsCondiments();  // hook

    virtual void brew() = 0;
    virtual void addCondiments() = 0;
};

#endif  // __I_CAFFEINE_BEVERAGE_H__