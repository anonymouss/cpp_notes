#ifndef __I_PIZZA_H__
#define __I_PIZZA_H__

#include "ICheese.h"
#include "IClams.h"
#include "IDough.h"
#include "ISauce.h"

#include <iostream>
#include <memory>
#include <string>

class IPizza {
public:
    virtual ~IPizza() = default;
    virtual void prepare() = 0;
    virtual void make() {
        std::cout << "  making pizza: " << mName << std::endl;
        auto add = [](const auto &ingredient) {
            if (ingredient) { std::cout << "\tadding some " << ingredient->getName() << std::endl; }
        };
        add(mDough);
        add(mCheese);
        add(mSauce);
        add(mClams);
    }

protected:
    std::string mName;

    std::unique_ptr<IDough> mDough;
    std::unique_ptr<ICheese> mCheese;
    std::unique_ptr<ISauce> mSauce;
    std::unique_ptr<IClams> mClams;
};

#endif  // __I_PIZZA_H__