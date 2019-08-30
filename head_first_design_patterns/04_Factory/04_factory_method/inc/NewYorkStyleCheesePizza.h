#ifndef __NEWYORK_STYLE_CHEESE_PIZZA_H__
#define __NEWYORK_STYLE_CHEESE_PIZZA_H__

#include "IPizza.h"

class NewYorkStyleCheesePizza : public IPizza {
public:
    NewYorkStyleCheesePizza();
    virtual ~NewYorkStyleCheesePizza() = default;
};

#endif  // __NEWYORK_STYLE_CHEESE_PIZZA_H__