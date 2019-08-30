#ifndef __CHEESE_PIZZA_H__
#define __CHEESE_PIZZA_H__

#include "IPizza.h"

class CheesePizza : public IPizza {
public:
    explicit CheesePizza(const char *name);
    virtual ~CheesePizza() = default;
};

#endif  // __CHEESE_PIZZA_H__