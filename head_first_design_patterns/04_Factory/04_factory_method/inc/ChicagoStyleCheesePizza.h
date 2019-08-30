#ifndef __CHICAGO_STYLE_CHEESE_PIZZA_H__
#define __CHICAGO_STYLE_CHEESE_PIZZA_H__

#include "IPizza.h"

class ChicagoStyleCheesePizza : public IPizza {
public:
    ChicagoStyleCheesePizza();
    virtual ~ChicagoStyleCheesePizza() = default;
};

#endif  // __CHICAGO_STYLE_CHEESE_PIZZA_H__