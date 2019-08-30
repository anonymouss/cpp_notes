#ifndef __VEGGIE_PIZZA_H__
#define __VEGGIE_PIZZA_H__

#include "IPizza.h"

class VeggiePizza : public IPizza {
public:
    explicit VeggiePizza(const char *name);
    virtual ~VeggiePizza() = default;
};

#endif  // __VEGGIE_PIZZA_H__