#ifndef __NEWYORK_STYLE_CLAM_PIZZA_H__
#define __NEWYORK_STYLE_CLAM_PIZZA_H__

#include "IPizza.h"

class NewYorkStyleClamPizza : public IPizza {
public:
    NewYorkStyleClamPizza();
    virtual ~NewYorkStyleClamPizza() = default;
};

#endif  // __NEWYORK_STYLE_CLAM_PIZZA_H__