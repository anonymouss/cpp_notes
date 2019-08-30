#ifndef __CHICAGO_STYLE_CLAM_PIZZA_H__
#define __CHICAGO_STYLE_CLAM_PIZZA_H__

#include "IPizza.h"

class ChicagoStyleClamPizza : public IPizza {
public:
    ChicagoStyleClamPizza();
    virtual ~ChicagoStyleClamPizza() = default;
};

#endif  // __CHICAGO_STYLE_CLAM_PIZZA_H__