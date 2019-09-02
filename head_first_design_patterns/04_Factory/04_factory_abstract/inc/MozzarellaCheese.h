#ifndef __MOZZARELLA_CHEESE_H__
#define __MOZZARELLA_CHEESE_H__

#include "ICheese.h"

class MozzarellaCheese : public ICheese {
public:
    MozzarellaCheese() { mName = "<mozzarella cheese>"; }
};

#endif  // __MOZZARELLA_CHEESE_H__