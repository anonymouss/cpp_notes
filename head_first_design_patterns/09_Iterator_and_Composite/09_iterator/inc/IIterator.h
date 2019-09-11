#ifndef __I_ITERATOR_H__
#define __I_ITERATOR_H__

#include "MenuItem.h"

class IIterator {
public:
    virtual ~IIterator() = default;
    virtual bool hasNext() = 0;
    virtual MenuItem *next() = 0;
};

#endif  // __I_ITERATOR_H__