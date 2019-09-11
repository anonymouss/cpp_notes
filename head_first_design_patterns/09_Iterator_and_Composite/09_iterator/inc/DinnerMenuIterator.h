#ifndef __DINNER_MENU_ITERATOR_H__
#define __DINNER_MENU_ITERATOR_H__

#include "IIterator.h"
#include "MenuItem.h"

class DinnerMenuIterator : public IIterator {
public:
    virtual ~DinnerMenuIterator() = default;
    DinnerMenuIterator();
    DinnerMenuIterator(MenuItem *menu_array, std::size_t menu_size);

    virtual bool hasNext() final;
    virtual MenuItem *next() final;

private:
    std::size_t mArraySize;
    std::size_t mPosition;
    MenuItem *pMenuItemArray;
};

#endif  // __DINNER_MENU_ITERATOR_H__