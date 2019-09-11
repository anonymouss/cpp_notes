#include "DinnerMenuIterator.h"

DinnerMenuIterator::DinnerMenuIterator() : mArraySize(0), mPosition(0), pMenuItemArray(nullptr) {}

DinnerMenuIterator::DinnerMenuIterator(MenuItem *menu_array, std::size_t menu_size)
    : mArraySize(menu_size), mPosition(0), pMenuItemArray(menu_array) {}

bool DinnerMenuIterator::hasNext() {
    if (!pMenuItemArray) return false;
    if (mPosition >= mArraySize || !(pMenuItemArray + mPosition)) return false;
    return true;
}

MenuItem *DinnerMenuIterator::next() {
    MenuItem *next = nullptr;
    if (hasNext()) { next = pMenuItemArray + (mPosition++); }
    return next;
}

// DinnerMenuIterator::~DinnerMenuIterator() {
//     if (pMenuItemArray) { delete[] pMenuItemArray; }
// }
