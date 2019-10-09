#ifndef __ADD_SPACE_H__
#define __ADD_SPACE_H__

#include <iostream>

template <typename T>
class AddSpace {
public:
    AddSpace(const T &r) : ref(r) {}

    friend std::ostream &operator<<(std::ostream &os, AddSpace<T> s) { return os << s.ref << ' '; }

private:
    const T &ref;
};

template <typename... Args>
void print(Args... args) {
    (std::cout << ... << AddSpace(args)) << std::endl;
}

#endif  // __ADD_SPACE_H__