#ifndef __STACK_1_H__
#define __STACK_1_H__

#include <cassert>
#include <iostream>
#include <vector>

// method 3. declare and define outside class (need class fwd declaration)
// template <typename T>
// class Stack;

// template <typename T>
// friend std::ostream &operator<<(std::ostream &os, const Stack<T> &s) {
//     s.printOn(os);
//     return os;
// }

template <typename T>
class Stack {
public:
    Stack() = default;
    virtual ~Stack() = default;

    void push(const T &elem) { mElems.push_back(elem); }
    void pop() {
        assert(!mElems.empty());
        mElems.pop_back();
    }
    const T &top() const {
        assert(!mElems.empty());
        return mElems.back();
    }
    bool empty() const { return mElems.empty(); }

    void printOn(std::ostream &os) const {
        for (const T &elem : mElems) { os << elem << ' '; }
    }

    // method 1: declare and define both in class
    friend std::ostream &operator<<(std::ostream &os, const Stack<T> &s) {
        s.printOn(os);
        return os;
    }

    // method 2: declare in class
    // template <typename U>
    // friend std::ostream &operator<<(std::ostream &os, const Stack<T> &s);

    // method 3: instantiate in class for T
    // friend std::ostream &operator<<<T>(std::ostream &os, const Stack<T> &s);

private:
    std::vector<T> mElems;
};

// method 2: define outside class
// template <typename U>
// friend std::ostream &operator<<(std::ostream &os, const Stack<T> &s) {
//     s.printOn(os);
//     return os;
// }

#endif  // _STACK_1_H__