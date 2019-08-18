/**
 * Rvalue References, Move Semantics, and Perfect Forwarding
 * - item 23. Understand `std::move` and `std::forward`
 * - item 24. Distinguish universal references from rvalue references
 * - item 25. Use `std::move` on rvalue references, `std::forward` on rvalue references
 * - item 26. Avoid overloading on universal references
 * - item 27. Familiarize yourself with alternatives to overloading on universal references
 * - item 28. Understand reference collapsing
 * - item 29. Assume that move operations are not present, not cheap, and not used
 * - item 30. Familiarize yourself with perfect forwarding failure cases
 */

#include "utils.h"

#include <cstdio>
#include <string>
#include <type_traits>

struct Int {
    Int(int v = 0) : value(v) {
        // printf("construt to %d\n", v);
    }
    virtual ~Int() = default;
    Int(const Int &rhs) {
        printf("copy construct %d\n", rhs.value);
        this->value = rhs.value;
    }
    Int &operator=(const Int &rhs) {
        printf("copy assign %d\n", rhs.value);
        this->value = rhs.value;
        return *this;
    }
    Int(Int &&rhs) {
        printf("move construct %d\n", rhs.value);
        this->value = std::move(rhs.value);
        rhs.value = -1;  // destroy
    }
    Int &operator=(Int &&rhs) {
        printf("move assign %d\n", rhs.value);
        this->value = std::move(rhs.value);
        rhs.value = -1;  // destroy
        return *this;
    }
    Int &operator+=(const Int &rhs) {
        this->value += rhs.value;
        return *this;
    }
    Int operator+(const Int &rhs) {
        Int sum;
        sum = this->value + rhs.value;
        return sum;
    }

    int value;
};

struct MoveTest {
    void tryMoveWithConst(const Int i) {
        printf("%s\n", __func__);
        value = std::move(i);
    }
    void tryMoveWithNoConst(Int i) {
        printf("%s\n", __func__);
        value = std::move(i);
    }
    Int value;
};

void handleInt(const Int &it) { printf("handling lvalue Int %d\n", it.value); }
void handleInt(Int &&it) { printf("handling rvalue int %d\n", it.value); }
template <typename T>
void handleForward(T &&it) {
    ECHO_TYPE_T(T);
    handleInt(std::forward<T>(it));
}

template <typename INT>
auto stealInt(INT &&i) {
    // move on universal reference
    return std::move(i);
}

Int operator+(Int &&lhs, const Int &rhs) {
    lhs += rhs;
    // return lhs;
    return std::move(lhs);
}

template <typename STR>
void processItem(STR &&name) {
    printf("calling %s with universal reference argument\n", __func__);
}

void processItem(int index) { printf("calling %s with int argument\n", __func__); }

const static std::string kNames[] = {"Name 1", "Name 2", "Name 3"};

struct Person {
    template <typename T>
    explicit Person(T &&n) : name(std::forward<T>(n)) {
        printf("univ reference\n");
    }
    explicit Person(int index) {
        if (index < 0 || index > 2) { index = 0; }
        name = kNames[index];
    }
    Person(const Person &p) {
        printf("cpy ctor\n");
        this->name = p.name;
    }
    Person(Person &&p) {
        printf("mv ctor\n");
        this->name = std::move(p.name);
    }

    std::string name;
};

// std::is_base_of_t is from C++17
struct Animal {
    template <typename T, typename = typename std::enable_if_t<
                              !std::is_base_of<Animal, typename std::decay_t<T> >::value> >
    explicit Animal(T &&n) {
        printf("Animal univ ref\n");
        mName = std::forward<T>(n);
    }
    Animal(const Animal &animal) {
        printf("Animal cpy ctor\n");
        this->mName = animal.mName;
    }
    Animal(Animal &&animal) {
        printf("Animal mv ctor\n");
        this->mName = std::move(animal.mName);
    }

    std::string mName;
};

struct Cat : public Animal {
    Cat(const char *name) : Animal(name) {}
    Cat(const Cat &c) : Animal(c) { printf("Cat cpy ctor\n"); }
    Cat(Cat &&c) : Animal(std::move(c)) { printf("Cat mv ctor\n"); }
};

int main() {
    // disable copy elision by -fno-elide-constructors
    // https://zh.cppreference.com/w/cpp/language/copy_elision
    INFO("move and forward");
    {
        MoveTest mt;
        mt.tryMoveWithConst(1);    // copy
        mt.tryMoveWithNoConst(2);  // move

        Int it = 3;
        handleForward(it);
        handleForward(std::move(it));
    }

    INFO("move, forward on rvalue reference and universal reference");
    {
        Int i(100);
        printf("orginal i.value = %d\n", i.value);
        auto s = stealInt(i);  // pass lvalue. but it's moved
        printf("now     i.value = %d\n", i.value);
        printf("now     s.value = %d\n", s.value);

        Int a = 20, b = 50;
        // auto sum =  a + 10; // call member func operator+
        auto sum = 10 + a;
        printf("%d + 10 = %d\n", a.value, sum.value);
    }

    INFO("don't overloading on universal reference");
    {
        short idx = 0;
        processItem(idx);  // call universal reference version, not int version

        // universal reference version is best match
        Person p("test");
        // ERROR, match universal reference version, not default cpy ctor
        // because p's type is Person while cpy ctor is const P &, so univ ref
        // is the best match
        // auto cloneOfP(p);
        const Person q("test");
        auto cloneOfQ(q);  // OK, call cpy ctor
    }

    INFO("alternatives for univ ref overloading");
    {
        Animal ani1("hello");
        Animal ani2(ani1);
        printf("-----");
        Cat c1("world");
        Cat c2(c1);
    }
}