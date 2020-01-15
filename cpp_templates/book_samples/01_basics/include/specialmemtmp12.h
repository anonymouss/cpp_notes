#ifndef __SPECIAL_MEM_TMP_12_H__
#define __SPECIAL_MEM_TMP_12_H__

#include <utility>
#include <string>
#include <iostream>
#include <type_traits>

class Person {
public:
    template <typename STR, typename = std::enable_if_t<std::is_convertible_v<STR, std::string>>>
    explicit Person(STR &&n) : name(std::forward<STR>(n)) {
        std::cout << "TMPL-CONSTR for " << name << std::endl;
    }

    Person(const Person &p) : name(p.name) {
        std::cout << "COPY-CONSTR Person " << name << std::endl;
    }
    Person(Person &&p) : name(std::move(p.name)) {
        std::cout << "MOVE-CONSTR Person " << name << std::endl;
    }

private:
    std::string name;
};

#endif // __SPECIAL_MEM_TMP_12_H__