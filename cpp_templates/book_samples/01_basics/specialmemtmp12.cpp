#include "specialmemtmp12.h"

#include <string>

int main() {
    std::string s = "sname";
    Person p1(s);
    Person p2("tmp");

    Person p3(p2);
}