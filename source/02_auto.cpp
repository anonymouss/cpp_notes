/**
 * auto
 * - item 5: Prefer `auto` to explicit type declarations
 * - item 6: Use the explicitly typed initializer idiom when `auto` deduces undesired types
 */

#include "utils.h"

#include <cstdio>
#include <functional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// help class for type shortcut example
struct StringWrapper {
    std::string mString;

    StringWrapper(std::string s) : mString(std::move(s)) {}
    ~StringWrapper() { mString = mString + " is already destroyed!"; }
};

struct StringWrapperHash {
    std::size_t operator()(const StringWrapper &sw) const {
        return std::hash<std::string>{}(sw.mString);
    }
};

struct StringWrapperEqual {
    bool operator()(const StringWrapper &lhs, const StringWrapper &rhs) const {
        return lhs.mString == rhs.mString;
    }
};

int main() {
    // type shortcuts
    INFO("Type shortcuts example");
    std::unordered_map<StringWrapper, int, StringWrapperHash, StringWrapperEqual> map{
        {{"Apple"}, 0}, {{"Orange"}, 1}};

    INFO("Without auto");
    const StringWrapper *psw1 = nullptr;
    for (const std::pair<StringWrapper, int> &p : map) {  // 1.
        psw1 = &(p.first);
        printf("  inside  scope: %s\n", psw1->mString.c_str());
    }
    // at here, the object psw1 points to is already destroyed
    printf("  outside scope: %s\n", psw1->mString.c_str());

    INFO("With auto");
    const StringWrapper *psw2 = nullptr;
    for (const auto &p : map) {
        psw2 = &(p.first);
        printf("  inside  scope: %s\n", psw2->mString.c_str());
    }
    // the object psw2 points to is stll there
    printf("  outside scope: %s\n", psw2->mString.c_str());
    // what the hell...
    // https://zh.cppreference.com/w/cpp/container/unordered_map
    // https://zh.cppreference.com/w/cpp/container/unordered_map/deduction_guides
    // https://github.com/gcc-mirror/gcc/blob/master/libstdc%2B%2B-v3/include/bits/unordered_map.h
    // the underlying value_type of std::unordered_map is std::pair<const Key, T>, it's different
    // with declaration in `1.`. That means, type conversion occurs, temporial object is created,
    // that is what psw1 points to... obviously, temporial object will be destroyed after loop exit.
    // Well, auto can deduce the real underlying type. C++ is full of traps...

    INFO("auto deduced undesired type");
    std::vector<bool> vec{true, false, true, false};

    auto echo_bool = [](bool b) {
        if (b) {
            printf("Ture\n");
        } else {
            printf("False\n");
        }
    };

    auto disp_vec_bool = [&]() {
        for (bool b : vec) { printf("%s, ", b ? "true" : "false"); }
        printf("\n");
    };

    auto b1 = vec[1];
    ECHO_TYPE_PARAM(b1);
    // b1 is a proxy type, std::vector<bool>::reference, it just acts like bool
    b1 = !b1;  // will affect vec
    disp_vec_bool();

    auto b2 = static_cast<bool>(vec[2]);
    ECHO_TYPE_PARAM(b2);  // b2 is bool (it's a copy)
    b2 = !b2;             // no affect to vec
    disp_vec_bool();
}