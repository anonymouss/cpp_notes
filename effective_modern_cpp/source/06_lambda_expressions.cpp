/**
 * Lambda Expressions
 * - item 31. Avoid default capture modes.
 * - item 32. Use init capture to move objects into closures.
 * - item 33. Use `decltype` on `auto&&` parameters to std::forward them.
 * - item 34. Prefer `lambdas` to `std::bind`.
 */

#include "utils.h"

#include <cstdio>
#include <functional>
#include <vector>

using FilterContainer = std::vector<std::function<bool(int)>>;
FilterContainer filters;

class FilterWidget {
public:
    void setDivisor(int d) { divisor = d; }
    void addFilter() const {
        // if use [] or use [divisor], can't compile
        // [=] here, it actually captured this pointer. divisor is this->divisor
        filters.emplace_back([=](int value) { return value % divisor == 0; });
    }

private:
    int divisor;
};

int main() {
    INFO("Avoid using default capture mode");
    {
        {
            auto divisor = 5;
            filters.emplace_back([&](int value) { return value % divisor == 0; });
            divisor = 0;
        }

        auto filter5 = filters.back();
        // printf("%d\n", filter5(5)); // raise exception! divided by 0

        FilterWidget fw;
        fw.setDivisor(3);
        fw.addFilter();
        auto filter3 = filters.back();
        printf("%d\n", filter3(3));
        fw.setDivisor(0);
        // printf("%d\n", filter3(3)); // raise exception
    }

    INFO("init capture");
    {
        {
            // for C++ 14
            std::vector<int> vec{1, 2, 3, 4, 5};
            auto moveVec = [data = std::move(vec)]() {
                for (const auto &v : data) { printf("%d, ", v); }
                printf("\n");
            };
            moveVec();
            printf("after move, vec has size = %zu\n", vec.size());
        }
        {
            // for C++ 11
            std::vector<int> vec{1, 2, 3, 4, 5};
            auto moveVec = std::bind(
                [](const std::vector<int> &data) {
                    for (const auto &v : data) { printf("%d, ", v); }
                    printf("\n");
                },
                std::move(vec));
            moveVec();
            printf("after move, vec has size = %zu\n", vec.size());
        }
    }
}