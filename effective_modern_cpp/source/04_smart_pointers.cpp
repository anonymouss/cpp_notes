/**
 * Smart Pointers
 * - item 18. Use `std::unique_ptr` for exclusive-ownership resource management
 * - item 19. Use `std::shared_ptr` for shared-ownership resource management
 * - item 20. Use `std::weak_ptr` for `std::shared_ptr`-like pointers that can dangle
 * - item 21. Prefer `std::make_unique` and `std::make_shared` to direct use of new
 * - item 22. When using the Pimpl Idiom, define special member functions in the implementation file
 */

#include "utils.h"

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

// std::unique_ptr
class Investment {
public:
    Investment() : mKind("Investment") {}
    virtual ~Investment() { printf("Destroying [%s]\n", mKind.c_str()); };
    virtual void whatKind() { printf("This is [%s]\n", mKind.c_str()); }
    std::string mKind;
};

class Stock : public Investment {
public:
    Stock() { mKind = "Stock"; }
};

class Bond : public Investment {
public:
    Bond() { mKind = "Bond"; }
};

class RealEstate : public Investment {
public:
    RealEstate() { mKind = "Real Estate"; }
};

// Factory
auto makeInvestment(std::string &&kind = "null") {
    auto delInvmt = [](Investment *pInvestment) {
        printf("now we delete pInvestment\n");
        delete pInvestment;
    };

    std::unique_ptr<Investment, decltype(delInvmt)> pInv(nullptr, delInvmt);
    if (kind == "stock") {
        pInv.reset(new Stock);
    } else if (kind == "bond") {
        pInv.reset(new Bond);
    } else if (kind == "real") {
        pInv.reset(new RealEstate);
    }
    return pInv;
}

// shared_ptr
class NoneShared;
std::vector<std::shared_ptr<NoneShared>> processedNoneShareds;

class NoneShared {
public:
    explicit NoneShared(int id) : mId(id) {}
    virtual ~NoneShared() { printf("destroy NoneShared - %d\n", mId); }
    void process() {
        processedNoneShareds.emplace_back(this);  // create new control block
    }

private:
    int mId;
};

class Shared;
std::vector<std::shared_ptr<Shared>> processedShareds;

class Shared : public std::enable_shared_from_this<Shared> {
public:
    // shared_from_this() requires a shared_ptr already exist
    static std::shared_ptr<Shared> Create(int id) {
        return std::shared_ptr<Shared>(new Shared(id));
    }
    virtual ~Shared() { printf("destroy Shared - %d\n", mId); }
    void process() {
        // won't create new control block
        processedShareds.emplace_back(shared_from_this());
    }

private:
    explicit Shared(int id) : mId(id) {}
    int mId;
};

int main() {
    INFO("`std::unique_ptr` example");
    {
        auto pStock = makeInvestment("stock");
        auto pNull = makeInvestment();
    }

    INFO("`std::shared_ptr` example");
    {
        // unsafe
        std::shared_ptr<NoneShared> ns1(new NoneShared(1));
        // ns1->process(); // will be released twice, segment fault
        ns1 = nullptr;
        processedNoneShareds.clear();  // force release if elements exist
        // safe
        // std::shared_ptr<Shared> s1(new Shared(1));
        auto s1 = Shared::Create(1);
        s1->process();
        s1 = nullptr;
        processedShareds.clear();  // force release if elements exist
    }

    INFO("`std::weak_ptr` example");
    {
        auto spi = std::make_shared<int>(5);
        std::weak_ptr<int> wpi(spi);
        spi = nullptr;  // relase
        auto spw = wpi.lock();
        printf("%s\n", spw == nullptr ? "released" : "still there");
    }
}