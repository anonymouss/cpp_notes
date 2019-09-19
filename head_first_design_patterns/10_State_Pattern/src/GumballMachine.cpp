#include "GumballMachine.h"

#include <iostream>

constexpr int DEFAULT_COUNT = 5;

struct GumballMachine::BaseState : public IState {
    explicit BaseState(GumballMachine *machine) : mMachine(machine) {}
    virtual ~BaseState() = default;

    virtual void insertQuarter() override{};
    virtual void ejectQuarter() override{};
    virtual void turnCrank() override{};
    virtual void dispense() override{};

    virtual std::string toString() const { return mName; }

    std::string mName = "[BaseState]";
    GumballMachine *mMachine;
};

struct GumballMachine::NoQuarterState : public GumballMachine::BaseState {
    explicit NoQuarterState(GumballMachine *machine) : BaseState(machine) {}
    virtual void insertQuarter() final {
        std::cout << "[INFO] you inserted quarter" << std::endl;
        mMachine->changeState(mMachine->mHasQuarterState);
    }

    virtual void ejectQuarter() final {
        std::cout << "[ERROR] you have not insert quarter" << std::endl;
    };

    virtual void turnCrank() final {
        std::cout << "[ERROR] you have not insert quarter" << std::endl;
    };

    virtual void dispense() final {
        std::cout << "[ERROR] you have not insert quarter" << std::endl;
    };

    virtual std::string toString() const final { return mName; }

    std::string mName = "[NoQuarterState]";
};

struct GumballMachine::HasQuarterState : public GumballMachine::BaseState {
    explicit HasQuarterState(GumballMachine *machine) : BaseState(machine) {}

    virtual void insertQuarter() final {
        std::cout << "[ERROR] you have already insert quarter" << std::endl;
    }

    virtual void ejectQuarter() final {
        std::cout << "[INFO] ok, give back quarter to you" << std::endl;
        mMachine->changeState(mMachine->mNoQuarterState);
    };

    virtual void turnCrank() final {
        std::cout << "[INFO] turning crank..." << std::endl;
        mMachine->changeState(mMachine->mSoldState);
    };

    virtual void dispense() final { std::cout << "[ERROR] no gumball dispense" << std::endl; };

    virtual std::string toString() const final { return mName; }

    std::string mName = "[HasQuarterState]";
};

struct GumballMachine::SoldState : public GumballMachine::BaseState {
    explicit SoldState(GumballMachine *machine) : BaseState(machine) {}

    virtual void insertQuarter() final {
        std::cout << "[ERROR] please wait, an order is handling" << std::endl;
    }

    virtual void ejectQuarter() final {
        std::cout << "[ERROR] sorry, you have already turnned the crank" << std::endl;
    };

    virtual void turnCrank() final {
        std::cout << "[ERROR] oh. you can't turn the crank twice" << std::endl;
    };

    virtual void dispense() final {
        mMachine->releaseGumball();
        if (mMachine->isEmpty()) {
            mMachine->changeState(mMachine->mSoldOutState);
        } else {
            mMachine->changeState(mMachine->mNoQuarterState);
        }
    };

    virtual std::string toString() const final { return mName; }

    std::string mName = "[SoldState]";
};

struct GumballMachine::SoldOutState : public GumballMachine::BaseState {
    explicit SoldOutState(GumballMachine *machine) : BaseState(machine) {}

    virtual void insertQuarter() final {
        std::cout << "[ERROR] gumball sold out... don't insert quater" << std::endl;
    }

    virtual void ejectQuarter() final { std::cout << "[ERROR] gumball sold out..." << std::endl; };

    virtual void turnCrank() final { std::cout << "[ERROR] gumball sold out..." << std::endl; };

    virtual void dispense() final { std::cout << "[ERROR] gumball sold out..." << std::endl; };

    virtual std::string toString() const final { return mName; }

    std::string mName = "[SoldOutState]";
};

GumballMachine::GumballMachine() {
    mGumballCounts = DEFAULT_COUNT;
    mNoQuarterState = std::make_shared<NoQuarterState>(this);
    mHasQuarterState = std::make_shared<HasQuarterState>(this);
    mSoldState = std::make_shared<SoldState>(this);
    mSoldOutState = std::make_shared<SoldOutState>(this);

    mState = std::make_shared<BaseState>(this);

    if (mGumballCounts > 0) {
        changeState(mNoQuarterState);
    } else {
        changeState(mSoldOutState);
    }
}

void GumballMachine::setGumballCount(int count) {
    if (mState == mNoQuarterState || mState == mSoldOutState) {
        mGumballCounts = count;
    } else {
        std::cout << "[ERROR] transaction in progress, invalid state to set count" << std::endl;
        return;
    }

    if (mGumballCounts > 0) {
        changeState(mNoQuarterState);
    } else {
        changeState(mSoldOutState);
    }
}

void GumballMachine::changeState(std::shared_ptr<GumballMachine::BaseState> state) {
    if (mState == state) {
        std::cout << "[WARNING] State Unchanged!" << std::endl;
        return;
    }

    std::cout << "[INFO] " << mState->toString() << " -> " << state->toString() << std::endl;
    mState = state;
}

void GumballMachine::releaseGumball() {
    std::cout << "[INFO] a gumball comes rolling out the slot..." << std::endl;
    --mGumballCounts;
}

bool GumballMachine::isEmpty() const { return mGumballCounts <= 0; }

void GumballMachine::insertQuarter() { mState->insertQuarter(); }

void GumballMachine::ejectQuarter() { mState->ejectQuarter(); }

void GumballMachine::turnCrank() {
    mState->turnCrank();
    dispense();
}

void GumballMachine::dispense() { mState->dispense(); }