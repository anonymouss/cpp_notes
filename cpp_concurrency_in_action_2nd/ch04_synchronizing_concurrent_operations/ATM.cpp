// CSP (Communicating Sequential Processes)

#include <condition_variable>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

namespace msg {

struct MessageBase {
    virtual ~MessageBase() {}
};

template <typename Msg>
struct MessageWrapper : public MessageBase {
    Msg details;
    explicit MessageWrapper(const Msg &_details) : details(_details) {}
};

class MessageQueue {
public:
    template <typename Msg>
    void push(const Msg &msg) {
        std::scoped_lock lock(mMutex);
        mMsgQueue.push(std::make_shared<MessageWrapper<Msg>>(msg));
        mCond.notify_all();
    }

    std::shared_ptr<MessageBase> wait_and_pop() {
        std::unique_lock lock(mMutex);
        mCond.wait(lock, [&] { return !mMsgQueue.empty(); });
        auto msg = mMsgQueue.front();
        mMsgQueue.pop();
        return msg;
    }

private:
    std::mutex mMutex;
    std::condition_variable mCond;
    std::queue<std::shared_ptr<MessageBase>> mMsgQueue;
};

struct CloseMessageQueue {};

class Sender {
public:
    Sender() : mMsgQueue(nullptr) {}
    explicit Sender(MessageQueue *queue) : mMsgQueue(queue) {}

    template <typename Msg>
    void send(const Msg &msg) {
        if (mMsgQueue) {
            mMsgQueue->push(msg);
        } else {
            throw std::runtime_error("Sender is uninitialized");
        }
    }

private:
    MessageQueue *mMsgQueue;
};

template <typename Disp, typename Msg, typename Func>
class DispatcherImpl {
public:
    DispatcherImpl(DispatcherImpl &&other)
        : mMsgQueue(other.mMsgQueue),
          mPrevDispatcher(other.mPrevDispatcher),
          mFunction(std::move(other.mFunction)),
          mIsChained(other.mIsChained) {
        other.mIsChained = true;
    }

    DispatcherImpl(MessageQueue *queue, Disp *prev, Func &&fn)
        : mMsgQueue(queue),
          mPrevDispatcher(prev),
          mFunction(std::forward<Func>(fn)),
          mIsChained(false) {
        mPrevDispatcher->mIsChained = true;
    }

    template <typename OtherMsg, typename OtherFunc>
    DispatcherImpl<DispatcherImpl, OtherMsg, OtherFunc> handle(OtherFunc &&otherFn) {
        return DispatcherImpl<DispatcherImpl, OtherMsg, OtherFunc>{
            mMsgQueue, this, std::forward<OtherFunc>(otherFn)};
    }

    virtual ~DispatcherImpl() noexcept(false) {
        if (!mIsChained) { wait_and_dispatch_l(); }
    }

private:
    MessageQueue *mMsgQueue;
    Disp *mPrevDispatcher;
    Func mFunction;
    bool mIsChained;

    DispatcherImpl() = delete;
    DispatcherImpl(const DispatcherImpl &) = delete;
    DispatcherImpl &operator=(const DispatcherImpl &) = delete;

    template <typename OtherDisp, typename OtherMsg, typename OtherFunc>
    friend class DispatcherImpl;

    void wait_and_dispatch_l() {
        for (;;) {
            auto msg = mMsgQueue->wait_and_pop();
            if (dispatch_l(msg)) {
                break;
            } else {
                // throw std::runtime_error("Failed to dispatch message");
            }
        }
    }

    bool dispatch_l(const std::shared_ptr<MessageBase> &msg) {
        if (MessageWrapper<Msg> *wrapper = dynamic_cast<MessageWrapper<Msg> *>(msg.get())) {
            mFunction(wrapper->details);
            return true;
        } else {
            return mPrevDispatcher->dispatch_l(msg);
        }
    }
};

class Dispatcher {
public:
    Dispatcher(Dispatcher &&other) : mMsgQueue(other.mMsgQueue), mIsChained(other.mIsChained) {
        other.mIsChained = true;
    }
    explicit Dispatcher(MessageQueue *queue) : mMsgQueue(queue), mIsChained(false) {}

    template <typename Msg, typename Func>
    DispatcherImpl<Dispatcher, Msg, Func> handle(Func &&fn) {
        return DispatcherImpl<Dispatcher, Msg, Func>{mMsgQueue, this, std::forward<Func>(fn)};
    }

    virtual ~Dispatcher() noexcept(false) {
        if (!mIsChained) { wait_and_dispatch_l(); }
    }

private:
    MessageQueue *mMsgQueue;
    bool mIsChained;

    Dispatcher() = delete;
    Dispatcher(const Dispatcher &) = delete;
    Dispatcher &operator=(const Dispatcher &) = delete;

    template <typename Disp, typename Msg, typename Func>
    friend class DispatcherImpl;

    void wait_and_dispatch_l() {
        for (;;) {
            auto msg = mMsgQueue->wait_and_pop();
            dispatch_l(msg);
        }
    }

    bool dispatch_l(const std::shared_ptr<MessageBase> &msg) {
        if (dynamic_cast<MessageWrapper<CloseMessageQueue> *>(msg.get())) {
            throw CloseMessageQueue{};
        }
        return false;
    }
};

class Receiver {
public:
    operator Sender() { return Sender{&mMsgQueue}; }
    Dispatcher wait() { return Dispatcher{&mMsgQueue}; }

private:
    MessageQueue mMsgQueue;
};

}  // namespace msg

// ATM withdraw related
struct Withdraw {
    std::string account;
    unsigned amount;
    mutable msg::Sender atm_sender;

    Withdraw(const std::string &_account, unsigned _amount, msg::Sender _sender)
        : account(_account), amount(_amount), atm_sender(_sender) {}
};
struct Withdraw_OK {};
struct Withdraw_DENIED {};
struct Withdraw_CANCELLING {
    std::string account;
    unsigned amount;

    Withdraw_CANCELLING(const std::string &_account, unsigned _amount)
        : account(_account), amount(_amount) {}
};
struct Withdraw_PROCESSED {
    std::string account;
    unsigned amount;
    Withdraw_PROCESSED(const std::string &_account, unsigned _amount)
        : account(_account), amount(_amount) {}
};
struct Withdraw_PRESSED {
    unsigned amount;
    explicit Withdraw_PRESSED(unsigned _amount) : amount(_amount) {}
};

// ATM digital operations
struct Digit_PRESSED {
    char digit;
    explicit Digit_PRESSED(char _digit) : digit(_digit) {}
};
struct Digit_CLEAR_LAST {};

// ATM card related
struct Card {
    std::string account;
    explicit Card(const std::string &_account) : account(_account) {}
};
struct Card_INSERTED {
    std::string account;
    Card_INSERTED(const std::string &_account) : account(_account) {}
};
struct Card_EJECTING {};

// ATM cancel operation
struct Cancel_PRESSED {};

// ATM money related
struct Money_ISSUING {
    unsigned amount;
    explicit Money_ISSUING(unsigned _amount) : amount(_amount) {}
};

// ATM balance related
struct Balance {
    unsigned amount;
    explicit Balance(unsigned _amount) : amount(_amount) {}
};
struct Balance_GET {
    std::string account;
    mutable msg::Sender atm_sender;
    Balance_GET(const std::string &_account, msg::Sender _sender)
        : account(_account), atm_sender(_sender) {}
};
struct Balance_PRESSED {};

// ATM PIN code related
struct PIN_VERIFYING {
    std::string account;
    std::string pin;
    mutable msg::Sender atm_sender;
    PIN_VERIFYING(const std::string &_account, const std::string &_pin, msg::Sender _sender)
        : account(_account), pin(_pin), atm_sender(_sender) {}
};
struct PIN_VERIFIED {};
struct PIN_INCORRECT {};

// ATM display related
struct Display_ENTER_PIN {};
struct Display_INSERT_CARD {};
struct Display_INSUFFICIENT_FUNDS {};
struct Display_WITHDRAWAL_CANCELED {};
struct Display_PIN_INCORRECT {};
struct Display_WITHDRAWAL_OPTIONS {};
struct Display_BALANCE {
    unsigned balance;
    explicit Display_BALANCE(unsigned _balance) : balance(_balance) {}
};

// ATM facilities
class ATMMachine {
public:
    ATMMachine(msg::Sender bank, msg::Sender intf) : mBankSender(bank), mHWSender(intf) {}
    void done() { get_sender().send(msg::CloseMessageQueue{}); }
    void run() {
        mState = &ATMMachine::handle_wait_for_card;
        try {
            for (;;) { (this->*mState)(); }
        } catch (const msg::CloseMessageQueue &e) { std::cout << "ATM: closing" << std::endl; }
    }

    msg::Sender get_sender() { return mOpReceiver; }

private:
    msg::Receiver mOpReceiver;
    msg::Sender mBankSender;
    msg::Sender mHWSender;
    void (ATMMachine::*mState)();
    std::string mAccount;
    std::string mPINCode;
    unsigned mWithdrawalAmount;

    void handle_withdrawal() {
        mOpReceiver.wait()
            .handle<Withdraw_OK>([&](const Withdraw_OK &msg) {
                mHWSender.send(Money_ISSUING{mWithdrawalAmount});
                mBankSender.send(Withdraw_PROCESSED{mAccount, mWithdrawalAmount});
                mState = &ATMMachine::handle_done;
            })
            .handle<Withdraw_DENIED>([&](const Withdraw_DENIED &msg) {
                mHWSender.send(Display_INSUFFICIENT_FUNDS{});
                mState = &ATMMachine::handle_done;
            })
            .handle<Cancel_PRESSED>([&](const Cancel_PRESSED &msg) {
                mBankSender.send(Withdraw_CANCELLING{mAccount, mWithdrawalAmount});
                mHWSender.send(Display_WITHDRAWAL_CANCELED{});
                mState = &ATMMachine::handle_done;
            });
    }
    void handle_balance() {
        mOpReceiver.wait()
            .handle<Balance>([&](const Balance &msg) {
                mHWSender.send(Display_BALANCE{msg.amount});
                mState = &ATMMachine::handle_wait_for_action;
            })
            .handle<Cancel_PRESSED>(
                [&](const Cancel_PRESSED &msg) { mState = &ATMMachine::handle_done; });
    }
    void handle_wait_for_action() {
        mHWSender.send(Display_WITHDRAWAL_OPTIONS{});
        mOpReceiver.wait()
            .handle<Withdraw_PRESSED>([&](const Withdraw_PRESSED &msg) {
                mWithdrawalAmount = msg.amount;
                mBankSender.send(Withdraw{mAccount, mWithdrawalAmount, mOpReceiver});
                mState = &ATMMachine::handle_withdrawal;
            })
            .handle<Balance_PRESSED>([&](const Balance_PRESSED &msg) {
                mBankSender.send(Balance_GET{mAccount, mOpReceiver});
                mState = &ATMMachine::handle_balance;
            })
            .handle<Cancel_PRESSED>(
                [&](const Cancel_PRESSED &msg) { mState = &ATMMachine::handle_done; });
    }
    void handle_pin_verification() {
        mOpReceiver.wait()
            .handle<PIN_VERIFIED>(
                [&](const PIN_VERIFIED &msg) { mState = &ATMMachine::handle_wait_for_action; })
            .handle<PIN_INCORRECT>([&](const PIN_INCORRECT &msg) {
                mHWSender.send(Display_PIN_INCORRECT{});
                mState = &ATMMachine::handle_done;
            })
            .handle<Cancel_PRESSED>(
                [&](const Cancel_PRESSED &msg) { mState = &ATMMachine::handle_done; });
    }
    void handle_pin_inputting() {
        mOpReceiver.wait()
            .handle<Digit_PRESSED>([&](const Digit_PRESSED &msg) {
                constexpr unsigned LEN = 4;
                mPINCode += msg.digit;
                if (mPINCode.length() == LEN) {
                    mBankSender.send(PIN_VERIFYING{mAccount, mPINCode, mOpReceiver});
                    mState = &ATMMachine::handle_pin_verification;
                }
            })
            .handle<Digit_CLEAR_LAST>([&](const Digit_CLEAR_LAST &msg) {
                if (!mPINCode.empty()) { mPINCode.pop_back(); }
            })
            .handle<Cancel_PRESSED>(
                [&](const Cancel_PRESSED &msg) { mState = &ATMMachine::handle_done; });
    }
    void handle_wait_for_card() {
        mHWSender.send(Display_INSERT_CARD{});
        mOpReceiver.wait().handle<Card_INSERTED>([&](const Card_INSERTED &msg) {
            mAccount = msg.account;
            mPINCode = "";
            mHWSender.send(Display_ENTER_PIN{});
            mState = &ATMMachine::handle_pin_inputting;
        });
    }
    void handle_done() {
        mHWSender.send(Card_EJECTING{});
        mState = &ATMMachine::handle_wait_for_card;
    }
};

class Bank {
public:
    explicit Bank(unsigned balance = 199) : mBalance(balance) {}
    void done() { get_sender().send(msg::CloseMessageQueue{}); }
    void run() {
        try {
            for (;;) {
                mOpReceiver.wait()
                    .handle<PIN_VERIFYING>([&](const PIN_VERIFYING &msg) {
                        if (msg.pin == "1234") {
                            msg.atm_sender.send(PIN_VERIFIED{});
                        } else {
                            msg.atm_sender.send(PIN_INCORRECT{});
                        }
                    })
                    .handle<Withdraw>([&](const Withdraw &msg) {
                        if (mBalance >= msg.amount) {
                            msg.atm_sender.send(Withdraw_OK{});
                            mBalance -= msg.amount;
                        } else {
                            msg.atm_sender.send(Withdraw_DENIED{});
                        }
                    })
                    .handle<Balance_GET>(
                        [&](const Balance_GET &msg) { msg.atm_sender.send(Balance{mBalance}); })
                    .handle<Withdraw_PROCESSED>([&](const Withdraw_PROCESSED &msg) {})
                    .handle<Withdraw_CANCELLING>([&](const Withdraw_CANCELLING &msg) {});
            }
        } catch (const msg::CloseMessageQueue &) { std::cout << "Bank: closing" << std::endl; }
    }

    msg::Sender get_sender() { return mOpReceiver; }

private:
    msg::Receiver mOpReceiver;
    unsigned mBalance;
};

class HardwareInterface {
public:
    void done() { get_sender().send(msg::CloseMessageQueue{}); }
    void run() {
        try {
            for (;;) {
                mOpReceiver.wait()
                    .handle<Money_ISSUING>([&](const Money_ISSUING &msg) {
                        std::scoped_lock lock(mMutex);
                        std::cout << "Issuing $" << msg.amount << std::endl;
                    })
                    .handle<Display_INSUFFICIENT_FUNDS>([&](const Display_INSUFFICIENT_FUNDS &msg) {
                        std::scoped_lock lock(mMutex);
                        std::cout << "Insufficient Founds!!" << std::endl;
                    })
                    .handle<Display_ENTER_PIN>([&](const Display_ENTER_PIN &msg) {
                        std::scoped_lock lock(mMutex);
                        std::cout << "Please Enter Your PIN (0-9)" << std::endl;
                    })
                    .handle<Display_INSERT_CARD>([&](const Display_INSERT_CARD &msg) {
                        std::scoped_lock lock(mMutex);
                        std::cout << "Please Insert Your Bank Card (I)" << std::endl;
                    })
                    .handle<Display_BALANCE>([&](const Display_BALANCE &msg) {
                        std::scoped_lock lock(mMutex);
                        std::cout << "Balance of Your Account is: $" << msg.balance << std::endl;
                    })
                    .handle<Display_WITHDRAWAL_OPTIONS>([&](const Display_WITHDRAWAL_OPTIONS &msg) {
                        std::scoped_lock lock(mMutex);
                        std::cout << "Withdraw 50? (w)" << std::endl;
                        std::cout << "Display Balance? (b)" << std::endl;
                        std::cout << "Cencel? (c)" << std::endl;
                    })
                    .handle<Display_WITHDRAWAL_CANCELED>(
                        [&](const Display_WITHDRAWAL_CANCELED &msg) {
                            std::scoped_lock lock(mMutex);
                            std::cout << "Withdraw Canceled." << std::endl;
                        })
                    .handle<Display_PIN_INCORRECT>([&](const Display_PIN_INCORRECT &msg) {
                        std::scoped_lock lock(mMutex);
                        std::cout << "PIN Incorrect." << std::endl;
                    })
                    .handle<Card_EJECTING>([&](const Card_EJECTING &msg) {
                        std::scoped_lock lock(mMutex);
                        std::cout << "Ejecting Card." << std::endl;
                    });
            }
        } catch (const msg::CloseMessageQueue &) { std::cout << "Hardware: closing" << std::endl; }
    }

    msg::Sender get_sender() { return mOpReceiver; }

private:
    msg::Receiver mOpReceiver;
    std::mutex mMutex;
};

int main() {
    Bank shitBank;
    HardwareInterface shitHarware;
    ATMMachine shitATM{shitBank.get_sender(), shitHarware.get_sender()};

    std::thread thread_bank{&Bank::run, std::ref(shitBank)};
    std::thread thread_intf{&HardwareInterface::run, std::ref(shitHarware)};
    std::thread thread_atm{&ATMMachine::run, std::ref(shitATM)};
    msg::Sender atmSender{shitATM.get_sender()};
    bool quit = false;

    while (!quit) {
        char input = getchar();
        if ('0' <= input && input <= '9') {
            atmSender.send(Digit_PRESSED{input});
        } else if (input == 'b' || input == 'B') {
            atmSender.send(Balance_PRESSED{});
        } else if (input == 'w' || input == 'W') {
            atmSender.send(Withdraw_PRESSED{50});
        } else if (input == 'c' || input == 'C') {
            atmSender.send(Cancel_PRESSED{});
        } else if (input == 'i' || input == 'I') {
            atmSender.send(Card_INSERTED("aac1234"));
        } else if (input == 'q' || input == 'Q') {
            quit = true;
        }
    }

    shitBank.done();
    shitATM.done();
    shitHarware.done();
    thread_atm.join();
    thread_bank.join();
    thread_intf.join();
}