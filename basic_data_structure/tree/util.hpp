#ifndef __TREE_UTIL_H__
#define __TREE_UTIL_H__

#include <iostream>
#include <queue>
#include <stack>

namespace ds {

template <typename NodeType>
void TreePrintLevelOrder(const NodeType *const root) {
    std::cout << "[Level Order] ";
    if (!root) {
        std::cout << "Empty Tree.\n";
        return;
    }
    std::queue<const NodeType *const> queue;
    queue.emplace(root);
    while (!queue.empty()) {
        auto *cur = queue.front();
        queue.pop();
        std::cout << *cur << " ";
        if (cur->left) queue.emplace(cur->left.get());
        if (cur->right) queue.emplace(cur->right.get());
    }
    std::cout << std::endl;
}

template <typename NodeType>
void TreePrintPreOrder(const NodeType *const root) {
    std::cout << "[Pre   Order] ";
    if (!root) {
        std::cout << "Empty Tree.\n";
        return;
    }
    std::stack<const NodeType *const> stack;
    auto *cur = root;
    while (cur || !stack.empty()) {
        while (cur) {
            std::cout << *cur << " ";
            stack.emplace(cur);
            cur = cur->left.get();
        }
        if (!stack.empty()) {
            auto *top = stack.top();
            stack.pop();
            cur = top->right.get();
        }
    }
    std::cout << std::endl;
}

template <typename NodeType>
void TreePrintInOrder(const NodeType * const root) {
    std::cout << "[In    Order] ";
    if (!root) {
        std::cout << "Empty Tree.\n";
        return;
    }
    std::stack<const NodeType *const> stack;
    auto *cur = root;
    while (cur || !stack.empty()) {
        while (cur) {
            stack.emplace(cur);
            cur = cur->left.get();
        }
        if (!stack.empty()) {
            auto *top = stack.top();
            std::cout << *top << " ";
            stack.pop();
            cur = top->right.get();
        }
    }
    std::cout << std::endl;
}

template <typename NodeType>
void TreePrintPostOrder(const NodeType * const root) {
    std::cout << "[Post  Order] ";
    if (!root) {
        std::cout << "Empty Tree.\n";
        return;
    }
    std::stack<const NodeType *const> stack;
    auto *cur = root;
    const NodeType *last = nullptr;
    while (cur || !stack.empty()) {
        if (cur) {
            stack.emplace(cur);
            cur = cur->left.get();
        } else {
            auto *top = stack.top();
            if (top->right && top->right.get() != last) {
                cur = top->right.get();
            } else {
                std::cout << *top << " ";
                stack.pop();
                last = top;
            }
        }
    }
    std::cout << std::endl;
}

}  // namespace ds

#endif  // __TREE_UTIL_H__