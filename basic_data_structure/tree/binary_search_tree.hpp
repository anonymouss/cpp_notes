#ifndef __TREE_BST_TREE_H__
#define __TREE_BST_TREE_H__

#include "util.hpp"

#include <memory>
#include <iostream>
#include <string>
#include <cstdint>
#include <cassert>
#include <utility>

namespace ds {

/**
 * 二叉搜索树的简单实现 (为了简化实现，禁止重复元素)
 *
 * 二叉搜索树满足：
 *     - 任意节点的左子树（若不空）上所有节点值都比它小
 *     - 任意节点的右子树（若不空）上所有节点值都比它大
 *     - 任意节点的左右子树（若不空）也分别为二叉搜索树
 * 一般情况下，二叉搜索树的查找，插入，删除操作时间复杂度都是O(logN)的，但在最坏的情况下是O(N)的，即退化为单
 * 链表的情况。原因就是插入删除操作时没有保证树的平衡性，一种简单的改进就是AVL树。
 */
template <typename T = int32_t> // T must be comparable
class BinarySearchTree {
public:
    BinarySearchTree() : mRoot(nullptr), mSize(0) {}
    virtual ~BinarySearchTree() = default;

    std::size_t size() const { return mSize; }
    bool empty() const { return mSize == 0; }

    void insert(T &&v) {
        if (insert_to(mRoot, std::forward<T>(v))) {
            ++mSize;
        }
    }

    void remove(T &&v) {
        assert(mSize > 0);
        if (remove_from(mRoot, std::forward<T>(v))) {
            --mSize;
        }
    }

    bool contains(T &&v) {
        return find_from(mRoot, std::forward<T>(v));
    }

    void display() const {
        std::cout << "BST Tree size = " << size() << std::endl;
        TreePrintLevelOrder(mRoot.get());
        TreePrintPreOrder(mRoot.get());
        TreePrintInOrder(mRoot.get());
        TreePrintPostOrder(mRoot.get());
    }

private:
    struct Node {
        T value;
        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;
        explicit Node(T &&v) : value(std::forward<T>(v)), left(nullptr), right(nullptr) {}
        friend std::ostream &operator<<(std::ostream &os, const Node &node) {
            os << node.value;
            return os;
        }
    };

    std::unique_ptr<Node> mRoot;
    std::size_t mSize;

    Node *find_max_node_from(std::unique_ptr<Node> &node) {
        auto *cur = node.get();
        while (cur->right.get()) {
            cur = cur->right.get();
        }
        return cur;
    }

    Node *find_min_node_from(std::unique_ptr<Node> &node) {
        if (!node) return nullptr;
        auto *cur = node.get();
        while (cur->left.get()) {
            cur = cur->left.get();
        }
        return cur;
    }

    bool insert_to(std::unique_ptr<Node> &node, T &&v) {
        if (!node) {
            // 空树 or 已到叶子
            node = std::make_unique<Node>(std::forward<T>(v));
            return true;
        }

        if (v < node->value) {
            // 比当前节点小，往左子树插入
            return insert_to(node->left, std::forward<T>(v));
        } else if (v > node->value) {
            // 比当前节点大，往右子树插入
            return insert_to(node->right, std::forward<T>(v));
        } else {
            std::cerr << "Insert: Element " << v << " already exist..." << std::endl;
            return false;
        }
    }

    bool remove_from(std::unique_ptr<Node> &node, T &&v) {
        if (!node) {
            std::cerr << "Remove: Element " << v << " doesn't exist" << std::endl;
            return false;
        }

        if (v == node->value) {
            auto *right_min = find_min_node_from(node->right);
            if (!right_min) {
                // 无右子树，废弃当前节点并用当前节点左节点作为当前节点
                auto deprecated_node = std::move(node);
                node = std::move(deprecated_node->left);
                deprecated_node.reset(nullptr);
                return true;
            }
            node->value = right_min->value;
            return remove_from(node->right, std::forward<T>(right_min->value));
        } else if (v < node->value) {
            return remove_from(node->left, std::forward<T>(v));
        } else {
            return remove_from(node->right, std::forward<T>(v));
        }
    }

    bool find_from(std::unique_ptr<Node> &node, T &&v) {
        if (!node) return false;

        if (v == node->value) {
            return true;
        } else if (v < node->value) {
            return find_from(node->left, std::forward<T>(v));
        } else {
            return find_from(node->right, std::forward<T>(v));
        }
    }
};

}  // namespace ds

namespace test {

static void Test_BinarySearchTree() {
    std::cout << "==== " << __func__ << " =====\n\n";
    ds::BinarySearchTree<int> tree;
    tree.insert(3);
    tree.insert(1);
    tree.insert(5);
    tree.insert(0);
    tree.insert(2);
    tree.insert(4);
    tree.insert(6);
    std::cout << "After inserted [3, 1, 5, 0, 2, 4, 6], current tree structure is:\n";
    std::cout << "          3\n";
    std::cout << "        /   \\\n";
    std::cout << "       1     5\n";
    std::cout << "     /  \\  /  \\\n";
    std::cout << "    0    2 4    6\n";
    tree.display();
    assert(tree.size() == 7);
    assert(tree.contains(5));
    assert(!tree.contains(10));
    tree.remove(3);
    tree.remove(1);
    tree.remove(5);
    std::cout << "After removed [3, 1, 5], current tree structure is:\n";
    std::cout << "          4\n";
    std::cout << "        /   \\\n";
    std::cout << "       2     6\n";
    std::cout << "     /\n";
    std::cout << "    0\n";
    tree.display();
    assert(tree.size() == 4);
    assert(!tree.contains(5));
    assert(tree.contains(4));
    tree.remove(4);
    tree.remove(2);
    tree.remove(6);
    tree.remove(0);
    assert(tree.size() == 0);
    // tree.remove(1); // trigger assertion
    std::cout << "\n[OK]: Done.\n" << std::endl;
}

}  // namespace test

#endif  // __TREE_BST_TREE_H__
