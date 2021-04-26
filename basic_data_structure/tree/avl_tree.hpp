#ifndef __TREE_AVL_TREE_H__
#define __TREE_AVL_TREE_H__

#include "util.hpp"

#include <memory>
#include <iostream>
#include <string>
#include <cstdint>
#include <cassert>
#include <utility>
#include <algorithm>

namespace ds {

/**
 * AVL树（Adelson-Velsky and Landis Tree），是最早被发明的自平衡二叉查找树。AVL树中，任意节点对应的左右
 * 两棵子树的最大高度差都不超过1，因此它也被称为高度平衡树。其查找，插入和删除操作平均和最坏情况下的时间复杂
 * 度都是 O(logN)。与普通的二叉查找树（BST）相比，它多了一个关键的旋转操作。
 * https://zh.wikipedia.org/wiki/AVL%E6%A0%91
 */
template <typename T = int32_t>
class AVLTree {
public:
    AVLTree() : mRoot(nullptr), mSize(0) {}
    virtual ~AVLTree() = default;

    std::size_t size() const { return mSize; }
    bool empty() const { return mSize == 0; }

    void insert(T &v) {
        if (insert_to(mRoot, std::forward<T>(v))) {
            ++mSize;
        }
    }
    void insert(T &&v) {
        if (insert_to(mRoot, std::forward<T>(v))) {
            ++mSize;
        }
    }

    void remove(T &v) {
        assert(mSize > 0);
        if (remove_from(mRoot, std::forward<T>(v))) {
            --mSize;
        }
    }

    void remove(T &&v) {
        assert(mSize > 0);
        if (remove_from(mRoot, std::forward<T>(v))) {
            --mSize;
        }
    }

    bool contains(T &v) {
        return find_from(mRoot, std::forward<T>(v));
    }

    bool contains(T &&v) {
        return find_from(mRoot, std::forward<T>(v));
    }

    void display() const {
        std::cout << "AVL Tree size = " << size() << std::endl;
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

    // get sub-tree height (root = node)
    uint32_t get_height_from_node(const std::unique_ptr<Node> &node) const {
        if (!node) return 0;
        return 1 + std::max(get_height_from_node(node->left), get_height_from_node(node->right));
    }

    // get balance value, left height - right height
    int get_balance_value_from_node(const std::unique_ptr<Node> &node) const {
        return static_cast<int>(get_height_from_node(node->left))
                - static_cast<int>(get_height_from_node(node->right));
    }

    Node *find_min_node_from(std::unique_ptr<Node> &node) {
        if (!node) return nullptr;
        auto *cur = node.get();
        while (cur->left.get()) {
            cur = cur->left.get();
        }
        return cur;
    }

    /**
     * LL - case：向左子树插入左孩子导致了树不平衡，需要右旋
     *            y
     *           / \
     *          x   t4                         x
     *         / \      --right rotate-->    /  \
     *        z   t3                        z    y
     *       / \                          / \   / \
     *     t1  t2                       t1  t2 t3 t4
     */
    void rotate_right(std::unique_ptr<Node> &y) {
        auto x = std::move(y->left);
        y->left = std::move(x->right);
        x->right = std::move(y);
        y = std::move(x);
    }

    /**
     * RR - case ：向右子树插入右孩子导致树不平衡，需要左旋
     *       y
     *      / \
     *    t4   x                         x
     *        / \   --left rotate-->    /  \
     *      t3  z                      y    z
     *         / \                   / \   / \
     *        t1  t2                t4 t3 t1 t2
     */
    void rotate_left(std::unique_ptr<Node> &y) {
        auto x = std::move(y->right);
        y->right = std::move(x->left);
        x->left = std::move(y);
        y = std::move(x);
    }

    /**
     * LR（先左再右旋）：向左子树插入右孩子导致不平衡
     *  (8)  --insert-->     (10)            (10)
     *                       /               /
     *                     (7)             (7)
     *                                       \
     *                                       (8)
     * 
     *          y                                y
     *        /  \                              / \
     *       x    t4                           z  t4
     *      / \      --left rotate x-->       / \       (LL) then right rotate y
     *    t1  z                              x  t3
     *       / \                            / \
     *      t2 t3                         t1  t2
     * 
     * RL（先右旋再左旋）：向右子树插入左孩子导致不平衡
     *  (11)  --insert-->     (10)            (10)
     *                          \                \
     *                          (12)             (12)
     *                                           /
     *                                         (11)
     * 
     *          y                                y
     *        /  \                              / \
     *      t1    x                           t1   z
     *          / \      --right rotate x-->      / \    (RR) then left rotate y
     *         z  t4                            t2   x
     *       / \                                    / \
     *      t2 t3                                 t3  t4
     */

    void adjust_balance(std::unique_ptr<Node> &node) {
        if (!node) { return; }
        auto balance_value = get_balance_value_from_node(node);
        if (balance_value > 1) {
            if (get_balance_value_from_node(node->left) < 0) {
                // LR case, step 1: left rotate to LL
                rotate_left(node->left);
            }
            // LL case
            rotate_right(node);

        } else if (balance_value < -1) {
            if (get_balance_value_from_node(node->right) > 0) {
                // RL case, step 1: right rotate to RR
                rotate_right(node->right);
            }
            // RR case
            rotate_left(node);
        }
        // else, no need to adjust
    }

    bool insert_to(std::unique_ptr<Node> &node, T &&v) {
        if (!node) {
            // empty tree or leaf node
            node = std::make_unique<Node>(std::forward<T>(v));
            return true;
        }

        bool inserted = true;

        if (v < node->value) {
            // 比当前节点小，往左子树插入
            inserted = insert_to(node->left, std::forward<T>(v));
        } else if (v > node->value) {
            // 比当前节点大，往右子树插入
            inserted = insert_to(node->right, std::forward<T>(v));
        } else {
            std::cerr << "Insert: Element " << v << " already exist..." << std::endl;
            return false;
        }

        if (inserted) {
            adjust_balance(node);
        }
        return inserted;
    }

    bool remove_from(std::unique_ptr<Node> &node, T &&v) {
        if (!node) {
            std::cerr << "Remove: Element " << v << " doesn't exist" << std::endl;
            return false;
        }

        bool removed = false;

        if (v == node->value) {
            auto *right_min = find_min_node_from(node->right);
            if (!right_min) {
                // 无右子树，废弃当前节点并用当前节点左节点作为当前节点
                auto deprecated_node = std::move(node);
                node = std::move(deprecated_node->left);
                deprecated_node.reset(nullptr);
                removed = true;
            }
            if (!removed) {
                node->value = right_min->value;
                removed = remove_from(node->right, std::forward<T>(right_min->value));
            }
        } else if (v < node->value) {
            removed = remove_from(node->left, std::forward<T>(v));
        } else {
            removed = remove_from(node->right, std::forward<T>(v));
        }

        if (removed) {
            adjust_balance(node);
        }
        return removed;
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

static void Test_AVLTree() {
    std::cout << "==== " << __func__ << " =====\n\n";
    ds::AVLTree<int> tree;
    for (auto i = 0; i < 5; ++i) { tree.insert(i); }
    tree.insert(-1);
    tree.insert(-2);
    std::cout << "After inserted [0, 1, 2, 3, 4, -1, -2], current tree structure is:\n";
    std::cout << "          1\n";
    std::cout << "        /   \\\n";
    std::cout << "      -1     3\n";
    std::cout << "     /  \\  /  \\\n";
    std::cout << "   -2    0 2    4\n";
    tree.display();
    assert(tree.size() == 7);
    assert(tree.contains(1));
    assert(!tree.contains(5));
    tree.insert(1);
    assert(tree.size() == 7);
    tree.remove(-2);
    tree.remove(-1);
    tree.remove(0);
    std::cout << "After removed [-2, -1, 0], current tree structure is:\n";
    std::cout << "      3\n";
    std::cout << "    /   \\\n";
    std::cout << "   1     4\n";
    std::cout << "    \\\n";
    std::cout << "     2\n";  
    tree.display();
    assert(tree.size() == 4);
    assert(!tree.contains(-2));
    tree.remove(3);
    tree.remove(1);
    tree.remove(4);
    tree.remove(2);
    assert(tree.size() == 0);
    //tree.remove(2); // mute assertion error
    std::cout << "\n[OK]: Done.\n" << std::endl;
}

}  // namespace test

#endif  // __TREE_AVL_TREE_H__