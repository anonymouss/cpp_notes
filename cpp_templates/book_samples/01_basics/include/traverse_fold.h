#ifndef __TRAVERSE_FOLD_H__
#define __TRAVERSE_FOLD_H__

struct Node {
    int value;
    Node *left, *right;
    Node(int i = 0) : value(i), left(nullptr), right(nullptr) {}
};

auto left = &Node::left;
auto right = &Node::right;

template <typename T, typename... TP>
Node *traverse(T np, TP... paths) {
    return (np->*...->*paths);  // (np->*paths1)->*paths2 ...
}

#endif  // __TRAVERSE_FOLD_H__