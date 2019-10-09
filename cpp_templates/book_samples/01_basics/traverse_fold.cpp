#include "traverse_fold.h"

int main() {
    Node *root = new Node(0);
    root->left = new Node(1);
    root->left->right =  new Node(2);

    Node *node = traverse(root, left, right);
}