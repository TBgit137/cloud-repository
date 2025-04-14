class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BST:
    def __init__(self):
        self.root = None

    def preorder(self, root):
        if root:
            print(root.value)
            self.preorder(root.left)
            self.preorder(root.right)
    def inorder(self, root):
        if root:
            self.inorder(root.left)
            print(root.value)
            self.inorder(root.right)
    def postorder(self, root):
        if root:
            self.postorder(root.left)
            self.postorder(root.right)
            print(root.value)

    def insert(self, value):
        if not self.root:
            self.root = Node(value)
        else:
            self._insert(value, self.root)

    def _insert(self, value, current):
        if value < current.value:
            if current.left:
                self._insert(value, current.left)
            else:
                current.left = Node(value)
        else:
            if current.right:
                self._insert(value, current.right)
            else:
                current.right = Node(value)