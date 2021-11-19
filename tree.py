'''
Description: Binary Tree Traversal
Author: cinque
Date: 2021-09-02 11:03:31
LastEditTime: 2021-11-17 10:32:14
'''


"""
二叉树遍历:
    #* 先序遍历: 节点本身 --> 节点的左子树 --> 节点的右子树
    #? 中序遍历: 节点的左子树 --> 节点本身 --> 节点的右子树
    #! 后序遍历: 节点的左子树 --> 节点的右子树 --> 节点本身
    #* 层次遍历: 从左到右、从上到下遍历二叉树

example:
            D
           / \
          L   R

先序: DLR  中序: LDR  后序: LRD
"""


class TreeNode(object):
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


def CreateTreeRecursive(root, ls, i):
    """
    在层次遍历的列表中，对于索引为k的父节点
    其左子节点的索引为2k+1，右子节点的索引为2k+2
    """
    if i < len(ls):
        if ls[i] is None:
            return None
        else:
            root = TreeNode(ls[i])
            if 2*i+1 < len(ls):
                root.left = CreateTreeRecursive(root, ls, 2*i+1)
            else:
                root.left = None
            if 2*i+2 < len(ls):
                root.right = CreateTreeRecursive(root, ls, 2*i+2)
            else:
                root.right = None
            return root
    return root


def CreateTreeNonRecursive(ls):
    treelist = [TreeNode(item) if item != None else None for item in ls]
    for i,root in enumerate(treelist):
        if root and 2*i+1 < len(treelist):
            treelist[i].left = treelist[2*i+1]
        if root and 2*i+2 < len(treelist):
            treelist[i].right = treelist[2*i+2]
    return treelist[0]


def PreOrderTraversalRecursive(root):
    res = []
    def recursive(root, res):
        if root:
            res.append(root.val)
            recursive(root.left, res)
            recursive(root.right, res)
        return res
    return recursive(root, res)


def PreOrderTraversalNonRecursive(root):
    stack = [root]
    res = []
    while stack:
        node = stack.pop()
        if node:
            res.append(node.val)
            stack.append(node.right)
            stack.append(node.left)
    return res


def InOrderTraversalRecursive(root):
    res = []
    def recursive(root, res):
        if root:
            recursive(root.left, res)
            res.append(root.val)
            recursive(root.right, res)
        return res
    return recursive(root, res)


def InOrderTraversalNonRecursive(root):
    stack = [root]
    res = []
    while stack:
        #! 一直找左节点，直到找到最后一层的左叶子
        #! 并且每一层的左节点入栈
        while stack[-1]:
            stack.append(stack[-1].left)
        #! 此时栈顶必定为None
        #! 叶子的左/右None出栈
        stack.pop()
        if stack:
            node = stack.pop()
            res.append(node.val)
            #! 右节点入栈
            stack.append(node.right)
    return res


def PostOrderTraversalRecursive(root):
    res = []
    def recursive(root, res):
        if root:
            recursive(root.left, res)
            recursive(root.right, res)
            res.append(root.val)
        return res
    return recursive(root, res)


def PostOrderTraversalNonRecursive(root):
    stack = [root]
    res = []
    while stack:
        node = stack.pop()
        if node:
            res.append(node.val)
            stack.append(node.left)
            stack.append(node.right)
    return res[::-1]

def PostOrderTraversalItertive(root):
    if not root:
        return []
    res = []
    stack = []
    
    while stack or root:
        #! 一直遍历左节点，入栈两次，直至叶子节点
        while root:
            #! 第一次入栈为了保留该节点
            stack.append(root)
            #! 第二次入栈为了后续遍历其右子树
            stack.append(root)
            root = root.left
        #! 当前节点root
        root = stack.pop()
        #! 如果stack为空，则已经遍历完根节点的左右子树，最后添加根节点
        #! 如果stack中最后一个节点与当前节点不是同一节点
        #! 说明当前节点的左右子树已经遍历完，只需添加该节点
        if stack and stack[-1] == root:
            root = root.right
        else:
            res.append(root.val)
            root = None
    return res

def LevelTraversal(root):
    queue = [root]
    res = []
    while len(queue):
        node = queue.pop(0)
        res.append(node.val)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    return res
        

ls = [3,9,20,None,None,15,7]
Rtree = CreateTreeRecursive(None, ls, 0)
NRtree = CreateTreeNonRecursive(ls)

PreOrderTraversalRecursive(Rtree)
PreOrderTraversalRecursive(NRtree)

PreOrderTraversalNonRecursive(Rtree)
PreOrderTraversalNonRecursive(NRtree)

InOrderTraversalRecursive(Rtree)
InOrderTraversalRecursive(NRtree)

InOrderTraversalNonRecursive(Rtree)

PostOrderTraversalRecursive(Rtree)
PostOrderTraversalRecursive(NRtree)

PostOrderTraversalNonRecursive(Rtree)
PostOrderTraversalNonRecursive(NRtree)

LevelTraversal(Rtree)
LevelTraversal(NRtree)



def CreateTreeRecursive(ls, i=0):
    """
    在层次遍历的列表中，对于索引为k的父节点
    其左子节点的索引为2k+1，右子节点的索引为2k+2
    """
    if i < len(ls):
        if ls[i] is None:
            return None
        else:
            root = TreeNode(ls[i])
            if 2*i+1 < len(ls):
                root.left = CreateTreeRecursive(ls, 2*i+1)
            else:
                root.left = None
            if 2*i+2 < len(ls):
                root.right = CreateTreeRecursive(ls, 2*i+2)
            else:
                root.right = None
            return root

          
          
'''
Description: 
Author: cinque
Date: 2021-11-15 16:59:19
LastEditTime: 2021-11-18 17:54:08
'''

class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BinaryTreeBasic:
    def __init__(self, data, iterative=True):
        self.root = self.build(data, iterative=iterative)
    
    def build(self, data, iterative=True):
        _build = self._build_itertive if iterative else self._build_recursive
        return _build(data)
    
    def _build_itertive(self, data):
        """
        在层次遍历的列表中，对于索引为k的父节点
        其左子节点的索引为2k+1，右子节点的索引为2k+2
        """
        nodes = [TreeNode(item) if item != None else None for item in data]
        for i,root in enumerate(nodes):
            if root and 2*i+1 < len(nodes):
                nodes[i].left = nodes[2*i+1]
            if root and 2*i+2 < len(nodes):
                nodes[i].right = nodes[2*i+2]
        return nodes[0]
    
    def _build_recursive(self, data, i=0):
        """
        在层次遍历的列表中，对于索引为k的父节点
        其左子节点的索引为2k+1，右子节点的索引为2k+2
        """
        if i < len(data):
            if data[i] == None:
                return None
            else:
                root = TreeNode(data[i])
                if 2*i+1 < len(data):
                    root.left = self._build_recursive(data, 2*i+1)
                else:
                    root.left = None
                if 2*i+2 < len(data):
                    root.right = self._build_recursive(data, 2*i+2)
                else:
                    root.right = None
                return root

    def preorder(self, iterative=True):
        _preorder = self._preorder_itertive if iterative else self._preorder_recursive
        return _preorder(self.root)
    
    def _preorder_itertive(self, root):
        traveral = []
        stack = [root]
        while stack:
            node = stack.pop()
            if node:
                traveral.append(node.value)
                stack.append(node.right)
                stack.append(node.left)
        return traveral
    
    def _preorder_recursive(self, root):
        def recursive(root):
            if root:
                traveral.append(root.value)
                recursive(root.left)
                recursive(root.right)
            return traveral
        traveral = []
        recursive(root)
        return traveral
    
    def inorder(self, iterative=True):
        _inorder = self._inorder_itertive if iterative else self._inorder_recursive
        return _inorder(self.root)
    
    def _inorder_itertive(self, root):
        traversal = []
        stack = [root]
        while stack:
            #! 一直找左节点，直到找到最后一层的左叶子
            #! 并且每一层的左节点入栈
            while stack[-1]:
                #! 左节点入栈
                stack.append(stack[-1].left)
            #! 此时栈顶必定为None
            #! 叶子的左/右None出栈
            stack.pop()
            if stack:
                node = stack.pop()
                traversal.append(node.value)
                #! 右节点入栈
                stack.append(node.right)
        return traversal
    
    def _inorder_recursive(self, root):
        def recursive(root):
            if root:
                recursive(root.left)
                traversal.append(root.value)
                recursive(root.left)     
        traversal = []
        recursive(root)
        return traversal
    
    def postorder(self, iterative=True):
        _postorder = self._postorder_itertive if iterative else self._postorder_recursive
        return _postorder(self.root)
    
    def _postorder_itertive(self, root):
        traversal = []
        stack = []
        while root or stack:
            #! 一直遍历左节点，入栈两次，直至叶子节点
            while root:
                #! 第一次入栈为了保留该节点
                stack.append(root)
                #! 第二次入栈为了后续遍历其右子树
                stack.append(root)
                root = root.left
            #! 当前节点root
            root = stack.pop()
            #! 如果stack为空，则已经遍历完根节点的左右子树，最后添加根节点
            #! 如果stack中最后一个节点与当前节点不是同一节点
            #! 说明当前节点的左右子树已经遍历完，只需添加该节点            
            if not stack or root != stack[-1]:
                root = root.right
            else:
                traversal.append(root.value)
                root = None
        return traversal

    def _postorder_recursive(self, root):
        def recursive(root):
            if root:
                recursive(root.left)
                recursive(root.right)
                traversal.append(root.value)
        traversal = []
        recursive(root)
        return traversal

    def levelorder(self):
        traversal = []
        queue = [self.root]
        while queue:
            node = queue.pop(0)
            traversal.append(node.value)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        return traversal
    
    def Depth(self, iterative=True):
        _Depth = self._Depth_itertive if iterative else self._Depth_recursive
        return _Depth(self.root)
    
    def _Depth_recursive(self, root):
        if not root:
            return 0
        return max(self._Depth_recursive(root.left), self._Depth_recursive(root.right)) + 1
    
    def _Depth_itertive(self, root):
        pass
      
      
      
      
'''
Description: 
Author: cinque
Date: 2021-11-11 11:46:44
LastEditTime: 2021-11-12 17:45:03
'''

class TreeNode(object):
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def FindMin(root):
    cur = root
    while cur:
        if cur.left:
            cur = cur.left
        else:
            return cur.value

def FindMax(root):
    cur = root
    while cur:
        if cur.right:
            cur = cur.right
        else:
            return cur.value

def InsertRecursive(root, value):
    if root == None:
        root = TreeNode(value)
    else:
        if value < root.value:
            root.left = InsertRecursive(root.left, value)
        elif value > root.value:
            root.right = InsertRecursive(root.right, value)
    return root

def InsertNonRecursive(root, value):
    cur = root
    parent = None
    
    while cur:
        parent = cur
        if cur.value < value:
            cur = cur.right
        elif cur.value > value:
            cur = cur.left
        else:
            break
    
    if parent:
        if parent.value < value:
            parent.right = TreeNode(value)
        elif parent.value > value:
            parent.left = TreeNode(value)
    else:
        root = TreeNode(value)
    
    return root

def InsertNonRecursive(root, value):
    if root is None:
        root = TreeNode(value)
    else:
        cur = root
        while cur:
            parent = cur
            if value < parent.value:
                cur = parent.left
            elif value > parent.value:
                cur = parent.right
        if value < parent.value:
            parent.left = TreeNode(value)
        elif value > parent.value:
            parent.right = TreeNode(value)
    return root

def SearchRecursive(root, value):
    if root == None:
        return False
    if root.value == value:
        return True
    elif root.value < value:
        return SearchRecursive(root.right, value)
    else:
        return SearchRecursive(root.left, value)

def SearchNonRecursive(root, value):
    cur = root
    while cur:
        if cur.value == value:
            return True
        elif cur.value < value:
            cur = cur.right
        else:
            cur = cur.left
        
    if cur == None:
        return False

def DeleteRecursive(root, value):
    """
    #!  待删除节点为叶子节点
        判断待删除节点的左右子树是否为空，如果是则为叶子节点
        判断待删除节点是其父节点的左子树还是右子树，将对应子树赋为None

    #!  待删除节点只有左子树或右子树
        将待删除节点的左/右子树赋给父节点的左/右子树
    
    #!  待删除节点有左子树和右子树
        将待删除节点的右子树的最小值替换待删除节点
        删除该右子树中的最小值
    """
    if root is None:
        return
    # 当前节点
    cur = root
    while cur and cur.value != value:
        # 记录父节点
        father = cur
        if cur.value < value:
            cur = cur.right
        elif cur.value > value:
            cur = cur.left
    # 要删除的节点为叶子节点
    if cur.left is None and cur.right is None:
        if father.left == cur:
            father.left = None
        else:
            father.right = None
    # 要删除的节点有左子树和右子树
    elif cur.left and cur.right:
        # 把该节点右子树中最小的节点代替要删除的节点
        min_ = FindMin(cur.right)
        cur.value = min_
        # 删除右子树的最小节点
        DeleteRecursive(cur.right, min_)
    else:
        # 要删除的节点为左节点
        if father.left == cur:
            if cur.left:
                father.left = cur.left
            else:
                father.left = cur.right
        else:
            if cur.left:
                father.right = cur.left
            else:
                father.right = cur.right

def DeleteNonRecursive(root, value):
    cur = root
    parent = None
    
    while cur and cur.value != value:
        parent = cur
        if cur.value < value:
            cur = cur.right
        elif cur.value > value:
            cur = cur.left
    
    if cur == None:
        print(f"value {value} not found in the tree")
        return root
    
    if cur.left == None or cur.right == None:
        new_cur = None
        
        if cur.left == None and cur.right:
            new_cur = cur.right
        elif cur.right == None and cur.left:
            new_cur = cur.left
        
        if cur == parent.left:
            parent.left = new_cur
        else:
            parent.right = new_cur
    else:
        prev = None
        min_node = cur.right
        
        while min_node.left:
            prev = min_node
            min_node = min_node.left
            
        if prev:
            prev.left = min_node.right
        else:
            cur.right = min_node.right
        
        cur.value = min_node.value
    
    return root


data = [17, 5, 35, 2, 11, 29, 38, 9, 7, 16, 8]
root = None
for i in data:
    root = InsertRecursive(root, i)

value = 5
DeleteRecursive(root, value)
SearchRecursive(root, value)




'''
Description: 
Author: cinque
Date: 2021-11-15 10:13:36
LastEditTime: 2021-11-17 16:58:44
'''

class TreeNode:
    def __init__(self, value):
        self.value = value
        self.height = 0
        self.left = None
        self.right = None
    
def depth(root):
    if not root:
        return 0
    return max(depth(root.left), depth(root.right)) + 1

def balance_factor(node):
    return depth(node.left) - depth(node.right)

def is_balanced(node):
    if not node:
        return True
    balancefactor = balance_factor(node)
    if abs(balancefactor) > 1:
        return False
    return is_balanced(node.left) and is_balanced(node.right)

def FindMin(root):
    cur = root
    while cur:
        if cur.left:
            cur = cur.left
        else:
            return cur.value

def FindMax(root):
    cur = root
    while cur:
        if cur.right:
            cur = cur.right
        else:
            return cur.value

def InsertRecursive(root, value):
    if root == None:
        root = TreeNode(value)
    else:
        if value < root.value:
            root.left = InsertRecursive(root.left, value)
        elif value > root.value:
            root.right = InsertRecursive(root.right, value)
    return root

def SearchRecursive(root, value):
    if root == None:
        return False
    if root.value == value:
        return True
    elif root.value < value:
        return SearchRecursive(root.right, value)
    else:
        return SearchRecursive(root.left, value)
    
def leftRotate(z):
    """
      z                                y
     /  \                            /   \ 
    T1   y     Left Rotate(z)       z      x
        /  \   - - - - - - - ->    / \    / \ 
       T2   x                     T1  T2 T3  T4
           / \ 
         T3  T4    
    """
    y = z.right
    T2 = y.left

    y.left = z
    z.right = T2

    return y

def rightRotate(z):
    """
    T1, T2, T3 and T4 are subtrees.
           z                                      y 
          / \                                   /   \ 
         y   T4      Right Rotate (z)          x      z
        / \          - - - - - - - - ->      /  \    /  \ 
       x   T3                               T1  T2  T3  T4
      / \ 
    T1   T2    
    """
    y = z.left
    T3 = y.right

    y.right = z
    z.left = T3

    return y

def rotate(root):
    if not root:
        return root
    
    bf = balance_factor(root)

    if bf > 1:
        left_bf = balance_factor(root.left)
        if left_bf < 0:
            root.left = leftRotate(root.left)
        return rightRotate(root)
    elif bf < -1:
        right_bf = balance_factor(root.right)
        if right_bf > 0:
            root.right = rightRotate(root.right)
        return leftRotate(root)
    else:
        return root

def insert1(root, value):
    if not root:
        return TreeNode(value)
    
    if value < root.value:
        root.left = insert1(root.left, value)
    elif value > root.value:
        root.right = insert1(root.right, value)
        
    return rotate(root)
    
def insert(root, value):
    if not root:
        return TreeNode(value)
    if value < root.value:
        root.left = insert(root.left, value)
    elif value > root.value:
        root.right = insert(root.right, value)

    bf = balance_factor(root)
    
    if bf > 1 and value < root.left.value:
        return rightRotate(root)
    
    if bf < -1 and value > root.right.value:
        return leftRotate(root)
    
    if bf > 1 and value > root.left.value:
        root.left = leftRotate(root.left)
        return rightRotate(root)
    
    if bf < -1 and value < root.right.value:
        root.right = rightRotate(root.right)
        return leftRotate(root)
    
    return root

def get_min_node(root):
    if root == None or root.left == None:
        return root
    return get_min_node(root.left)

def delete(root, value):
    if not root:
        return root
    
    if value < root.value:
        root.left = delete(root.left, value)
    elif value > root.value:
        root.right = delete(root.right, value)
    else:
        if root.left == None:
            temp = root.right
            root = None
            return temp
        elif root.right == None:
            temp = root.left
            root = None
            return temp
        
        temp = get_min_node(root.right)
        root.value = temp.value
        root.right = delete(root.right, temp.value)
        
    if not root:
        return root

    return rotate(root)


def InOrderTraversalNonRecursive(root):
    stack = [root]
    res = []
    while stack:
        #! 一直找左节点，直到找到最后一层的左叶子
        #! 并且每一层的左节点入栈
        while stack[-1]:
            stack.append(stack[-1].left)
        #! 此时栈顶必定为None
        #! 叶子的左/右None出栈
        stack.pop()
        if stack:
            node = stack.pop()
            res.append(node.value)
            #! 右节点入栈
            stack.append(node.right)
    return res

def PreOrderTraversalNonRecursive(root):
    stack = [root]
    res = []
    while stack:
        node = stack.pop()
        if node:
            res.append(node.value)
            stack.append(node.right)
            stack.append(node.left)
    return res

def LevelTraversal(root):
    queue = [root]
    res = []
    while len(queue):
        node = queue.pop(0)
        res.append(node.value)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    return res

root = None
ls = [9, 5, 10, 0, 6, 11, -1, 1, 2]
for item in ls:
    root = insert1(root, item)

InOrderTraversalNonRecursive(root)
PreOrderTraversalNonRecursive(root)
LevelTraversal(root)












"""
平衡因子 = 左子树高度 - 右子树高度
当某棵树的平衡因子：

1、大于1，则代表需要右旋操作，如果此时其左子树的平衡因子小于0，则代表需要先左旋；
2、小于-1，则代表需要左旋操作，如果此时其右子树的平衡因子大于0，则代表需要先右旋。


插入一个节点后可能会破坏AVL树的平衡，第一个失衡的节点记为K
K的两棵子树的高度差为2:
    1、对K的左儿子的左子树进行一次插入  (LL  -->  右旋)
    2、对K的左儿子的右子树进行一次插入  (LR  -->  左旋->右旋)
    3、对K的右儿子的左子树进行一次插入  (RL  -->  右旋->左旋)
    4、对K的右儿子的右子树进行一次插入  (RR  -->  左旋)

   X > Y > Z

1、对K的左儿子的左子树进行一次插入

        X (2)
       /                  Y
      Y (1)     ---->    / \
     /                  Z   X
    Z (0)

2、对K的左儿子的右子树进行一次插入

      X (2)               X (2)
     /                   /                  Y
    Z (-1)     ---->    Y (1)     ---->    / \
     \                 /                  Z   X
      Y (0)           Z (0)

3、对K的右儿子的左子树进行一次插入

    Z (-2)            Z (-2)
     \                 \                    Y
      X (1)    ---->    Y (-1)    ---->    / \
     /                   \                Z   X
    Y (0)                 X (0)

4、对K的右儿子的右子树进行一次插入

    Z (-2)
     \                   Y
      Y (-1)   ---->    / \
       \               Z   X
        X (0)


1、对K的左儿子的左子树进行一次插入

        X (2)
       /                  Y
      Y (1)     ---->    / \
     /                  Z   X
    Z (0)
"""




