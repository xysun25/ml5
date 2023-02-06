

import torch

# 在训练神经网络时，最常用的算法是反向传播。 在该算法中，参数（模型权重）根据损失函数相对于给定参数的梯度进行调整。
# 为了计算这些梯度，PyTorch 有一个名为 torch.autograd 的内置微分引擎。 它支持任何计算图的梯度自动计算。
# 考虑最简单的一层神经网络，输入 x、参数 w 和 b，以及一些损失函数。 它可以通过以下方式在 PyTorch 中定义：

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

# 在这个网络中，w 和 b 是我们需要优化的参数。
# 因此，我们需要能够计算损失函数相对于这些变量的梯度。
# 为此，我们设置了这些张量的 requires_grad 属性。

# 我们应用于张量以构建计算图的函数实际上是类 Function 的对象。
# 这个对象知道如何计算正向的函数，以及如何在反向传播步骤中计算它的导数。
# 对反向传播函数的引用存储在张量的 grad_fn 属性中。

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

# 计算梯度
# 为了优化神经网络中参数的权重，需要计算损失函数对参数的导数

loss.backward()
print(w.grad)
print(b.grad)

# 禁用梯度跟踪
# 默认情况下，所有 requires_grad=True 的张量都在跟踪它们的计算历史并支持梯度计算。
# 但是，在某些情况下我们不需要这样做，例如，当我们训练了模型并且只想将其应用于某些输入数据时，
# 即我们只想通过网络进行前向计算。 可以通过用 torch.no_grad()
# 块包围我们的计算代码来停止跟踪计算：

z = torch.matmul(x, w)+b
print(z.requires_grad)
with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)

# 实现相同结果的另一种方法是在张量上使用detach()方法：
# 将神经网络中的某些参数标记为冻结参数。这是微调预训练网络的一个非常常见的场景
# 当您只进行前向传递时加快计算速度，因为在不跟踪梯度的张量上进行计算会更有效。

z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)


# 张量梯度 和 雅可比矩阵
# 在许多情况下，我们有一个标量损失函数，我们需要计算一些参数的梯度。
# 但是，有些情况下输出函数是任意张量。
# 在这种情况下，PyTorch 允许您计算所谓的雅可比矩阵，而不是实际的梯度。
inp = torch.eye(5, requires_grad=True)
out = (inp+1).pow(2)
out.backward(torch.ones_like(inp), retain_graph=True)
print(f"First call\n{inp.grad}")
out.backward(torch.ones_like(inp), retain_graph=True)
print(f"\nSecond call\n{inp.grad}")
inp.grad.zero_()
out.backward(torch.ones_like(inp), retain_graph=True)
print(f"\nCall after zeroing gradients\n{inp.grad}")

