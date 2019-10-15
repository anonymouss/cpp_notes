# 感知机模型

**模型**：

$f(x) = sign(w \cdot x + b)$

**策略**：

极小化损失函数 $\min_{w,b}L(w,b) = -\sum_{x_i\subseteq M}{y_i(w\cdot x_i + b)}$

**求解算法**

1. 原始形式

    1. 选取初值 $w_0$，$b_0$

    2. 从训练集选取数据 $(x_i, y_i)$

    3. 误分判断：若$y_i(w\cdot x_i + b) \le 0$，更新：$$\begin{align*} w \leftarrow& w + \eta y_i x_i\\ b \leftarrow& b + \eta y_i \end{align*}$$

    4. 跳转到 2. 继续循环，直到没有误分点。

2. 对偶形式 $f(x) = sign\left\{ \sum_{j=1}^N{\alpha_j y_j x_j\cdot x+b} \right\}$

    1. 选取初值 $\alpha \leftarrow 0, b \leftarrow 0, \eta \leftarrow 1$

    2. 从训练集选取数据 $(x_i, y_i)$

    3. 误分判断：若$y_i\left(\sum_{j=1}^N{\alpha_jy_jx_j\cdot x_i + b} \right ) \le 0$，更新：$$\begin{align*} \alpha_i \leftarrow& \alpha_i + \eta\\ b \leftarrow& b + \eta y_i \end{align*}$$

    4. 跳转到 2. 继续循环，直到没有误分点。

    5. 最后计算$w = \sum_{i=1}^N{\alpha_iy_ix_i}$ 

    * Gram Maxtrix: $G = [x_i\cdot x_j]_{N\times N}$