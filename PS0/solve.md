## QUESTION 1
### (a)
$$
\nabla f(x) = Ax + b
$$
### (b)
$$
\nabla f(x) = g'(h(x)) \nabla_x h(x)
$$
### (c)
$$
\nabla^2 f(x) = A^T
$$
### (d)
$$
\begin{gather*}
    \nabla f(x) = g'(a^Tx)a \\
    \nabla^2 f(x) = g''(a^Tx)\ a a^T
\end{gather*}
$$

## QUESTION 2
### (a)
$$
\begin{gather*}
z \in \mathbb{R}^n \\
A = z z^T \\
x^TAx = x^Tz z^Tx = (z^Tx)z^Tx \ge 0
\end{gather*}
$$
### (b)
$$
\begin{gather*}
z \in \mathbb{R}^n \quad \text{is a non-zero vector}\\
A = z z^T \\
\empty(A) = \{ x \in \mathbb{R}: x^Tz = 0 \} \\
Rank(A) = 1
\end{gather*}
$$
### (c)
$$
\begin{gather*}
(BAB^T)^T = BA^TB^T = BAB^T \\
x^TBAB^Tx = (B^Tx)^TA(B^Tx) \ge 0
\end{gather*}
$$

## QUESTION 3
### (a)
$$
\begin{gather*}
A = T\Lambda T^{-1} \\
AT = T\Lambda
\end{gather*}
$$
### (b)
$$
\begin{gather*}
A = U\Lambda U^T \\
AU = U\Lambda
\end{gather*}
$$
### (c)
$$
\begin{gather*}
At^{(i)} = \lambda_it^{(i)} \\
t^{(i)T}At^{(i)} = \lambda_i\|t^{(i)}\|_2 = \lambda_i \geq 0
\end{gather*}
$$