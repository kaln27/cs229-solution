## QUESTION 4
### (b)
$$
\begin{align*}
\displaystyle
Q_i^{(t)}(z^{(i)} = j) &= p(z^{(i)}=j|x^{(i)};\mu_j,\Sigma_j,\phi_j) \\
&= \frac{p(x^{(i)},z^{(i)}=j;\mu_j,\Sigma_j,\phi_j)}{\sum_k p(x^{(i)},z^{(i)}=k;\mu_k,\Sigma_k,\phi_k)} \\
&= \frac{
    \frac{1}{(2 \pi)^{n/2}|\Sigma_j|^{1/2}}\exp(-\frac{1}{2}(x^{(i)}-\mu_j)^T\Sigma_j^{-1}(x^{(i)}-\mu_j)) \phi_j
}{
    \sum_k \frac{1}{(2 \pi)^{n/2}|\Sigma_k|^{1/2}}\exp(-\frac{1}{2}(x^{(i)}-\mu_k)^T\Sigma_k^{-1}(x^{(i)}-\mu_k))\phi_k
}
\end{align*}
$$
### (c)
$$
\begin{align*}
\displaystyle
w_j^{(i)} &:= Q_i^{(t)}(z^{(i)}=j) \\
\phi_j &= \frac{\sum_{i=1}^mw_j^{(i)}}{m} \\
\mu_j &= \frac{\sum_{i=1}^mw_j^{(i)}x^{(i)} + \alpha \sum_{i=1}^{\tilde{m}}\mathbb{I}\{\tilde{z}^{(i)}=j\}\tilde{x}^{(i)}}{\sum_{i=1}^mw_j^{(i)} + \alpha \sum_{i=1}^{\tilde{m}}\mathbb{I}\{\tilde{z}^{(i)}=j\}} \\
\Sigma_j &= \frac{
    \sum_{i=1}^mw_j^{(i)}(x^{(i)}-\mu_j)(x^{(i)}-\mu_j)^T + \alpha \sum_{i=1}^{\tilde{m}}\mathbb{I}\{\tilde{z}^{(i)}=j\} (\tilde{x}^{(i)}-\mu_j)(\tilde{x}^{(i)}-\mu_j)^T
}{
    \sum_{i=1}^mw_j^{(i)} + \alpha \sum_{i=1}^{\tilde{m}}\mathbb{I}\{\tilde{z}^{(i)}=j\}
}
\end{align*}
$$
