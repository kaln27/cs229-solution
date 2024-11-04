## QUESTION 1
In file [p01_nn.py](src/p01_nn.py)
## QUESTION 2
### (a)
$$
\begin{gather*}
\mathbb{E}_{\substack{s \sim p(s) \\ a \sim \pi_0(s, a)}} \frac{\pi_1(s, a)}{\hat{\pi}_0(s, a)}R(s, a) \\
= \int_s \int_a \frac{\pi_1(s, a)}{\hat{\pi}_0(s, a)}R(s, a)p(s, a) ds da \\
= \int_s \int_a \frac{\pi_1(s, a)}{\hat{\pi}_0(s, a)}R(s, a)p(s) \pi_0(s, a) ds da \\
= \int_s \int_a \pi_1(s, a)R(s, a)p(s) ds da \\
= \mathbb{E}_{\substack{s \sim p(s) \\ a \sim \pi_1(s, a)}} R(s, a)
\end{gather*}
$$
### (b)
$$
\begin{gather*}
    \frac{\mathbb{E}_{\substack{s \sim p(s) \\ a \sim \pi_0(s, a)}} \frac{\pi_1(s, a)}{\hat{\pi}_0(s, a)}R(s, a) }{\mathbb{E}_{\substack{s \sim p(s) \\ a \sim \pi_0(s, a)}} \frac{\pi_1(s, a)}{\hat{\pi}_0(s, a)}} \\
    = \frac{\mathbb{E}_{\substack{s \sim p(s) \\ a \sim \pi_1(s, a)}} R(s, a)}{\int_s \int_a \pi_1(s, a)p(s) ds da} \\
    = \mathbb{E}_{\substack{s \sim p(s) \\ a \sim \pi_1(s, a)}} R(s, a)
\end{gather*}
$$
### (c)
$$
\begin{gather*}
    \frac{\mathbb{E}_{\substack{s \sim p(s) \\ a \sim \pi_0(s, a)}} \frac{\pi_1(s, a)}{\hat{\pi}_0(s, a)}R(s, a) }{\mathbb{E}_{\substack{s \sim p(s) \\ a \sim \pi_0(s, a)}} \frac{\pi_1(s, a)}{\hat{\pi}_0(s, a)}} \\
    = \frac{\iint \frac{\pi_1(s, a)}{\hat{\pi}_0(s, a)}R(s, a) p(s,a) dsda}{\iint \frac{\pi_1(s, a)}{\hat{\pi}_0(s, a)} p(s,a) dsda} \\
    = R(s_0,a_0) \quad \text{if there is only one exapmle($s_0, a_0$)} \\
    \text{if $\pi_0 \neq \pi_1$ then given $s_0$, $\pi_1$ will not generate $a_0$} \\
     \Rightarrow \mathbb{E}_{\substack{s \sim p(s) \\ a \sim \pi_0(s, a)}} R(s, a) \neq \mathbb{E}_{\substack{s \sim p(s) \\ a \sim \pi_1(s, a)}} R(s, a)
\end{gather*}
$$
### (d)
#### (i)
$$
\begin{gather*}
    \mathbb{E}_{\substack{s \sim p(s) \\ a \sim \pi_0(s, a)}} \mathbb{E}_{a \sim \pi_1(s,a)}\hat{R}(s,a) = \mathbb{E}_{\substack{s \sim p(s) \\ a \sim \pi_1(s, a)}} R(s, a) \\
    \mathbb{E}_{\substack{s \sim p(s) \\ a \sim \pi_0(s, a)}} ((\mathbb{E}_{a \sim \pi_1(s,a)}\hat{R}(s,a)) + \frac{\pi_1(s, a)}{\hat{\pi}_0(s, a)}(R(s, a) - \hat{R}(s,a))) \\
     = \mathbb{E}_{\substack{s \sim p(s) \\ a \sim \pi_1(s, a)}} R(s, a)    
\end{gather*}
$$
#### (ii)
$$
\begin{gather*}
    \mathbb{E}_{\substack{s \sim p(s) \\ a \sim \pi_0(s, a)}} ((\mathbb{E}_{a \sim \pi_1(s,a)}\hat{R}(s,a)) + \frac{\pi_1(s, a)}{\hat{\pi}_0(s, a)}(R(s, a) - \hat{R}(s,a))) = \mathbb{E}_{\substack{s \sim p(s) \\ a \sim \pi_0(s, a)}} \mathbb{E}_{a \sim \pi_1(s,a)}\hat{R}(s,a) \\
    = \mathbb{E}_{\substack{s \sim p(s) \\ a \sim \pi_1(s, a)}} R(s, a)  
\end{gather*}
$$
### (e)
#### (i)
In this situation, we will use **importance sampling estimator** to check whether a policy is good or not. Cause It's easy to sample data, and easy to estimate $\hat{\pi}_0$
#### (ii)
In this situation, we will use **regression estimator** to check whether a policy is good or not. Cause $\hat{R}(s,a)$ is easy to estimate, and more accuracy.

## QUESTION 3
$$
\begin{gather*}
\mathcal{V} = \{\alpha u : \alpha \in \mathbb{R}\} \\
f_u(x) = arg \ \underset{v \in \mathcal{V}}{min} \| x - v \|_2^2 \\
let \ f_u(x) = v \quad x^T u = \|u\|_2 \|v\|_2 \Rightarrow \|v\|_2 = x^T u \\
\because v = \alpha u \Rightarrow \|v\|_2 = \alpha \|u\|_2 = \alpha \\
\therefore v = x^T u u \\
f_u(x) = x^T u u
\end{gather*}
$$

$$
\begin{align*}

\mathcal{L} &= arg \ \underset{u:u^T u = 1}{min} \sum_{i=1}^{m} \|x^{(i)} - f_u(x^{(i)})\|_2 \\
&=  arg \ \underset{u:u^T u = 1}{min} \sum_{i=1}^{m} \| x^{(i)} - x^{(i)T} u u \|_2 \\
&= arg \ \underset{u:u^T u = 1}{min} \sum_{i=1}^{m} (x^{(i)} - x^{(i)T} u u)^T (x^{(i)} - x^{(i)T} u u)\\
&= arg \ \underset{u:u^T u = 1}{min} \sum_{i=1}^{m} x^{(i)T}x^{(i)} - u^T x^{(i)}x^{(i)T} u \\
&= arg \ \underset{u:u^T u = 1}{min} 1 - u^T (\sum_{i=1}^{m} x^{(i)}x^{(i)T}) u \\
&= arg \ \underset{u:u^T u = 1}{max} u^T (\sum_{i=1}^{m} x^{(i)}x^{(i)T}) u  \quad \text{same as textbook one}
\end{align*}
$$

## QUESTION 4
In file [p04_ica.py](src/p04_ica.py)
### (a)
$$
\begin{gather*}
s_j \sim \mathcal{N}(0, 1) \\
\ell(W) = \sum_{i=1}^n \left( log|W| + \sum_{j=1}^d\log g'(w_j^Tx^{(i)}) \right) \\
\nabla_W \ell(W) = \sum_{i=1}^n \left( 
    \begin{bmatrix} g''(w_1^Tx^{(i)}) / g'(w_1^Tx^{(i)}) \\ g''(w_2^Tx^{(i)}) / g'(w_2^Tx^{(i)}) \\ \vdots \\ g''(w_d^Tx^{(i)}) / g'(w_d^Tx^{(i)}) \end{bmatrix} x^{(i)T} + (W^T)^{-1}
\right) \\
\text{Cause $g'(x)$ is normal distribution ($\mu = 0,\ \sigma^2 = 1$) here we have:} \\
g''(x) / g'(x) = \frac{\frac{1}{\sqrt{2\pi}}\exp(-\frac{1}{2}x^2) -x}{\frac{1}{\sqrt{2\pi}}\exp(-\frac{1}{2}x^2)} = -x \\
\nabla_W \ell(W) = \sum_{i=1}^n \left( 
    \begin{bmatrix} -w_1^Tx^{(i)} \\ -w_2^Tx^{(i)} \\ \vdots \\ -w_d^Tx^{(i)}  \end{bmatrix} x^{(i)T} + (W^T)^{-1}
\right) = \sum_{i=1}^n \left( -W x^{(i)}x^{(i)T} + (W^T)^{-1} \right) = 0\\
W \sum_{i=1}^n x^{(i)}x^{(i)T} = n (W^T)^{-1} \\
\frac{1}{n} \sum_{i=1}^n x^{(i)}x^{(i)T} = (W^T W)^{-1} \\
\text{if $W' = RW$ \ where ($R R^T = R^T R = I$) $R$ is a orthigonal matrix:}\\
\frac{1}{n} \sum_{i=1}^n x^{(i)}x^{(i)T} = ((RW)^T(RW))^{-1} \\
\frac{1}{n} \sum_{i=1}^n x^{(i)}x^{(i)T} = (W^T R^T RW)^{-1} \\
\frac{1}{n} \sum_{i=1}^n x^{(i)}x^{(i)T} = (W^TW)^{-1} \\
\text{There is no way to distinguish $W$ and $W'$}
\end{gather*}
$$
### (b)
$$
\begin{gather*}
s_i \sim \mathcal{L}(0, 1) \\
f_{\mathcal{L}}(s) = \frac{1}{2}\exp(-|s|) \\
\nabla_W \ell(W) = \begin{bmatrix} g''(w_1^Tx^{(i)}) / g'(w_1^Tx^{(i)}) \\ g''(w_2^Tx^{(i)}) / g'(w_2^Tx^{(i)}) \\ \vdots \\ g''(w_d^Tx^{(i)}) / g'(w_d^Tx^{(i)}) \end{bmatrix} x^{(i)T} + (W^T)^{-1} \\
g'(x) = \frac{1}{2}\exp(-|s|) \\
g''(x) = -\frac{1}{2} \text{sign}(x) \exp(-|s|) \\
g''(x) / g'(x) = -\text{sign}(x) \\
W := W + \alpha \left( -\begin{bmatrix} \text{sign}(w_1^Tx^{(i)}) \\ \text{sign}(w_2^Tx^{(i)}) \\ \vdots \\ \text{sign}(w_d^Tx^{(i)}) \end{bmatrix} x^{(i)T} + (W^T)^{-1} \right) \\
W := W + \alpha \left( -\text{sign}(W x^{(i)}) x^{(i)T} + (W^T)^{-1} \right)
\end{gather*}
$$

## QEUSTION 5
### (a)
$$
\begin{gather*}
V'(s) = R(s) + \gamma \ \underset{a \in \mathcal{A}}{\max} \sum_{s' \in \mathcal{S}} P_{sa}(s')V(s') \\
\| B(V_1) - B(V_2) \|_{\infty} \le \gamma \|V_1 - V_2 \|_{\infty} \\
\| \gamma \ \underset{a \in \mathcal{A}}{\max} \sum_{s' \in \mathcal{S}} P_{sa}(s') (V_1(s') - V_2(s'))\|_{\infty} \le \gamma \ \underset{s \in \mathcal{S}}{max} |V_1(s) - V_2(s)| = \gamma \| V_1 - V_2\|_{\infty}
\end{gather*}
$$
### (b)

$$
\begin{gather*}
\text{if there exist 2 fixed point $V_1, V_2$ s.t.} \\
B(V_1) = V_1 \quad B(V_2) = V_2 \\
\| B(V_1) - B(V_2) \|_{\infty} \le \gamma \ \|V_1 - V_2\|_{\infty} \\
\| V_1 - V_2 \|_{\infty} \le \gamma \ \|V_1 - V_2\|_{\infty} \\
1 \le \gamma \quad \text{that invalid cause $\gamma \lt 1$}
\end{gather*}
$$
## QUESTION 6
