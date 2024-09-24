一种mutate的方法
其中$w_{max}$ 和 $w_{min}$是每层神经网络中的最大值和最小值 $$
$\mathcal{N}$是正态分布
$$
\begin{align*} &\sigma=\frac{w_{max}- w_{min}}{6}\tag{1}
\\ &w^{\prime}=w+\mathcal{N}(0, \sigma)\tag{2}\end{align*}
$$
source: DOI: 10.1109/e-Science58273.2023.10254878

---
```
#init the global best particle, g
#randomly init the local particle
for each p in S
    evaluate the fitness
    update the velocity of p using PSO
    update the position of p using PSO

select the top k particels from S, using them to produce the new particles using GA

add the new particles to S
update the global best
```

source: DOI: 10.1109/ICM60448.2023.10378946

crossover 可以交换整个neural layer  
也可以交换特定位置的parameter

---


```code
if the loss is lower than some value, it will enter the genetic evolution
cross w_k with randomly selected other parameters in the repository
with some certain probability, cross w_k with global model
With the limitation of the r_mutation probability, the mutated model parameters obtained from the crossover are perturbed by adding a disturbance with mean 0 and variance stddev to only p_mutation of the individual
```

source: DOI: 10.1109/CIPAE60493.2023.00114

---

```code
randomly select two elements to produce higher elemen
```
source: DOI: 10.1109/ACCESS.2023.3304368

---
这里的GA操作很简单  
source: DOI: 10.1109/TNSE.2024.3391613