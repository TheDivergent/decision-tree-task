# 练习

1.假设有一个3分类问题，标签类别为第2类，模型输出的类别标签为[-0.1,-0.3,0.4]，请计算对应的指数损失。		

答：$ y=[-\frac{1}{2},1,-\frac{1}{2}]$​​,$f=[-0.1,-0.3,0.4]$​​

​				$L(y,f)=exp(-\frac{y^Tf}{K})=e^{(-(0.5*0.1-0.3-0.2)/3)}=e^{0.15}$​​

2.左侧公式的第二个等号是由于当样本分类正确时，$y^Tb=\frac{K}{K-1}$​​，当样本分类错误时，$y^Tb=-\frac{K}{(K-1)^2}$，请说明原因。

答：当分类正确时,

​	$y^Tb=1+\frac{1}{(K-1)^2}*(K-1)=\frac{K}{K-1}$

当样本分类错误时:

$y^Tb=2*(-\frac{1}{K-1})+(K-2)*(\frac{1}{(K-1)^2})=-\frac{K}{(K-1)^2}$



3.对公式进行化简，写出K=2时的SAMME算法流程，并与李航《统计学习方法》一书中所述的Adaboost二分类算法对比是否一致。

​	答:对于$L(\beta^{(m)},b^{(m)})=\sum_{i=1}^{n}w_ie^{-\frac{1}{K-1}\beta^{(m)}}+[e^{\frac{1}{(K-1)^2}\beta^{(m)}}-e^{-\frac{1}{K-1}\beta{{(m))}}}]\sum_{i\notin{T}}w_i$。则$K=2$条件下可以化简为，$L(\beta^{(m)},b^{(m)})=\sum_{i=1}^{n}w_ie^{-\beta^{(m)}}+[e^{\beta^{(m)}}-e^{-\beta{{(m))}}}]\sum_{i\notin{T}}w_i$，$b^{\*(m)}=argmin\sum_{i=1}^nw_iI(i\notin{T})$ ,其中$I(x)$为示性函数，满足条件时取1，其他情况下取0.计算得到第m轮$\beta$的最优值为$\beta^{\*(m)}=\frac{1}{2}ln(\frac{1-err}{err})$,其中$err=\frac{\sum_{i\notin{T}}w_i}{\sum_{i=1}^nw_i}$​.概论权重的更新值为

![image-20211103201251994](C:\Users\86188\AppData\Roaming\Typora\typora-user-images\image-20211103201251994.png).

经过对比会发现，与李航《统计学习方法》中的一致。

算法流程如下：

***

Algorithm 1:Adaboost 方法的SAMME实现（二分类版本）

***

Data：训练样本$X=(x_1,...,x_n)$和$y=(y_1,...,y_n)$，基分类器G、迭代轮数M、测试样本x

Result：测试样本的预测类别$c(x)$

1. **for** $i\leftarrow1 $​​ **to** $n$​ ​**do**  ​​
2. |  $w_i\leftarrow \frac{1}{n}$
3. **end**
4. **for** $m\leftarrow 1$​​​ **to** $M$​​ **do**
5.  $ G^\*\leftarrow argmin_G\sum_{i=1}^nwi\prod_{\{i\notin{T}\}}$​​​
6. |    $err^{(m)} \leftarrow \sum_{i=1}^n\frac{w_i\prod_{\{i\notin{T}\}}}{\sum_{i=1}^nw_i}$​
7. |    $\beta^{\*(m)}\leftarrow \frac{1}{2}log\frac{1-err^{(m)}}{err^{(m)}}$​​​​​
8. |    **for** $i\leftarrow1$​​ **to** n **do**
9. |       | $b^{\*(m)}(x_i) \leftarrow G^\*(x_i) $​
10. |       |$w_i \leftarrow w_i*exp(-\frac{1}{2}\beta^{\*(m)}y_i^Tb^{\*(m)}(x_i))$​​ 
11. |    **end**
12. |    $f^{(m)}\leftarrow f^{(m-1)}+\beta^{\*(m)}b^{\*(m)}$​
13. **end**
14.  $c(x) \leftarrow argmax_kf^{(M)}(x)$

4.找一找。

5.

6.请说明左式第三个等号为何成立。

答：在样本$S(y)=k$的条件下，$y=[-\frac{1}{K-1},...,1,...,-\frac{1}{K-1}]$,和$h^{(m)}(x)=[h_{1}^{(m)}(x),...,h_{k}^{(m)}(x),...,h_{K}^{(m)}(x)]$，二者的内积为$y^Th^{(m)}(x)=1*h_k^{(m)}(x)-\frac{1}{K-1}{\sum_{i\neq{k}}^Kh_i^{(m)}(x)}=h_k^{(m)}(x)-\frac{1}{K-1}(-h_k^{(m)}(x))=\frac{K}{K-1}h^{(m)}_k(x)$

将该等式带入其中，即可以得到第三个等式。

7.验证$h_k^{\*(m)}$的求解结果。

答: $E_{y\mid x}[L\mid x]=\sum_{k=1}^{K}P_w(s(y)=k\mid{x})exp(-\frac{h_k^{(m)}(x)}{K-1})$

$L=\sum_{k=1}^{K}P_w(s(y)=k\mid{x})exp(-\frac{h_k^{(m)}(x)}{K-1})+\lambda{\sum_{k=1}^{K}h_k^{(m)}}$

对其中(K+1)个变量求导得到(K+1)个等式。

$\begin{cases}-\frac{1}{K-1}P_w(s(y)=k\mid{x})exp(-\frac{h_k^{(m)}(x)}{K-1})+\lambda=0&k\in[1,K]\\\sum_{k=1}^{K}h_k^{(m)}=0\end{cases}$​​​

联立求解得到:

$\lambda=\frac{1}{K(K-1)}\sum_{k=1}^{K}P_w(s(y)=k\mid{x})exp(-\frac{h^{(m)}_k(x)}{K-1})$

则：

$h_k^{(m)}=(K-1)lnP_w(s(y)=k\mid{x})-\frac{K-1}{K}\sum_{k=1}^{K}lnP_w(s(y)=k\mid{x})+\sum_{k=1}^{K}\frac{h_k^{(m)}}{K-1}$

通过对称约束条件，可得最后一项为0.

则$h_k^{\*(m)}=(K-1)[lnP_w(s(y)=k\mid{x})-\frac{1}{K}\sum_{k=1}^{K}lnP_w(s(y)=k\mid{x})]$​​

得证！​

8.算法3的第14行给出了wi的更新策略，请说明其合理性。

答：经过上面的等式$\lambda$的求解，我们可以发现一个规律，在每轮的计算中$P_w(s(y)=k\mid{x})exp(-\frac{h^{(m)}_k(x)}{K-1})$是一个定值，（$k\in [1,K]$值都相等），我们令该值为$\lambda$。

还有另外一个规律，在w进行权重更新的时候有一个’‘归一化’‘的操作。即$w_i\leftarrow{\frac{w_i}{\sum_{i=0}^nw_i}}$,这样我们会发现如果在$w_i$计算时同时乘上一个系数对结果无影响。



在此基础上，我们来进行下面的计算.

$h_k^{(m)}(x)=-(K-1)*ln\frac{\lambda}{P_{k}^{(m)}}$​,其中$P_{k}^{(m)}=P_w(s(y)=k\mid x)$​​，

$w_i\leftarrow{w_iexp(-\frac{1}{K}y^Th^{(m)})}=w_iexp(-\frac{1}{K}*(-(K-1))y^T[ln\frac{\lambda}{P_{1}^{(m)}},..,ln\frac{\lambda}{P_{k}^{(m)}},...,ln\frac{\lambda}{P_{K}^{(m)}}])=w_iexp(\frac{K-1}{K}ln\lambda)exp(\frac{K-1}{K}y^T[lnP_1^{(m)},...,lnP_k^{(m)},...,lnP_K^{(m)}])$​​​​根据前面的规则，常数项系数可以省略。

即可得到第14行的更新策略:

$w_i\leftarrow{w_iexp(\frac{K-1}{K}y^T[lnP_1^{(m)},...,lnP_k^{(m)},...,lnP_K^{(m)}])}$​



9.请结合加权中位数的定义解决以下问题：

- 当满足什么条件时，Adaboost.R2的输出结果恰为每个基预测器输出值的中位数？
- Adaboost.R2模型对测试样本的预测输出值是否一定会属于MM个分类器中的一个输出结果？若是请说明理由，若不一定请给出反例。
- 设k∈{y1,...,yM}，记k两侧（即大于或小于等于k）的样本集合对应的权重集合为W+和W−，证明使这两个集合元素之和差值最小的k就是Adaboost.R2输出的y。
- 相对于普通中位数，加权中位数的输出结果鲁棒性更强，请结合公式说明理由。

答：

1. 满足中位数的左侧权重和与右侧权重和差值最小条件的时候。

2. 是，因为加权中位数不想中位数可能会产生$\frac{mid[1]+mid[2]}{2}$​的情况，加权中位数的结果都是在已知结果中产生的。

3. 因为: $min(\sum_{y\in W_-}\alpha-\sum_{y\in W_+}\alpha),且\sum_{y\in W_-}\alpha>\sum_{y\in W_+}\alpha<=>\frac{\sum_{y\in W_-}\alpha}{\sum\alpha}>=0.5,且k为最小分割点$​

   即为加权中位数的定义。

4. 相比于普通中位数，根据公式，加权中位数法更容易取得权重占比较大的值，这正好与我们想要选择误差较小的基学习器的想法不谋而合，所以相较于普通中位数加权中位数输出结果的鲁棒性更强。

   

   









# 知识回顾

1.参见练习3.

2.不一定。可能当前该轮权重调整并未调整过来，A的权重不一定大于B。

3.在处理分类问题时，Adaboost的损失函数是什么？请叙述其设计的合理性。

答：损失函数$L(y,f)=exp(-\frac{y^Tf}{K})$。该函数设计的合理性在于要是损失函数最小，经过数学分析可以得到,该问题等价于$argmax(y)=argmax(f)$，则损失函数最小等价于分类函数实现正确分类。

4.Adaboost.R2算法即可处理回归问题。

5.

答：

分类问题：对于输入的新样本，每一个基模型产生一个预测标签向量,$b^{(1)},...,b^{(m)}$​，以及一个加权系数,$\beta^{1},...,\beta^{(m)}$​, 如果是SAMME.R​算法则而这合为一个,$h^{(1)},...,h^{(m)}$，然后求和，取出取值最大的一类作为输出类别。

回归问题：对于输入的新样本，每一个基模型产生一个预测输出y1,...,ym,以及相应的预测器权重，从中选出加权中位数的预测输出作为最终输出。



6.

