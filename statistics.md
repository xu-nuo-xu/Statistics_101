<!-- TOC -->

- [统计学课程Statistics 101笔记](#统计学课程statistics-101笔记)
    - [PL15 - Multiple Linear Regression](#pl15---multiple-linear-regression)
    - [PL16 - Logistic Regression](#pl16---logistic-regression)
    - [PL17 - ANCOVA (ANalysis of COVAriance)](#pl17---ancova-analysis-of-covariance)
    - [PL19 - Nonparametric Methods](#pl19---nonparametric-methods)
        - [Sign Test For Median Examples](#sign-test-for-median-examples)
        - [Mann-Whitney-Wilcoxon Rank Sum Test](#mann-whitney-wilcoxon-rank-sum-test)

<!-- /TOC -->
# 统计学课程Statistics 101笔记
## PL15 - Multiple Linear Regression
>相比于之前我们在课本中提到过的一元线性回归，这里拓展到了多元线性回归。如下图所示，我们讨论三种观点：
>+ 我们不能将一元线性回归分别用于多元线性回归，这样会造成过拟合。
>+ 多个自变量之间可能会有相关关系，叫做多重共线性，我们要适当进行取舍。
>+ 理想的情况下是每个自变量都和因变量有相关关系，但自变量之间相互独立。
<br>
<div align=center><img src="pictures/1.png"  width="80%" height="80%"><br>
<div align=left>
<br>

>因此，我们除了要讨论自变量和因变量之间的关系，还要讨论各个自变量之间的关系。如下图若有四个自变量我们要讨论10种相关关系。<br>
<div align=center><img src="pictures/2.png"  width="80%" height="80%"><br>
<div align=left>
<br>

>下面一张图给我们介绍了多元线性回归的步骤：<br>
<div align=center><img src="pictures/3.png"  width="80%" height="80%"><br>
<div align=left>
<br>

>我们从第三步、第四步开始说起，一般先是用散点图进行目测，下面的例子中有三个自变量。因此我们做出三个自变量和因变量之间的散点图、三个自变量之间的散点图：<br>
<div align=center><img src="pictures/4.png"  width="80%" height="80%"><br>
<div align=left>
<br>

>下图1，我们观察到前两个自变量和因变量之间有强线性关系，最后一个自变量和因变量之间没有强线性关系。因此我们在构建模型时往往不会使用第三个自变量。下图2，我们观察到自变量 x1,x2 之间有强线性关系; x1,x3 之间 x2,x3 之间没有强线性关系。因此我们在构建模型时往往会在 x1 和 x2 之间选择一个，因为他们之间存在多重共线性问题。<br>
<div align=center><img src="pictures/5.png"  width="80%" height="80%"><br>
<div align=left>
<br>
<div align=center><img src="pictures/6.png"  width="80%" height="80%"><br>
<div align=left>
<br>

>我们还可以用 matlab 求出各个变量之间准确的相关系数(Pearson correlation)和 p-value (统计学根据显著性检验方法所得到的P 值，一般以P < 0.05 为显著， P <0.01 为非常显著，其含义是样本间的差异由抽样误差所致的概率小于 0.05 或 0.01)。可以得出和上面目测散点图相似的结论：<br>
<div align=center><img src="pictures/7.png"  width="80%" height="80%"><br>
<div align=left>
<br>

>现在我们进行第五步，我们用三个自变量分别进行一元回归分析。得出下面的统计表.这里的 F 值来自于 ANOVA (方差分析中 $F=Q_A-hat/Q_E-hat$，需要补充理解的是：[线性回归和方差分析的关联](https://blog.csdn.net/xiangmin_meng/article/details/22402545))。F 值越高，说明二者线性关系越显著，p-value 越小。$R^2=SSR/SST$，越接近 1 表明方程的变量对 y 的解释能力越强，这个模型对数据拟合的也较好。你可以看到 $R^2$ 分为 $R^2(adjusted) 和 R^2(prediction)$，adjusted 表明拟合程度，而prediction 表明对新的数据点的预测准确度：<br>
<div align=center><img src="pictures/8.png"  width="80%" height="80%"><br>
<div align=left>
<br>

>下面我们引入两个变量的回归指标和 VIF (varience inflation factor)。前三行是之前单自变量拟合情况，后三个分别以 $(x1,x2)、(x1,x3)、(x2,x3)$ 为自变量进行拟合，我们发现因为每一种组合要么包含 x1,要么包含 x2 因此$R^2(adjusted)$ 都不低，但是第四行的 VIF 值很高，并且$R^2(prediction)$相比于$R^2(adjusted)$低了很多，这就说明出现了过拟合情况。VIF 的作用在下面的黑字中有介绍，用来评价自变量的相关性。单从这个表中我们发现第五行第六行在各方面都有不错的表现。但是我们在拟合后的系数中其实会发现有的系数是不符合实际情况的，即 x3 的系数是负值，之前我们介绍了 x3 和 y 之间的线性性不强，因此这里引入 x3 有些多余：<br>
<div align=center><img src="pictures/9.png"  width="80%" height="80%"><br>
<div align=left>
<br>

>最后我们将三个变量统一起来构成自变量$(x1,x2,x3)$，回归后的指标如下。我们看到 $R^2(prediction)=57.49%$ 很低，VIF $x1=14.94$ 很高，$x2=17.35$ 很高，因此存在很严重的多重共线性问题：<br>
<div align=center><img src="pictures/10.png"  width="80%" height="80%"><br>
<div align=left>
<br>

>那么最后，我们需要选择一个最好的模型。排除高的 VIF 选项第 4 个$(x1,x2)$ 和第 7 个后$(x1,x2,x3)$ ，我们将关注以下几个因素进行进一步筛选：
>+ $R^2(adjusted)$ $R^2(prediction)$ 较高，并且二者很接近
>+ 标准差 s 较低
>+ 模型尽可能简单

>于是，很明显的，第一个模型就是最佳模型，只由 x1 构成自变量的模型。以上的相关拟合数据可以由 Minitab 生成，在 Minitab 中还会生成一个叫做 $Mallows C_p$的指标，介绍见下图第四点。1.9 接近于 1+1=2 因此第一个模型的这个指标也是很好的：<br>
<div align=center><img src="pictures/11.png"  width="80%" height="80%"><br>
<div align=left>
<br>

>下面还有关于 Dummy Varible 的补充。我们上面见到的自变量都有连续的取值，但是有些时候影响因变量结果的因素不一定是连续的，我们称之为分类变量。比如下面这个例子关于房屋价格影响因素，x1 是房屋的面积，x2 是所属高中是否是典范高中。因此 x2 只有两种情况，是或不是。因此我们需要对 x2 进行编码，很容易想到是的话编码为 x2=1 ，否的话编码为 x2=0。线性模型为 $y=β0+β1·x1+β2·x2$。其他的操作和之前多变量线性回归相同。但是如果大于 1 个变量是分类变量，编码方法如下，下图为方向为 4 种选项的情况。而此时 $Dummy Varible = n - 1 = 4-1=3(x1,x2,x3三个虚变量)$，第一种情况$Dummy Varible = n - 1 = 2-1=1$：<br>
<div align=center><img src="pictures/12.png"  width="60%" height="80%"><br>
<div align=left>
<br>

>对于房屋的例子，我们最终得到的结果往往如下，图一是综合考虑二者的平均，图二是将二者区分开进行考虑：<br>
<div align=center><img src="pictures/13.png"  width="80%" height="80%"><br>
<div align=left>
<br><div align=center><img src="pictures/14.png"  width="80%" height="80%"><br>
<div align=left>
<br>

>下面依旧是房屋价格的例子，只不过这次除了房屋面积，还增加了 location 和典范高中的信息，location 分为东南西北四个，因此$Dummy Varible = n - 1 = 4-1=3$，下图是关于部分数据的编码情况，我们看到只有三个 Dummy Varible，其中 east 被编码为 $(0,0,0)$：<br>
<div align=center><img src="pictures/15.png"  width="80%" height="80%"><br>
<div align=left>
<br>

>下面是模型的 Equation，还有一个例子，以及模型包含的等式个数：<br>
<div align=center><img src="pictures/16.png"  width="80%" height="80%"><br>
<div align=left>
<br>

>之后通过 Minitab 或其他工具包，我们可以得到各个变量的系数信息。我们重点关注 P-Value 的值，我们可以看到 sqrt，exemplary，west 三者的 P-Value 较低，也就是对应的显著性水平较高。因此他们三者对房屋价格影响应当是最大的：<br>
<div align=center><img src="pictures/18.png"  width="80%" height="80%"><br>
<div align=left>
<br>

>我们按 exemplary 的 yes or no 对结果进行分类作图，两个图的绿色线都是 west 对应的拟合结果，可以看到 west location 房屋相对便宜很多。两幅图对比看整体，图一的截距整体相对于图二的截距要低一些，说明 exemplary 为 yes 的房屋价格要高于 no 的：<br>
<div align=center><img src="pictures/17.png"  width="80%" height="80%"><br>
<div align=left>
<br>

## PL16 - Logistic Regression
>本节只对逻辑回归进行一个简单了解性的介绍。首先我介绍一下关于几率(odds)的概念，几率和概率是紧密联系的，下面这张图介绍了几率和概率(probability)、以及几率比率(odds raito)的概念。我们看到均匀的硬币掷出 head 的概率为 0.5，不均匀的(loaded)硬币掷出 head 的概率为 0.7。几率就是事件发生的概率/时间不发生的概率。几率比率就是两个事件的几率比值：<br>
<div align=center><img src="pictures/19.png"  width="80%" height="80%"><br>
<div align=left>
<br>

>而逻辑回归问题往往处理的是自变量有很多种，因变量是一个二分类的变量的问题。比如根据信用评分来决定是否进行贷款的问题。因此，逻辑回归的问题主要集中在概率的讨论上，比如给出一个信用评分，有多大的概率可以进行贷款。而根据已知数据点进行拟合时，我们常常用下面的 Sigmoid 函数进行，Sigmoid 函数就是 $ln(p/1-p)$ 的逆函数。其中的参数计算用到了最大似然估计的内容：<br>
<div align=center><img src="pictures/20.png"  width="80%" height="80%"><br>
<div align=left>
<br>
<div align=center><img src="pictures/21.png"  width="60%" height="60%"><br>
<div align=left>
<br>

>需要注意的是逻辑回归中，同间隔的两个自变量点的几率比值始终为一个定值，我们看下面这个例子，讲的是根据信用评分来决定是否进行贷款的问题。相应的系数计算过程课程中没有讲，在此就略去。我们分别代入 $x=720 和 x=721$ 两个点的几率，进一步我们可以得到二者的几率比值为 1.0146，如下图所示：<br>
<div align=center><img src="pictures/22.png"  width="80%" height="60%"><br>
<div align=left>
<br>

>也就是说，无论起始点在哪，只要 x 增加一个单位，原来的几率与增加后的几率比值都是 1.0146，并且 Minitab 等软件会给我们一个置信区间(confidence interval ( CI ))，如果这个置信区间中包含 1 就说明 x 的增加可能会对判定结果的几率/概率没有影响(这里 1.0147 和 1.0146 只是舍入的区别)：<br>
<div align=center><img src="pictures/23.png"  width="80%" height="60%"><br>
<div align=left>
<br>

>最后一点是关于 x (FICO) 增加的幅度，得到的几率比率图。这幅图有意思的点在于，它完全符合一个指数函数的形式，并且指数的系数 0.0146 恰好等于我们得到的 p-hat 的系数 β1 ：<br>
<div align=center><img src="pictures/24.png"  width="80%" height="60%"><br>
<div align=left>
<br>

## PL17 - ANCOVA (ANalysis of COVAriance)
>这里只简要说明一下 ANCOVA 的作用。作者举了一个例子是关于学生年级为分类型自变量和学习能力评分为连续型因变量之间的关系。如果只有这一个自变量和因变量就可以使用 One-way ANOVA。但是，我们知道影响学习能力评分的因素可能还有学生自身的 GPA，也就是说 GPA 也可以被讨论进来。于是就有两个自变量，分类型的学生年级和连续型的 GPA：<br>
<div align=center><img src="pictures/25.png"  width="70%" height="70%"><br>
<div align=left>
<br>

>ANCOVA 其实就是根据协变量 GPA 与学习能力评分之间存在的线性关系，来调整学生年级变量对学习能力评分的影响。体现在误差上就是，一部分 SSE 分给了 Cov 协变量，毕竟协变量也会对最终结果产生影响。如果协变量与因变量之间存在强线性关系，那么会有很大一部分误差分给 Cov，于是原变量的显著性水平就会大幅提高。<br>
<div align=center><img src="pictures/26.png"  width="80%" height="70%"><br>
<div align=left>
<br>

>ANCOVA 最终对原变量的测试其实是在保证其他所有协变量一致的情况下进行的。更准确的来说就是通过对原变量各个学生年级的学习能力评分的平均值来测试的。下面这张图就是 ANOVA 和 ANCOVA 二者对均值的改变情况。并且最终将 GPA 控制在一个相同的水平 2.9843 下：<br>
<div align=center><img src="pictures/27.png"  width="80%" height="80%"><br>
<div align=left>
<br>

## PL19 - Nonparametric Methods
>在一个统计推断问题中，如果总体分布的具体形式已知(最常见的是假定为正态分布)，则我们只需对其中含有的若干个未知参数作出估计或进行某种形式的假设检验，这类推断方法称为参数方法。但在许多实际问题中，我们对总体分布的形式往往所知甚少(如只能作出诸如连续型分布、关于均值对称等微弱的假定)，甚至一无所知。这时就需要使用不必(或很少)依赖于总体分布形式的统计推断方法，此类推断方法通常称为非参数方法(non-parametric method)。<br>

### Sign Test For Median Examples

>我们从最简单的符号检验中位数开始讲起。比如我们对于一个容量为 12 的样本，来估计总体的中位数，样本如下图所示。我们假设总体的中位数为 60200 ，即大约在第三个样本和第四个样本之间的一个值。并检验假设 $H_0:Median=60200$ $H_α：Median≠60200$。我们统计样本中小于 60200 和大于 60200 的样本个数以及占总体的概率 p。是否符合假设 $H_0:P\{x<60200\}=0.5$ $H_α:P\{x<60200\}≠0.5$。若不符合，有多大的概率可以否定原假设。注意假设 $H_0:p=0.5$ $H_α:p≠0.5$使得样本中大于/小于 60200 的个数 N 符合二项分布 $B(n,p)=B(12,0.5)$。<br>
<div align=center><img src="pictures/28.png"  width="80%" height="80%"><br>
<div align=left>
<br>

>按照上图所示的样本，我们看到样本中小于 60200 的有 3 个，根据原假设 $H_0:p=0.5$，我们得到下面的二项分布表，即 12 个样本中若出现 0-12 个小于 60200 的值的概率。根据之前的假设检验知识我们知道， 0/1/2/3 个小于 60200 的值出现的概率之和，即 0.073，我们可以作为单边的小概率事件的概率。因为原假设涉及的检验是双边检验。因此 $p-value = 0.073 * 2 = 0.146$，若取显著性水平为 0.10 ，则 $p-value > α$，也就是说对于显著性水平 0.10 对应的 N 应当是小于 3 或者大于 9 的，于是我们样本中出现的 3 个就满足相应的要求。于是我们不能拒绝原假设$H_0:P\{x<60200\}=0.5$，也就是$H_0:Median=60200$。 <br>
<div align=center><img src="pictures/29.png"  width="80%" height="80%"><br>
<div align=left>
<br>

>但这貌似是不合理的，因为按照常理来说我们应当选取第六个到第七个样本之间的一个值作为中位数。其他的都应该尽量否定假设。问题就出在样本量上，样本太小导致估计不准确。我们接下来再假设$H_0:Median=97600$ $H_α：Median≠97600$。我们样本中有 2 个是大于 97600 ，10 个小于97600。于是我们统计二项分布 $B(n,p)=B(12,0.5)$ 中小于 2 和 大于 10 二者的概率和 p-value = 0.036，对于显著性水平 $0.10 > p-value$，于是我们否定了原假设。于是对于二项分布 $B(n,p)=B(12,0.5)$ 显著性水平 0.10，恰好满足的 N 的数量应当是介于 2 个到 3 个之间的一个值，或者说 9 个到 10 个之间的一个值：<br>
<div align=center><img src="pictures/30.png"  width="80%" height="80%"><br>
<div align=left>
<br>

>如果我们选择更大一点的样本量，根据以往的知识，二项分布 $B(n,p)$ 可近似看为连续的正态分布 $N(np,np(1-p))$。下面这个例子就讲的是更大的样本我们用正态分布近似二项分布进行非参数检验。右下角是我们的假设，我们看到这是一个单边假设。样本中有 51 个工资小于 75K 的员工，34 个工资大于 75K 的员工。<br>
<div align=center><img src="pictures/31.png"  width="80%" height="80%"><br>
<div align=left>
<br>

>于是我们将 $B(n,p)=B(85,0.5)$ 近似于 $N(np,np(1-p))=N(42.5,21.25)$，我们按照下面的步骤进行检验。需要强调的是，离散的情况下我们本来要检验 $P(x<=34)$ 与对应显著性水平大小关系，但是转换为连续的就需要加 0.5 变成 $P(x<=34.5)$ (continuity correction factor)。因为连续的二项分布我们就构造统计量：$\frac{X-μ}{σ}$，查表 $N(0,1)$ 即可得到对应的大小关系。
<br>
<div align=center><img src="pictures/33.png"  width="80%" height="80%"><br>
<div align=left>
<br>

>查表后结果如下，若显著性水平取 $α=0.05$，$P=0.0413 < α$，于是否定原假设 $Median >= 75000$。:<br>
<div align=center><img src="pictures/34.png"  width="80%" height="80%"><br>
<div align=left>
<br>

>最后总结一下关键点如下图。其实关键就是把中位数的检验转化为样本中大于/小于中位数个数的检验。这个检验就叫做符号检验(sign-test)。我们不用知道原来数据的分布情况，仅凭大于/小于中位数的样本数即可决定检验结果：
<br>
<div align=center><img src="pictures/35.png"  width="80%" height="80%"><br>
<div align=left>
<br>

### Mann-Whitney-Wilcoxon Rank Sum Test
>我们先介绍一下Mann-Whitney-Wilcoxon Rank Sum Test的背景(作用)。在参数检验中，我们如果需要检验两个正态总体的均值是否相等，我们会用到 t-test，但在非参数检验中，Mann-Whitney-Wilcoxon Rank Sum Test 给我们提供了一个当两个总体分布未知的情况下，二者的中位数是否相等的方法。我们采用的是从两个总体中分别抽取一定样本，来对这些样本进行一定的“排名(rank)”，我们在检验时，并不关注样本的准确数值而是只关注样本的排名情况，来判定二者的中位数是否相同：<br>
<div align=center><img src="pictures/36.png"  width="80%" height="80%"><br>
<div align=left>
<br>
