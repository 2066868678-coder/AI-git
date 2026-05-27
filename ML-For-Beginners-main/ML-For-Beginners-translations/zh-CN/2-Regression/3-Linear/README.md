# 使用 Scikit-learn 构建回归模型：四种回归方法

## 初学者提示

线性回归用于预测<strong>数值型数据</strong>（例如房价、温度或销售额）。
其原理是找到一条最佳拟合输入特征与输出之间关系的直线。

本课重点理解概念，再探讨更高级的回归技术。
![线性回归与多项式回归信息图](../../../../translated_images/zh-CN/linear-polynomial.5523c7cb6576ccab.webp)
> 信息图作者：[Dasani Madipalli](https://twitter.com/dasani_decoded)
## [课前测验](https://ff-quizzes.netlify.app/en/ml/)

> ### [本课程也提供 R 语言版本！](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)
### 介绍

到目前为止，你已经了解了什么是回归，并用我们本课将持续使用的南瓜定价数据集进行了探索。你还用 Matplotlib 将数据进行了可视化。

现在你准备深入学习机器学习中的回归。虽然可视化有助于理解数据，但机器学习的真正威力来自于_训练模型_。模型在历史数据上训练，能够自动捕捉数据间的依赖关系，并能对未见过的新数据做出预测。

本课将介绍两种回归类型：_基础线性回归_和_多项式回归_，并讲解其中的数学原理。这些模型将帮助我们根据不同输入预测南瓜价格。

[![为初学者讲解线性回归](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "为初学者讲解线性回归")

> 🎥 点击上方图片观看线性回归的简短视频概览。

> 在整个课程中，我们假设学生数学基础有限，力求让其他领域的学生也能理解，因此请关注笔记、🧮 数学提示、图表等辅助学习工具。

### 先决条件

到目前为止，你应该熟悉我们正在研究的南瓜数据结构了。课中的 _notebook.ipynb_ 文件中已经预加载并预处理了数据。文件中每个蒲式耳的南瓜价格展示于新的数据框中。确保你可以在 Visual Studio Code 的内核中运行这些笔记本。

### 准备工作

提醒一下，你加载数据是为了提出问题。

- 什么时候买南瓜最合适？
- 一箱迷你南瓜价格大概是多少？
- 应该买半蒲式耳篮装，还是1 1/9蒲式耳盒装？
让我们继续挖掘数据。

上一课你创建了 Pandas 数据框，并填充了部分原始数据，将价格标准化为每蒲式耳价格。但这样你只能获得约400条数据，且仅限秋季月份。

看看本课随附笔记本中预加载的数据。数据已加载，并绘制了月份散点图。或许我们可以通过更深入的数据清洗来了解数据的本质。

## 线性回归直线

正如第1课所学，线性回归的目标是绘制一条线来：

- <strong>展示变量关系</strong>。显示变量之间的关系。
- <strong>进行预测</strong>。准确预测新数据点相对于该线的位置。

<strong>最小二乘回归</strong>通常会绘制这样的线。“最小二乘”指的是最小化模型总误差的过程。对每个数据点，测量其与回归线的垂直距离（称为残差）。

我们平方这些距离主要有两个原因：

1. <strong>大小优先于方向</strong>：想让误差-5与+5同等对待，平方将所有值变为正数。

2. <strong>惩罚离群值</strong>：平方会给较大误差更高权重，迫使回归线更靠近远离的点。

接着将所有平方值相加。目标是找到使该总和最小的线——这就是“最小二乘”名称的由来。

> **🧮 看数学表达** 
> 
> 这条称为_最佳拟合线_的线可以用[方程](https://en.wikipedia.org/wiki/Simple_linear_regression)表达： 
> 
> ```
> Y = a + bX
> ```
>
> `X` 是“自变量”，`Y` 是“因变量”。斜率为 `b`，截距为 `a`，表示 `X=0` 时的 `Y` 值。
>
>![计算斜率](../../../../translated_images/zh-CN/slope.f3c9d5910ddbfcf9.webp)
>
> 首先计算斜率 `b`。信息图作者 [Jen Looper](https://twitter.com/jenlooper)
>
> 换句话说，对于南瓜数据：“按月份预测南瓜每蒲式耳价格”，`X` 是价格，`Y` 是销售月份。
>
>![完整方程](../../../../translated_images/zh-CN/calculation.a209813050a1ddb1.webp)
>
> 计算 `Y` 值。如果你付了大约4美元，那一定是4月！信息图作者 [Jen Looper](https://twitter.com/jenlooper)
>
> 计算公式要展示斜率，还要考虑截距，即当 `X=0` 时 `Y` 的位置。
>
> 你可以参考 [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html) 网站来学习计算方法。还可以访问[这个最小二乘计算器](https://www.mathsisfun.com/data/least-squares-calculator.html)观察数字如何影响回归线。

## 相关性

另一个要理解的术语是给定 `X` 和 `Y` 变量之间的<strong>相关系数</strong>。用散点图可以快速观察这一系数。数据点整齐排列成线的图表相关性高，点散布杂乱无章则相关性低。

好的线性回归模型会有较高的（接近于1而非0）相关系数，采用最小二乘法拟合回归线。

✅ 运行本课的笔记本，看看“月份与价格”的散点图。根据你的视觉判断，这两个变量的南瓜销售价格相关性是高还是低？如果用更细化的度量（如一年中的某一天，即从年初起的天数）替代 `Month`，相关性会变化吗？

下面的代码假设已清理数据并获得名为 `new_pumpkins` 的数据框，内容类似：

ID | Month | DayOfYear | Variety | City | Package | Low Price | High Price | Price
---|-------|-----------|---------|------|---------|-----------|------------|-------
70 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364
71 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
72 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
73 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 17.0 | 17.0 | 15.454545
74 | 10 | 281 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364

> 清理数据的代码可见于[`notebook.ipynb`](notebook.ipynb)中。我们执行了与上一课相同的清理步骤，并用如下表达式计算了 `DayOfYear` 列：

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```

既然已理解线性回归背后的数学原理，让我们创建一个回归模型，看看是否能预测哪种南瓜包装的价格最优。想为节日南瓜园采购南瓜的人可能想知道这些信息，以便优化采购计划。

## 寻找相关性

[![为初学者讲解相关性：线性回归的关键](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "为初学者讲解相关性：线性回归的关键")

> 🎥 点击上方图片观看关于相关性的简短视频。

上一课你可能已经看到，不同月份的平均价格如下图所示：

<img alt="按月平均价格" src="../../../../translated_images/zh-CN/barchart.a833ea9194346d76.webp" width="50%"/>

这说明应存在某种相关性，我们可以试着训练线性回归模型来预测 `Month` 与 `Price` 之间，或 `DayOfYear` 与 `Price` 之间的关系。以下是价格与一年中日期关系的散点图：

<img alt="价格与年日期散点图" src="../../../../translated_images/zh-CN/scatter-dayofyear.bc171c189c9fd553.webp" width="50%" /> 

我们用 `corr` 函数来看看相关性：

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```

看起来相关性非常小，`Month` 为 -0.15，`DayOfYear` 为 -0.17，但可能存在另一种重要关系。不同南瓜品种对应不同价格集群。为确认此假设，我们用不同颜色绘制各品种。通过给 `scatter` 函数传递 `ax` 参数，可以在同一图表上绘制所有点：

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```

<img alt="价格与年日期不同颜色散点图" src="../../../../translated_images/zh-CN/scatter-dayofyear-color.65790faefbb9d54f.webp" width="50%" /> 

调查显示，品种对整体价格的影响大于销售日期。我们用柱状图表示：

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```

<img alt="按品种划分的价格柱状图" src="../../../../translated_images/zh-CN/price-by-variety.744a2f9925d9bcb4.webp" width="50%" /> 

先暂时只关注一种品种——“pie type”，看看日期对价格的影响：

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
<img alt="pie type 南瓜的价格与年日期散点图" src="../../../../translated_images/zh-CN/pie-pumpkins-scatter.d14f9804a53f927e.webp" width="50%" /> 

若用 `corr` 函数计算 `Price` 和 `DayOfYear` 的相关系数，结果大概为 `-0.27`，说明训练预测模型是有意义的。

> 在训练线性回归模型前，务必保证数据干净。线性回归对缺失值不敏感，最合理的做法是去除所有空值：

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```

另一种做法是用对应列的均值填充空白值。

## 简单线性回归

[![为初学者讲解使用 Scikit-learn 进行线性和多项式回归](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "为初学者讲解使用 Scikit-learn 进行线性和多项式回归")

> 🎥 点击上方图片观看线性回归和多项式回归的简短视频。

我们将使用<strong>Scikit-learn</strong>库训练线性回归模型。

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```

首先，将输入值（特征）和预期输出（标签）分隔成独立的 numpy 数组：

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```

> 注意，我们对输入数据进行了 `reshape`，以便线性回归库能正确解读。线性回归要求输入为二维数组，数组每行为一组输入特征向量。这里只有一个输入，因此需要形状为 N×1 的数组，N 是数据集大小。

接着，需要将数据分割为训练集和测试集，以便训练后验证模型：

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

最后，训练线性回归模型只需两行代码。定义 `LinearRegression` 对象，用 `fit` 方法拟合数据：

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

`LinearRegression` 对象在 `fit` 之后包含了回归的所有系数，可以通过 `.coef_` 属性访问。在我们的例子中，只有一个系数，应该大约是 `-0.017`。这意味着价格似乎随着时间略有下降，但不多，大约每天降 2 美分。我们还可以通过 `lin_reg.intercept_` 访问回归与 Y 轴的截距点——在我们的例子中大约是 `21`，表示年初的价格。

为了查看模型的准确度，我们可以在测试数据集上预测价格，然后测量预测值与预期值的接近程度。可以使用均方根误差（RMSE）指标，即所有预期值与预测值差值平方的平均值的平方根。

```python
pred = lin_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test,pred))
print(f'RMSE: {rmse:3.3} ({rmse/np.mean(pred)*100:3.3}%)')
```

我们的误差大约是 2 点，约为 ~17%。不是很理想。模型质量的另一个指标是<strong>决定系数</strong>，可以这样获得：

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```
如果值为 0，表示模型没有考虑输入数据，表现为<em>最差的线性预测器</em>，即结果的均值。值为 1 表示我们可以完美预测所有预期输出。在我们的例子中，系数约为 0.06，相当低。

我们还可以将测试数据和回归线画在一起，更直观地观察回归效果：

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```

<img alt="Linear regression" src="../../../../translated_images/zh-CN/linear-results.f7c3552c85b0ed1c.webp" width="50%" />

## 多项式回归

另一种线性回归称为多项式回归。有时变量之间存在线性关系——比如，体积越大的南瓜价格越高，但有时这些关系不能用平面或直线表示。

✅ 这里有一些[更多示例](https://online.stat.psu.edu/stat501/lesson/9/9.8)演示可以使用多项式回归的数据

再看看日期和价格的关系。这个散点图是否必然适合用直线来分析？价格难道不会波动吗？这时可以尝试多项式回归。

✅ 多项式是由一个或多个变量及系数组成的数学表达式

多项式回归创建一条曲线来更好地拟合非线性数据。在我们的例子中，如果在输入数据中加入平方的 `DayOfYear` 变量，应该可以用抛物线拟合数据，在一年中的某个点达到最低点。

Scikit-learn 包含一个方便的[流水线 API](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline)来组合不同的数据处理步骤。<strong>流水线</strong>是由一系列<strong>估计器</strong>组成的链。在我们的例子中，我们将创建一个流水线，先添加多项式特征，然后训练回归模型：

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```

使用 `PolynomialFeatures(2)` 意味着包括输入数据中的所有二次多项式。在我们的例子中，这仅表示 `DayOfYear` 的平方，但如果有两个输入变量 X 和 Y，就会添加 X²、XY 和 Y²。如果需要，也可以使用更高次数的多项式。

流水线可以像原始的 `LinearRegression` 对象一样使用，即可以 `fit` 流水线，然后用 `predict` 获得预测结果。下面是测试数据和拟合曲线的图示：

<img alt="Polynomial regression" src="../../../../translated_images/zh-CN/poly-results.ee587348f0f1f60b.webp" width="50%" />

使用多项式回归，我们可以得到稍低的均方误差和更高的决定系数，但提升不大。我们还需要考虑其他特征！

> 你可以看到南瓜最低价格出现在万圣节附近。你如何解释这一现象？

🎃 恭喜，你刚刚创建了一个能够帮助预测派皮南瓜价格的模型。你可能会对所有南瓜类型都重复这个过程，但那会很繁琐。接下来我们学习如何在模型中考虑南瓜品种！

## 类别特征

理想情况下，我们希望用同一个模型预测不同南瓜品种的价格。但 `Variety` 一栏与 `Month` 等数值型列不同，它包含的是非数值数据。这类列称为<strong>类别特征</strong>。

[![ML for beginners - 使用线性回归预测类别特征](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "ML for beginners - 使用线性回归预测类别特征")

> 🎥 点击上图观看一个简短视频，介绍如何使用类别特征。

你可以看到平均价格如何根据品种变化：

<img alt="Average price by variety" src="../../../../translated_images/zh-CN/price-by-variety.744a2f9925d9bcb4.webp" width="50%" />

为了在模型中考虑品种，首先需要对其进行数值化，或称<strong>编码</strong>。编码有几种方法：

* 简单的<strong>数字编码</strong>会建立一个品种表，然后用该表中的索引替换品种名称。这对线性回归不是最优，因为线性回归将索引的数值直接参与计算，乘以某个系数后加到结果中。在这里，索引与价格关系明显非线性，即使索引按照特定顺序排列。
* <strong>独热编码</strong>会用四个不同的列替换掉 `Variety` 列，每个对应一个品种。对应品种的列取 1，其他列为 0。这样线性回归会得到四个系数，分别对应四种南瓜品种的“基础价”（或“附加价”）。

下面的代码展示如何对品种进行独热编码：

```python
pd.get_dummies(new_pumpkins['Variety'])
```

 ID | FAIRYTALE | MINIATURE | MIXED HEIRLOOM VARIETIES | PIE TYPE
----|-----------|-----------|--------------------------|----------
70  | 0         | 0         | 0                        | 1
71  | 0         | 0         | 0                        | 1
... | ...       | ...       | ...                      | ...
1738| 0         | 1         | 0                        | 0
1739| 0         | 1         | 0                        | 0
1740| 0         | 1         | 0                        | 0
1741| 0         | 1         | 0                        | 0
1742| 0         | 1         | 0                        | 0

使用独热编码的品种作为输入训练线性回归，只需正确初始化 `X` 和 `y` 数据：

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```

其余代码与之前训练线性回归时相同。你尝试后会发现均方误差差不多，但决定系数大大提升到 ~77%。为了得到更精准的预测，可以同时考虑更多类别特征和数值特征，比如 `Month` 或 `DayOfYear`。要合并成一个大特征数组，可以使用 `join`：

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```

这里还考虑了 `City` 和 `Package` 类型，最后得到均方误差 2.84（10%），决定系数 0.94！

## 综合应用

为了做出最佳模型，我们可以结合上述例子中独热编码的类别特征和数值特征，再使用多项式回归。以下是完整代码，供你方便使用：

```python
# 设置训练数据
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']

# 进行训练-测试集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 设置并训练流水线
pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
pipeline.fit(X_train,y_train)

# 预测测试数据结果
pred = pipeline.predict(X_test)

# 计算均方误差和决定系数
mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')

score = pipeline.score(X_train,y_train)
print('Model determination: ', score)
```

这应该能达到接近 97% 的最高决定系数，均方误差为 2.23（约 8% 预测误差）。

| 模型                     | MSE           | 决定系数      |
|--------------------------|---------------|---------------|
| `DayOfYear` 线性回归     | 2.77 (17.2%)  | 0.07          |
| `DayOfYear` 多项式回归   | 2.73 (17.0%)  | 0.08          |
| `Variety` 线性回归       | 5.24 (19.7%)  | 0.77          |
| 所有特征 线性回归        | 2.84 (10.5%)  | 0.94          |
| 所有特征 多项式回归      | 2.23 (8.25%)  | 0.97          |

🏆 干得好！你在一节课中创建了四个回归模型，并将模型质量提升到了 97%。回归的最后一节课中，你将学习用逻辑回归确定类别。

---
## 🚀挑战

尝试本笔记本中的几个不同变量，观察它们的相关性如何影响模型准确度。

## [课后小测](https://ff-quizzes.netlify.app/en/ml/)

## 复习与自学

本课我们学习了线性回归。还有其他重要的回归类型。请学习逐步回归、岭回归、套索回归和弹性网方法。推荐课程：[斯坦福统计学习课程](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning)

## 作业

[建立模型](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**免责声明**：  
本文档是使用 AI 翻译服务 [Co-op Translator](https://github.com/Azure/co-op-translator) 翻译的。虽然我们力求准确，但请注意自动翻译可能包含错误或不准确之处。原始文档的母语版本应被视为权威来源。对于关键信息，建议使用专业人工翻译。对于因使用本翻译而产生的任何误解或误读，我们概不负责。
<!-- CO-OP TRANSLATOR DISCLAIMER END -->