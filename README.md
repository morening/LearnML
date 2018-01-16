## 线性回归
[practice0112](https://github.com/morening/LearnML/blob/master/linear_regression/practice0112.py)

拟合 y = a*x1 + b*x2 + c

训练集 x:[[1, 2], [2, 1], [2, 3], [3, 5], [1, 3], [4, 2], [7, 3], [4, 5], [11, 3], [8, 7]]

训练集 y:[7, 8, 10, 14, 8, 13, 20, 16, 28, 26]

测试集 x:[[1, 4], [2, 2], [2, 5], [5, 3], [1, 5], [4, 1]]

**测试结果**
```
计算轮次：4009，耗时：0.434706，最小误差：0.000001
a = 2.000087
b = 1.000458
c = 2.997658
#测试：
[6.9986617712594938, 7.998290420402709, 9.9992069599056101, 14.00021041830318, 7.9991200410109435, 12.998922527943494, 19.999641554378947, 16.000297337197846, 27.999989229957613, 26.001561552279416]
#计算：
[8.999578310762395, 8.9987486901541605, 12.000123499408513, 15.99946771658961, 10.000036580513846, 11.998464258192042]
```

<img width="50%" height="50%" src="https://github.com/morening/LearnML/blob/master/snapshot/linear_regression/practice0112.png?raw=true" />

最后，感谢[《梯度下降原理及Python实现》](http://blog.csdn.net/programmer_wei/article/details/51941358)的帮助与指导，令我深刻理解梯度下降法的原理和完成python实现。

## 线性回归（矩阵）
[practice0115](https://github.com/morening/LearnML/blob/master/linear_regression/practice0115.py)

设 y = theta0 + theta1*x1 + theta2*x2

**测试结果**
```
#参数拟合：
[[ 2.9999716 ]
 [ 2.00000105]
 [ 1.00000556]]
#轮次：6528
#测试结果：
[[  6.99998377]
 [  7.99997927]
 [  9.99999038]
 [ 14.00000255]
 [  7.99998933]
 [ 12.99998693]
 [ 19.99999565]
 [ 16.00000361]
 [ 27.99999987]
 [ 26.00001894]]
```

## 逻辑回归

### 测试结果（[data1](https://github.com/morening/LearnML/blob/master/data/data1.txt)）

***一阶拟合***

```
#参数拟合：
[[-280.76963691]
 [   2.64287161]
 [   1.98393109]]
#正确率：92.0%
#测试结果：
[[  1.07319525e-15]
 [  4.42892157e-50]
 [  1.05119846e-18]
 [  1.00000000e+00]
 [  1.00000000e+00]
 [  2.14759841e-22]
 [  1.00000000e+00]
 [  9.99948356e-01]
 [  1.00000000e+00]
 [  1.00000000e+00]
 [  1.00000000e+00]
 [  3.39548891e-10]
 [  1.00000000e+00]
 [  1.00000000e+00]
 [  9.09687815e-12]
 [  1.00000000e+00]
 [  9.98353316e-01]
 [  1.86340791e-04]
 [  1.00000000e+00]
 [  9.99999961e-01]
 [  2.00405482e-08]
 [  1.00000000e+00]
 [  1.44882945e-22]
 [  2.64702210e-45]
 [  1.00000000e+00]
 [  1.00000000e+00]
 [  9.99999998e-01]
 [  1.00000000e+00]
 [  2.14898366e-08]
 [  3.81937603e-22]
 [  1.00000000e+00]
 [  1.00000000e+00]
 [  1.83418388e-08]
 [  4.19230161e-03]
 [  3.67661156e-15]
 [  5.76301349e-15]
 [  9.93404975e-01]
 [  1.00000000e+00]
 [  3.46954688e-01]
 [  1.33151719e-18]
 [  1.00000000e+00]
 [  3.98874970e-23]
 [  1.00000000e+00]
 [  9.99999975e-01]
 [  1.36908331e-24]
 [  2.17675534e-06]
 [  1.00000000e+00]
 [  1.00000000e+00]
 [  1.00000000e+00]
 [  1.00000000e+00]
 [  1.00000000e+00]
 [  1.00000000e+00]
 [  1.00000000e+00]
 [  5.34702346e-31]
 [  4.93701639e-22]
 [  3.23689090e-14]
 [  1.00000000e+00]
 [  6.61782487e-03]
 [  1.00000000e+00]
 [  1.00000000e+00]
 [  1.00000000e+00]
 [  1.19539567e-41]
 [  2.87507466e-24]
 [  1.96864105e-45]
 [  3.86293428e-14]
 [  7.34917872e-11]
 [  9.99999997e-01]
 [  1.23349461e-21]
 [  1.00000000e+00]
 [  9.99999940e-01]
 [  8.62910776e-48]
 [  1.00000000e+00]
 [  1.00000000e+00]
 [  1.00000000e+00]
 [  1.00000000e+00]
 [  1.00000000e+00]
 [  9.99999997e-01]
 [  9.51953113e-01]
 [  1.19931816e-16]
 [  9.99999999e-01]
 [  1.00000000e+00]
 [  1.00000000e+00]
 [  1.00000000e+00]
 [  1.09424428e-05]
 [  1.00000000e+00]
 [  1.00000000e+00]
 [  1.95134375e-06]
 [  1.00000000e+00]
 [  1.00000000e+00]
 [  3.21484293e-10]
 [  1.00000000e+00]
 [  1.00000000e+00]
 [  2.47143486e-28]
 [  1.00000000e+00]
 [  1.00000000e+00]
 [  1.00000000e+00]
 [  9.76584208e-01]
 [  1.00000000e+00]
 [  3.35954358e-03]
 [  1.00000000e+00]]
```

<img width="50%" height="50%" src="https://github.com/morening/LearnML/blob/master/snapshot/logistic_regression/practice0114_data1.png?raw=true" />

### 测试结果（[data2](https://github.com/morening/LearnML/blob/master/data/data2.txt)）

***四阶拟合***

```
#正确率：85.59322033898306%

#拟合轮次：6469

#拟合参数：
[[ 3.20336492]
 [ 3.42216743]
 [-4.82267847]
 [-0.9102518 ]
 [-4.10029314]
 [ 2.05013743]
 [-4.26437115]
 [-0.75903534]
 [-1.05585157]
 [-5.70498231]
 [-1.21458307]
 [-1.40552181]
 [ 0.6064388 ]
 [ 0.50207037]
 [-4.64027097]]

#测试结果
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1]
```

<img width="50%" height="50%" src="https://github.com/morening/LearnML/blob/master/snapshot/logistic_regression/practice0114_data2.png?raw=true" />
