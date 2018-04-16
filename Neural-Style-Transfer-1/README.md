# Neural_style_improvisations
It's all about cool stuff with neural style.
This is a keras implementation of neural style described in the paper by [Gatys et.al](https://arxiv.org/abs/1508.06576).
The main objective of neural style is to generate an image that is as close as in terms of content to the base image and as close as in terms of style to the style image.

https://medium.com/data-science-group-iitr/artistic-style-transfer-with-convolutional-neural-network-7ce2476039fd



cifar100) axon:Neural-Style-Transfer-1 egomez$ python3 Keras_StyleTransfer.py
/Users/egomez/miniconda3/envs/cifar100/lib/python3.5/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated.In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.
/Users/egomez/miniconda3/envs/cifar100/lib/python3.5/importlib/_bootstrap.py:222: RuntimeWarning: compiletime version 3.6 of module 'tensorflow.python.framework.fast_tensor_util' does not match runtime version 3.5
  return f(*args, **kwds)
(1, 512, 512, 3)
(1, 512, 512, 3)
2018-04-10 19:15:10.219327: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.2
/Users/egomez/dev/MiTiC UAG/5 Cuatri/Deep Learning/deeplearning_repo/Proyecto #5/Neural-Style-Transfer-1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
Ya existen los pesos para esta configuración
Start of iteration 0
Position of the minimum [-68.66584358 -10.25413373  72.396128   ... -17.31963327 -71.1011142
  97.88485415] and its value 32279478000.0
Info dict (info): {'funcalls': 21, 'warnflag': 1, 'nit': 13, 'task': b'STOP: TOTAL NO. of f AND g EVALUATIONS EXCEEDS LIMIT', 'grad': array([ -675.45251465, -1070.33093262,   376.13977051, ...,
         294.19558716,   389.9755249 ,   478.76654053])}
0 if converged, 1 if too many function evaluations or too many iterations, 2 if stopped for another reason, given in info task
d[task] b'STOP: TOTAL NO. of f AND g EVALUATIONS EXCEEDS LIMIT'
Grad at Minimum: [ -675.45251465 -1070.33093262   376.13977051 ...   294.19558716
   389.9755249    478.76654053]
Function Calls Made: 21
Number Iterations: 13
Iteration 0 completed in 1140s
Start of iteration 1
Position of the minimum [ -49.39201019   23.88398481   63.0785059  ...  -36.52706501 -121.46432216
   72.17448815] and its value 23164158000.0
Info dict (info): {'funcalls': 21, 'warnflag': 1, 'nit': 15, 'task': b'STOP: TOTAL NO. of f AND g EVALUATIONS EXCEEDS LIMIT', 'grad': array([ 227.08477783,  217.57481384,  370.39434814, ...,  738.38470459,
       1423.44445801, 1277.71679688])}
0 if converged, 1 if too many function evaluations or too many iterations, 2 if stopped for another reason, given in info task
d[task] b'STOP: TOTAL NO. of f AND g EVALUATIONS EXCEEDS LIMIT'
Grad at Minimum: [ 227.08477783  217.57481384  370.39434814 ...  738.38470459 1423.44445801
 1277.71679688]
Function Calls Made: 21
Number Iterations: 15
Iteration 1 completed in 1104s
Start of iteration 2
Position of the minimum [ -55.85668395   17.86585007   46.03015378 ...  -74.32203742 -207.06309269
    4.19603496] and its value 19720935000.0
Info dict (info): {'funcalls': 21, 'warnflag': 1, 'nit': 16, 'task': b'STOP: TOTAL NO. of f AND g EVALUATIONS EXCEEDS LIMIT', 'grad': array([1939.12890625, 2392.44775391, 2381.34960938, ...,  362.62756348,
        -97.92776489,  373.46325684])}
0 if converged, 1 if too many function evaluations or too many iterations, 2 if stopped for another reason, given in info task
d[task] b'STOP: TOTAL NO. of f AND g EVALUATIONS EXCEEDS LIMIT'
Grad at Minimum: [1939.12890625 2392.44775391 2381.34960938 ...  362.62756348  -97.92776489
  373.46325684]
Function Calls Made: 21
Number Iterations: 16
Iteration 2 completed in 1036s
Start of iteration 3
Position of the minimum [-101.08828104  -37.41344224   -6.23103497 ...  -91.59684325 -220.02011092
  -15.07572329] and its value 18326129000.0
Info dict (info): {'funcalls': 21, 'warnflag': 1, 'nit': 16, 'task': b'STOP: TOTAL NO. of f AND g EVALUATIONS EXCEEDS LIMIT', 'grad': array([-270.04354858, -609.76098633, -562.50750732, ...,  360.78839111,
        615.01849365,  424.4987793 ])}
0 if converged, 1 if too many function evaluations or too many iterations, 2 if stopped for another reason, given in info task
d[task] b'STOP: TOTAL NO. of f AND g EVALUATIONS EXCEEDS LIMIT'
Grad at Minimum: [-270.04354858 -609.76098633 -562.50750732 ...  360.78839111  615.01849365
  424.4987793 ]
Function Calls Made: 21
Number Iterations: 16
Iteration 3 completed in 1152s
Start of iteration 4
Position of the minimum [-103.37011404  -32.33264045    2.36608699 ... -110.57165799 -241.78954383
  -32.06569418] and its value 17692176000.0
Info dict (info): {'funcalls': 21, 'warnflag': 1, 'nit': 17, 'task': b'STOP: TOTAL NO. of f AND g EVALUATIONS EXCEEDS LIMIT', 'grad': array([331.88415527, 119.89001465, 264.16094971, ..., 298.3112793 ,
        11.80232239, 172.39199829])}
0 if converged, 1 if too many function evaluations or too many iterations, 2 if stopped for another reason, given in info task
d[task] b'STOP: TOTAL NO. of f AND g EVALUATIONS EXCEEDS LIMIT'
Grad at Minimum: [331.88415527 119.89001465 264.16094971 ... 298.3112793   11.80232239
 172.39199829]
Function Calls Made: 21
Number Iterations: 17
Iteration 4 completed in 1093s
Start of iteration 5
Position of the minimum [-113.32474789  -39.42320419   -4.97539947 ... -121.89474416 -245.51627799
  -37.68455472] and its value 17367845000.0
Info dict (info): {'funcalls': 21, 'warnflag': 1, 'nit': 17, 'task': b'STOP: TOTAL NO. of f AND g EVALUATIONS EXCEEDS LIMIT', 'grad': array([ 100.59750366,   35.20730591, -114.04934692, ...,  342.80822754,
        358.15374756,  201.22393799])}
0 if converged, 1 if too many function evaluations or too many iterations, 2 if stopped for another reason, given in info task
d[task] b'STOP: TOTAL NO. of f AND g EVALUATIONS EXCEEDS LIMIT'
Grad at Minimum: [ 100.59750366   35.20730591 -114.04934692 ...  342.80822754  358.15374756
  201.22393799]
Function Calls Made: 21
Number Iterations: 17
Iteration 5 completed in 1876s
Start of iteration 6
Position of the minimum [-117.02547955  -41.29912714   -4.85882713 ... -131.76278103 -253.56069049
  -42.51138712] and its value 17190126000.0
Info dict (info): {'funcalls': 21, 'warnflag': 1, 'nit': 17, 'task': b'STOP: TOTAL NO. of f AND g EVALUATIONS EXCEEDS LIMIT', 'grad': array([ 44.59740067, -68.46583557,  62.04046631, ..., 240.44769287,
        44.74942398, 132.7640686 ])}
0 if converged, 1 if too many function evaluations or too many iterations, 2 if stopped for another reason, given in info task
d[task] b'STOP: TOTAL NO. of f AND g EVALUATIONS EXCEEDS LIMIT'
Grad at Minimum: [ 44.59740067 -68.46583557  62.04046631 ... 240.44769287  44.74942398
 132.7640686 ]
Function Calls Made: 21
Number Iterations: 17
Iteration 6 completed in 2039s
Start of iteration 7
Position of the minimum [-120.00946995  -42.94373914   -7.22849155 ... -137.91404657 -255.2243112
  -45.27784916] and its value 17075969000.0
Info dict (info): {'funcalls': 21, 'warnflag': 1, 'nit': 17, 'task': b'STOP: TOTAL NO. of f AND g EVALUATIONS EXCEEDS LIMIT', 'grad': array([178.74102783, 151.26974487, 110.90898132, ..., 182.41041565,
       229.19296265, 109.55125427])}
0 if converged, 1 if too many function evaluations or too many iterations, 2 if stopped for another reason, given in info task
d[task] b'STOP: TOTAL NO. of f AND g EVALUATIONS EXCEEDS LIMIT'
Grad at Minimum: [178.74102783 151.26974487 110.90898132 ... 182.41041565 229.19296265
 109.55125427]
Function Calls Made: 21
Number Iterations: 17
Iteration 7 completed in 1858s
Start of iteration 8
Position of the minimum [-123.55312255  -46.57460526   -8.80377646 ... -142.78911805 -259.56395453
  -47.12933532] and its value 16995633000.0
Info dict (info): {'funcalls': 21, 'warnflag': 1, 'nit': 18, 'task': b'STOP: TOTAL NO. of f AND g EVALUATIONS EXCEEDS LIMIT', 'grad': array([ 105.2568512 ,   55.44471741,   93.53216553, ...,  122.27689362,
       -143.40956116,  -24.1235714 ])}
0 if converged, 1 if too many function evaluations or too many iterations, 2 if stopped for another reason, given in info task
d[task] b'STOP: TOTAL NO. of f AND g EVALUATIONS EXCEEDS LIMIT'
Grad at Minimum: [ 105.2568512    55.44471741   93.53216553 ...  122.27689362 -143.40956116
  -24.1235714 ]
Function Calls Made: 21
Number Iterations: 18
Iteration 8 completed in 3396s
Start of iteration 9
Position of the minimum [-125.57150896  -48.30599047  -10.30681979 ... -147.06024784 -260.57034116
  -48.52544293] and its value 16934557000.0
Info dict (info): {'funcalls': 21, 'warnflag': 1, 'nit': 18, 'task': b'STOP: TOTAL NO. of f AND g EVALUATIONS EXCEEDS LIMIT', 'grad': array([ 95.91752625, 146.02241516,  81.50404358, ..., 255.88401794,
       335.1182251 , 173.59873962])}
0 if converged, 1 if too many function evaluations or too many iterations, 2 if stopped for another reason, given in info task
d[task] b'STOP: TOTAL NO. of f AND g EVALUATIONS EXCEEDS LIMIT'
Grad at Minimum: [ 95.91752625 146.02241516  81.50404358 ... 255.88401794 335.1182251
 173.59873962]
Function Calls Made: 21
Number Iterations: 18
Iteration 9 completed in 3259s
Exception ignored in: <bound method BaseSession.__del__ of <tensorflow.python.client.session.Session object at 0x11cf35dd8>>
Traceback (most recent call last):
  File "/Users/egomez/miniconda3/envs/cifar100/lib/python3.5/site-packages/tensorflow/python/client/session.py", line 702, in __del__
TypeError: 'NoneType' object is not callable