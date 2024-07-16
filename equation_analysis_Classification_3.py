# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 05:21:28 2023

@author: Nikola Andelic
"""
import re
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from collections import Counter
import warnings 
warnings.filterwarnings('ignore')

data_train = pd.read_csv("train data.csv")
data_test = pd.read_csv("test data.csv")
print(data_train.columns)
print(data_test.columns)
data_final = pd.concat([data_train, data_test], axis=0).reset_index()
data_final.pop('index')
print(data_final)
del (data_train, data_test)

print(data_final.info())
print(data_final.isnull().sum().sum())


# Creation of Fault_detection Columns
classes = data_final['class'].copy()
print(classes.to_list())
fault_detection = []
for i in range(len(classes)):
    #print(classes[i])
    if not classes[i] == 0:
        fault_detection.append(1)
    else:
        fault_detection.append(0)

data_final_2 = pd.concat([data_final, pd.DataFrame(np.array(fault_detection), columns = ['fault_det'])], axis = 1)
print(data_final_2)
y_raw = data_final_2.pop('class')

data_final_2.pop('fault_det')
print(data_final_2)

y = pd.get_dummies(y_raw, dtype='float', prefix="class")

print(y)
y_real_3 = y.pop('class_3')


# data_final_class_3 = pd.concat([data_final_2, y_real_3], axis = 1)
# print(data_final_class_3)
# data_final_class_3.to_csv("Fault_classification_original_class_3.csv", index = False)
# del data_final_class_3









## data_final_2.to_csv("PF_Classification_original_class_0.csv",
##                     index=False)
def log2(x):
    with np.errstate(divide = "ignore", invalid = "ignore"):
          return np.where(np.abs(x) > 0.001, np.log2(np.abs(x)),0.)
def log10(x):
    with np.errstate(divide = "ignore", invalid = "ignore"):
          return np.where(np.abs(x) > 0.001, np.log10(np.abs(x)),0.)
import scipy.special as sp 
def CubeRoot(x):
    return sp.cbrt(x)
def log(x1):
      with np.errstate(divide = "ignore", invalid = "ignore"):
          return np.where(np.abs(x1) > 0.001, np.log(np.abs(x1)),0.)
def sqrt(x1):
    return np.sqrt(np.abs(x1))
def div(x1,x2):
    with np.errstate(divide = "ignore", invalid = "ignore"):
        return np.where(np.abs(x2) > 0.001, np.divide(x1,x2), 1.) 
#############################################
# Original dataset 
# ADASYN, BorderlineSMOTE, SMOTE, SVMSMOTE 
#############################################
Original_all =[####################################################################################################
# Original + ADASYN Class 3 
######################################
'np.multiply(np.add(log10(np.add(X15, X18)), div(X19, X26)), min(np.subtract(np.multiply(np.add(log10(np.add(X15, X18)), np.add(X18, X22)), min(np.subtract(X6, X0), X29)), max(X4, X10)), X29))',
'np.subtract(np.multiply(X29, X20), log2(X18))',
'max(np.add(div(np.subtract(X3, X20), np.abs(X26)), np.cos(log2(np.add(X21, X25)))), np.subtract(np.abs(np.multiply(np.abs(div(log2(np.add(np.tan(X9), np.add(X21, X13))), min(np.abs(max(X9, X2)), div(np.multiply(X3, X29), np.add(sqrt(np.multiply(min(X6, X13), np.add(X17, X29))), np.cos(np.cos(CubeRoot(X5)))))))), np.cos(np.add(log(np.add(min(X1, X14), np.cos(X10))), np.cos(sqrt(np.abs(X9))))))), sqrt(div(X18, X16))))',
'np.subtract(np.multiply(X29, np.abs(-629.091)), X25)',
'log2(div(div(np.multiply(X19, np.multiply(np.subtract(X1, np.multiply(log2(X1), X21)), X24)), X29), np.add(div(max(X12, -671.352), sqrt(log2(np.abs(max(log(np.sin(X19)), max(np.add(X18, X20), max(X19, X0))))))), np.add(X23, X19))))',



######################################
# Original + BorderlineSMOTE Class 3 
###################################### 
'log2(np.add(np.multiply(max(X24, X24), log2(div(X19, X29))), np.tan(div(X10, X21))))',
'np.add(log10(sqrt(np.tan(np.sin(np.tan(np.abs(log(np.subtract(np.add(np.subtract(X4, X29), log10(X3)), log(np.abs(X29)))))))))), np.multiply(np.subtract(np.abs(max(np.abs(X17), np.tan(min(log(np.multiply(np.tan(sqrt(X8)), CubeRoot(np.tan(X27)))), log(np.add(np.tan(log10(X28)), max(div(X20, X3), log(X11)))))))), max(log(np.multiply(np.sin(log(np.sin(log2(div(X9, X23))))), log(np.sin(np.sin(np.abs(sqrt(X12))))))), div(np.subtract(div(sqrt(min(sqrt(np.add(X20, X19)), np.multiply(max(X7, X8), np.abs(X28)))), np.abs(np.add(np.multiply(np.multiply(X18, X12), min(X3, 712.472)), CubeRoot(max(X5, X0))))), np.abs(np.tan(min(np.tan(min(X2, X4)), log10(np.add(X21, X26)))))), log10(min(min(np.add(np.tan(np.subtract(X2, X29)), np.subtract(np.tan(X9), sqrt(X17))), log2(max(np.multiply(X22, X26), CubeRoot(X18)))), np.cos(np.cos(log(np.cos(X19))))))))), log2(log2(np.cos(np.add(np.tan(max(np.sin(-274.253), np.cos(np.add(div(X29, X21), CubeRoot(X29))))), np.cos(np.cos(np.sin(sqrt(np.subtract(X5, X3)))))))))))',
'log2(np.multiply(log(np.add(np.multiply(log10(log10(log2(-401.564))), log10(log10(sqrt(X19)))), div(np.add(X3, X21), X29))), div(X23, X29)))',
'np.subtract(div(log(np.subtract(sqrt(np.subtract(np.cos(np.add(log2(np.subtract(min(log10(sqrt(X6)), np.sin(np.sin(X20))), np.multiply(np.multiply(sqrt(X23), CubeRoot(X1)), np.abs(np.sin(X4))))), min(sqrt(np.sin(div(np.add(X16, X2), np.tan(X2)))), min(np.abs(np.subtract(log2(X14), min(X2, X4))), np.tan(np.cos(np.add(X23, X25))))))), np.cos(sqrt(max(np.tan(np.add(log2(np.tan(X29)), CubeRoot(max(-213.742, X19)))), div(X29, X7)))))), max(sqrt(X7), max(sqrt(log(np.sin(log10(div(np.sin(np.sin(np.tan(X25))), log10(np.add(min(X8, X18), np.add(X22, X5)))))))), max(np.subtract(np.cos(np.abs(log(max(log2(X15), div(X10, X9))))), min(log(np.sin(div(np.abs(X29), CubeRoot(X17)))), np.subtract(np.add(CubeRoot(CubeRoot(X25)), np.multiply(sqrt(X29), log2(X3))), log2(np.cos(log10(X0)))))), np.sin(div(np.add(np.add(np.cos(div(X2, X13)), CubeRoot(np.multiply(X9, X25))), log10(max(np.cos(X21), sqrt(X16)))), np.cos(min(np.cos(div(X13, X27)), np.tan(np.sin(X14))))))))))), np.abs(np.abs(np.abs(np.cos(max(max(min(div(log10(div(np.add(X7, X6), min(X8, X28))), log2(log2(np.cos(X25)))), sqrt(max(np.cos(np.subtract(X12, X10)), np.sin(np.cos(X11))))), sqrt(sqrt(X2))), np.tan(np.cos(np.sin(X7))))))))), np.multiply(np.cos(div(np.subtract(log10(np.subtract(log(np.abs(CubeRoot(np.add(max(div(X2, X23), log10(X14)), np.sin(np.sin(X3)))))), np.multiply(sqrt(np.subtract(sqrt(CubeRoot(np.cos(X17))), div(CubeRoot(np.sin(X3)), log10(np.sin(X1))))), np.multiply(np.tan(np.subtract(np.sin(div(X4, X22)), min(np.multiply(X1, X18), div(X25, X7)))), np.sin(np.sin(np.abs(log2(X24)))))))), np.sin(log10(np.subtract(log10(min(np.cos(min(log2(X21), CubeRoot(X8))), np.multiply(np.multiply(log2(X13), np.add(X17, 702.114)), log(log10(X17))))), np.add(np.abs(sqrt(log10(np.add(X1, 351.605)))), min(log2(div(np.subtract(X5, X26), np.abs(X25))), div(log(np.cos(X18)), log(np.multiply(X24, X25))))))))), np.abs(sqrt(div(max(log(np.abs(log10(X4))), np.multiply(sqrt(np.cos(np.multiply(np.sin(X29), sqrt(X0)))), log2(log10(log2(np.cos(X8)))))), max(np.add(np.add(max(CubeRoot(sqrt(X9)), min(sqrt(X23), np.cos(X0))), max(min(np.add(X10, X8), np.subtract(X17, X27)), sqrt(np.abs(X19)))), CubeRoot(np.abs(max(max(X26, X6), np.tan(X25))))), np.sin(log(log(np.sin(np.add(X15, X14))))))))))), np.add(log2(np.cos(np.sin(log10(np.subtract(X12, X11))))), np.sin(min(log10(sqrt(div(log(np.tan(min(np.tan(np.add(X29, X18)), log2(np.sin(X12))))), CubeRoot(sqrt(np.tan(np.multiply(np.cos(X26), np.subtract(X3, X29)))))))), np.tan(log2(sqrt(log2(sqrt(np.subtract(np.multiply(min(X16, X29), sqrt(X3)), min(np.multiply(X20, 748.581), min(X24, X15)))))))))))))',
'np.subtract(min(np.subtract(X21, X5), np.abs(np.multiply(X18, X29))), sqrt(235.503))',



#####################################
# Original + SMOTE Class 3 
####################################
'np.multiply(np.subtract(np.multiply(np.abs(log2(log2(np.sin(X27)))), sqrt(CubeRoot(CubeRoot(log(X29))))), np.sin(log(np.add(div(np.cos(-470.749), log2(np.abs(X23))), CubeRoot(max(X24, X20)))))), max(X22, X10))',
'np.subtract(div(np.multiply(X18, X5), sqrt(X29)), max(np.add(X26, X23), sqrt(log(X4))))',
'np.add(np.multiply(log(div(X15, X22)), sqrt(np.sin(X15))), CubeRoot(div(np.subtract(log2(log2(np.sin(sqrt(X5)))), X18), np.sin(min(np.multiply(np.subtract(np.subtract(X20, X23), div(-413.415, X19)), np.add(np.sin(X24), log2(X22))), div(max(np.tan(X4), div(X16, X20)), np.subtract(log10(X29), X29)))))))',
'min(np.multiply(X3, X18), log2(np.abs(div(CubeRoot(np.subtract(X4, min(np.multiply(np.multiply(min(np.multiply(np.multiply(np.subtract(X4, np.multiply(np.multiply(X23, X17), X24)), np.multiply(X19, X21)), X29), log10(np.cos(X16))), np.multiply(X17, X19)), X29), np.abs(X6)))), log10(np.multiply(np.multiply(np.multiply(X19, X11), np.add(np.multiply(np.subtract(np.add(X22, np.multiply(np.multiply(X19, X11), np.abs(X14))), X21), X26), X4)), np.multiply(np.add(np.multiply(np.subtract(np.add(X21, np.multiply(np.multiply(max(X10, np.multiply(np.multiply(np.multiply(np.multiply(X24, X11), div(-358.011, X7)), div(-358.011, X7)), X27)), X19), div(np.add(X29, np.multiply(np.multiply(np.multiply(np.add(X12, X21), X22), np.multiply(np.multiply(np.multiply(X20, X21), np.multiply(np.multiply(np.multiply(X24, X19), -446.560), np.multiply(X19, X21))), np.multiply(X19, X21))), np.multiply(np.add(np.multiply(np.multiply(X17, np.multiply(X23, X22)), np.multiply(np.abs(np.multiply(X17, X22)), X21)), X3), np.subtract(np.multiply(np.multiply(X20, X21), np.multiply(np.add(X25, np.multiply(X21, np.multiply(np.multiply(X24, X19), -446.560))), np.multiply(X19, X21))), X21)))), X4))), X21), X26), X4), np.multiply(max(-283.717, np.multiply(-399.161, X27)), X19))))))))',
'np.tan(np.tan(div(np.add(np.sin(np.add(np.sin(np.sin(np.cos(min(CubeRoot(np.abs(np.cos(X17))), min(div(log10(X25), np.sin(X12)), min(CubeRoot(X20), min(X16, X23))))))), sqrt(18.168))), log(div(X29, np.multiply(X26, np.subtract(X3, np.sin(X29)))))), div(sqrt(log(X7)), sqrt(log10(X29))))))',


#####################################
# Original + SVMSMOTE Class 3
#####################################
'log2(CubeRoot(np.abs(np.subtract(div(np.multiply(X24, np.multiply(X22, X17)), X29), np.cos(div(log(CubeRoot(min(np.tan(X12), np.cos(-399.444)))), max(max(sqrt(X3), X17), sqrt(log(X24)))))))))',
'np.subtract(sqrt(np.multiply(div(np.multiply(X22, X19), sqrt(X13)), log10(X29))), X25)',
'log2(np.add(log2(max(div(np.multiply(div(np.multiply(np.multiply(np.multiply(np.abs(div(np.multiply(np.multiply(X5, max(div(np.multiply(np.multiply(np.add(X24, X2), np.multiply(np.multiply(np.abs(np.multiply(div(X5, X18), div(X18, X15))), X19), X19)), np.multiply(X24, np.multiply(np.multiply(np.multiply(np.multiply(np.abs(X7), np.multiply(np.multiply(X19, X19), X19)), X19), np.multiply(np.multiply(X24, X25), X19)), X19))), np.sin(X27)), X0)), np.multiply(X17, np.multiply(X22, X19))), min(X15, X29))), X19), np.multiply(np.multiply(np.abs(div(np.multiply(np.multiply(np.multiply(X3, X19), max(div(np.multiply(np.multiply(X21, np.multiply(np.multiply(np.abs(div(np.multiply(np.multiply(X19, np.multiply(np.multiply(X19, X19), X19)), np.multiply(np.multiply(X14, np.multiply(np.multiply(X20, X19), X19)), np.multiply(X11, X19))), np.abs(X29))), X19), X19)), np.multiply(X24, np.multiply(np.multiply(np.multiply(X3, X19), np.multiply(np.multiply(X24, X25), X19)), X19))), np.sin(X27)), X0)), np.multiply(np.multiply(np.multiply(X19, np.multiply(np.multiply(X19, X19), X19)), np.multiply(np.multiply(np.multiply(X3, X19), np.multiply(np.multiply(X20, X19), X19)), sqrt(X11))), np.multiply(np.multiply(sqrt(max(X6, X22)), np.multiply(np.multiply(X3, X19), X19)), np.multiply(X11, X19)))), np.abs(X29))), X19), X19)), np.multiply(np.multiply(np.multiply(np.multiply(np.multiply(np.abs(div(np.multiply(np.multiply(X5, max(div(np.multiply(np.multiply(np.add(X24, X2), np.multiply(np.multiply(np.multiply(X10, X5), X19), X19)), np.multiply(X24, np.multiply(np.multiply(np.multiply(np.multiply(np.abs(X7), np.multiply(np.multiply(X19, X19), X19)), X19), np.multiply(np.multiply(X24, X25), X19)), X19))), np.sin(X27)), X0)), np.multiply(X17, np.multiply(X22, X19))), min(X15, X29))), X19), np.multiply(np.multiply(np.abs(div(np.multiply(np.multiply(np.multiply(X3, X19), max(div(np.multiply(np.multiply(X21, np.multiply(np.multiply(np.abs(div(np.multiply(np.multiply(X19, np.multiply(np.multiply(X19, X19), X19)), np.multiply(np.multiply(X14, np.multiply(np.multiply(X20, X19), X19)), np.multiply(X11, X19))), np.abs(X29))), X19), X19)), np.multiply(X24, np.multiply(np.multiply(np.multiply(X3, X19), np.multiply(np.multiply(X24, X25), X19)), X19))), np.sin(X27)), X0)), np.multiply(np.multiply(np.multiply(X19, np.multiply(np.multiply(X19, X19), X19)), np.multiply(np.multiply(np.multiply(X3, X19), np.multiply(np.multiply(X20, X19), X19)), sqrt(X11))), np.multiply(np.multiply(log(np.abs(div(X18, X14))), np.multiply(np.multiply(X3, X19), X19)), np.multiply(X11, X19)))), np.abs(X29))), X19), X19)), np.multiply(np.multiply(np.multiply(np.multiply(X3, X19), X19), X3), np.multiply(max(np.add(min(X20, X20), np.cos(X29)), min(np.sin(X6), np.sin(151.585))), X19))), X3), np.multiply(max(np.add(min(X20, X20), np.cos(X29)), min(np.sin(X6), np.sin(151.585))), X19))), np.abs(X29)), np.multiply(X19, X19)), np.abs(X29)), np.subtract(X0, X16))), div(sqrt(min(sqrt(X8), np.add(X25, X14))), X24)))',
'np.tan(np.tan(np.cos(np.subtract(CubeRoot(div(log2(np.subtract(div(max(X27, X24), np.subtract(X29, X4)), np.multiply(X20, np.abs(sqrt(div(np.tan(log2(sqrt(log(np.multiply(np.multiply(sqrt(np.add(np.tan(X11), np.abs(div(sqrt(np.sin(np.abs(X8))), min(np.subtract(np.multiply(np.add(np.abs(np.cos(max(log10(log2(log(X24))), log2(X22)))), min(div(X17, X18), np.subtract(X13, X25))), log2(np.abs(np.add(np.abs(X15), min(np.tan(X22), X26))))), log(np.add(div(np.tan(X23), log2(X20)), np.abs(log(X13))))), log2(log(div(np.sin(np.cos(X7)), log2(log(X24)))))))))), sqrt(sqrt(X24))), np.abs(log2(np.add(X20, X7)))))))), np.tan(np.abs(CubeRoot(CubeRoot(log2(np.tan(np.sin(np.tan(X1)))))))))))))), log10(div(np.sin(X29), X15)))), log10(div(log(div(X16, X23)), X20))))))',
'np.subtract(np.multiply(min(X3, X29), np.multiply(div(X27, X29), np.multiply(X17, X20))), log2(X19))'


 ]



StandardScaler_raw = [####################################################################################################
# Standard Scaler + ADASYN CLASS 3 
####################################################################################################
'np.multiply(X29, np.add(np.abs(div(log(np.multiply(np.tan(X10), np.abs(log2(X10)))), np.tan(np.multiply(log10(np.add(max(np.sin(div(np.abs(X26), np.abs(sqrt(CubeRoot(X24))))), np.cos(min(np.sin(np.tan(min(X7, X27))), np.abs(log2(sqrt(X8)))))), np.multiply(np.subtract(sqrt(log(div(CubeRoot(X8), log(np.multiply(X0, X14))))), max(np.cos(np.subtract(sqrt(X23), np.cos(X13))), np.sin(np.multiply(np.tan(X9), np.tan(X5))))), max(np.multiply(div(log2(np.abs(X18)), np.add(min(X5, X10), np.sin(X12))), sqrt(np.abs(min(X0, X13)))), log10(sqrt(sqrt(min(X15, X1)))))))), np.multiply(np.tan(X27), min(X0, X19)))))), div(max(-726.753, div(np.sin(CubeRoot(X8)), X29)), np.multiply(CubeRoot(X8), np.subtract(X8, X15)))))',
'np.add(np.add(CubeRoot(np.add(CubeRoot(sqrt(X2)), np.add(X29, np.add(X29, np.tan(CubeRoot(X29)))))), np.add(np.add(np.add(np.add(np.add(np.add(CubeRoot(np.add(CubeRoot(np.subtract(np.sin(min(X1, -545.423)), log10(X17))), np.add(X29, np.add(X29, np.tan(CubeRoot(X29)))))), np.add(X29, np.add(X29, np.tan(CubeRoot(X29))))), np.add(X29, np.add(X29, sqrt(min(log2(np.add(X29, np.tan(CubeRoot(X29)))), np.tan(div(X5, -331.588))))))), np.add(X29, np.add(np.add(CubeRoot(np.add(CubeRoot(np.subtract(np.sin(min(X1, -545.423)), log10(X17))), np.add(X29, np.add(X29, np.tan(CubeRoot(X29)))))), np.add(X29, np.add(X29, np.tan(CubeRoot(X29))))), np.add(X29, np.add(X29, sqrt(min(log2(np.add(X29, np.tan(CubeRoot(X29)))), np.tan(div(X5, -331.588))))))))), np.add(X29, np.add(np.add(CubeRoot(np.add(CubeRoot(np.subtract(np.sin(min(X1, -545.423)), log10(X17))), np.add(X29, np.add(X29, np.tan(CubeRoot(X29)))))), np.add(X29, np.add(X29, np.tan(CubeRoot(X29))))), np.add(X29, np.add(X29, sqrt(min(log2(np.add(X29, np.tan(CubeRoot(X29)))), np.tan(div(X5, -331.588))))))))), np.add(X29, np.add(X29, sqrt(min(log2(np.add(X29, np.tan(CubeRoot(X29)))), np.tan(div(X5, -331.588))))))), np.add(X29, np.tan(CubeRoot(X29))))), np.add(X29, sqrt(min(log2(np.add(X29, np.tan(CubeRoot(X29)))), np.tan(X1)))))',
'np.tan(np.subtract(CubeRoot(np.tan(CubeRoot(np.tan(CubeRoot(CubeRoot(np.subtract(CubeRoot(CubeRoot(np.tan(np.subtract(CubeRoot(np.tan(np.subtract(CubeRoot(CubeRoot(np.subtract(CubeRoot(np.tan(CubeRoot(CubeRoot(CubeRoot(np.subtract(CubeRoot(CubeRoot(np.tan(np.subtract(CubeRoot(np.tan(np.subtract(CubeRoot(CubeRoot(CubeRoot(np.subtract(CubeRoot(CubeRoot(CubeRoot(np.tan(np.subtract(CubeRoot(np.tan(np.subtract(CubeRoot(CubeRoot(CubeRoot(np.subtract(CubeRoot(np.tan(CubeRoot(np.subtract(CubeRoot(np.tan(np.subtract(np.subtract(CubeRoot(CubeRoot(CubeRoot(np.subtract(CubeRoot(CubeRoot(CubeRoot(np.tan(np.subtract(CubeRoot(np.tan(np.subtract(CubeRoot(CubeRoot(CubeRoot(np.subtract(CubeRoot(np.tan(CubeRoot(np.subtract(CubeRoot(np.tan(np.subtract(CubeRoot(np.subtract(CubeRoot(CubeRoot(np.subtract(CubeRoot(CubeRoot(CubeRoot(np.subtract(CubeRoot(CubeRoot(np.subtract(X8, np.add(np.subtract(X1, X1), X16)))), X12)))), X4))), X12)), X12))), X12)))), sqrt(np.subtract(X27, X3)))))), X12))), X12))))), X12)))), X12), X12))), X12)))), log10(np.tan(np.subtract(CubeRoot(np.tan(np.subtract(CubeRoot(np.subtract(CubeRoot(CubeRoot(np.subtract(CubeRoot(CubeRoot(np.abs(log2(log(X22))))), X4))), X12)), X12))), X12))))))), X12))), X12))))), X12)))), X12))), X12)))), X12)))))), log10(CubeRoot(log10(log(np.sin(log10(CubeRoot(np.multiply(np.cos(X22), np.add(-717.168, X13)))))))))))), X12))), X12)))), np.cos(max(log10(np.tan(np.abs(np.cos(log(log2(CubeRoot(log2(X28)))))))), log(np.subtract(np.tan(np.subtract(CubeRoot(np.tan(np.subtract(CubeRoot(np.subtract(CubeRoot(CubeRoot(np.subtract(CubeRoot(CubeRoot(CubeRoot(np.subtract(CubeRoot(CubeRoot(np.subtract(X8, np.add(np.subtract(X1, X1), X16)))), X12)))), X4))), X12)), X12))), X12)), X29))))))))))), X12))',
'np.subtract(np.tan(np.tan(CubeRoot(CubeRoot(CubeRoot(CubeRoot(CubeRoot(CubeRoot(CubeRoot(np.tan(np.tan(CubeRoot(CubeRoot(CubeRoot(CubeRoot(CubeRoot(CubeRoot(CubeRoot(CubeRoot(np.subtract(CubeRoot(CubeRoot(CubeRoot(np.tan(CubeRoot(np.subtract(CubeRoot(np.subtract(CubeRoot(CubeRoot(CubeRoot(CubeRoot(CubeRoot(CubeRoot(CubeRoot(CubeRoot(CubeRoot(CubeRoot(CubeRoot(np.subtract(CubeRoot(CubeRoot(CubeRoot(CubeRoot(np.subtract(CubeRoot(np.subtract(CubeRoot(CubeRoot(CubeRoot(CubeRoot(CubeRoot(CubeRoot(CubeRoot(CubeRoot(CubeRoot(np.subtract(np.subtract(X29, X12), X12)))))))))), sqrt(min(sqrt(X24), np.sin(log10(np.sin(log10(log10(X5))))))))), X12))))), X12)))))))))))), np.cos(np.sin(X13)))), X12)))))), X12)))))))))))))))))))), np.add(X27, X27))',
'np.add(CubeRoot(CubeRoot(CubeRoot(np.add(CubeRoot(np.add(np.subtract(CubeRoot(np.add(CubeRoot(np.add(np.subtract(X29, X1), X3)), X3)), X1), X3)), X3)))), np.add(CubeRoot(CubeRoot(np.add(np.sin(np.multiply(X9, np.multiply(log(X17), X28))), CubeRoot(np.add(CubeRoot(CubeRoot(CubeRoot(np.add(CubeRoot(np.add(CubeRoot(np.add(np.subtract(CubeRoot(np.add(CubeRoot(np.add(np.subtract(X29, X1), X3)), X3)), X1), X3)), X3)), X3)))), X3))))), np.add(CubeRoot(CubeRoot(CubeRoot(np.add(CubeRoot(np.add(CubeRoot(np.add(np.subtract(CubeRoot(np.add(CubeRoot(np.add(np.subtract(X29, X1), X3)), X3)), X1), X3)), X3)), X3)))), np.add(np.add(CubeRoot(CubeRoot(CubeRoot(np.add(CubeRoot(np.add(CubeRoot(np.add(np.subtract(CubeRoot(np.add(CubeRoot(np.add(np.subtract(X29, X1), X3)), X3)), X1), X3)), X3)), X3)))), np.add(CubeRoot(CubeRoot(np.add(np.sin(X29), CubeRoot(np.add(CubeRoot(CubeRoot(np.add(CubeRoot(CubeRoot(CubeRoot(np.add(CubeRoot(np.add(np.subtract(CubeRoot(np.add(CubeRoot(np.add(np.subtract(X29, X1), X3)), X3)), X1), X3)), X3)))), np.add(CubeRoot(CubeRoot(np.add(np.sin(np.multiply(X9, np.multiply(log(X17), X28))), CubeRoot(np.add(CubeRoot(CubeRoot(CubeRoot(np.add(CubeRoot(np.add(CubeRoot(np.add(np.subtract(CubeRoot(np.add(CubeRoot(np.add(np.subtract(X29, X1), X3)), X3)), X1), X3)), X3)), X3)))), X3))))), np.add(CubeRoot(CubeRoot(CubeRoot(np.add(CubeRoot(np.add(CubeRoot(np.add(np.subtract(CubeRoot(np.add(CubeRoot(np.add(np.subtract(X29, X1), X3)), X3)), X1), X3)), X3)), X3)))), np.add(np.add(CubeRoot(CubeRoot(CubeRoot(np.add(CubeRoot(np.add(CubeRoot(np.add(np.subtract(CubeRoot(np.add(CubeRoot(np.add(np.subtract(X29, X1), X3)), X3)), X1), X3)), X3)), X3)))), np.add(CubeRoot(CubeRoot(np.add(np.sin(X29), CubeRoot(np.add(CubeRoot(CubeRoot(np.add(CubeRoot(np.add(np.subtract(X29, X1), X3)), X3))), X3))))), CubeRoot(np.add(CubeRoot(np.add(np.subtract(CubeRoot(np.add(CubeRoot(np.add(np.subtract(X29, X1), X3)), X3)), X1), np.add(CubeRoot(np.add(np.add(CubeRoot(CubeRoot(CubeRoot(np.add(CubeRoot(np.add(CubeRoot(np.add(np.subtract(CubeRoot(np.add(CubeRoot(np.add(np.subtract(X29, X1), X3)), X3)), X1), X3)), X3)), X3)))), np.add(CubeRoot(CubeRoot(np.add(div(X28, X8), CubeRoot(CubeRoot(CubeRoot(np.add(CubeRoot(np.add(np.subtract(X29, X1), X3)), X3))))))), CubeRoot(np.add(np.subtract(CubeRoot(np.add(CubeRoot(CubeRoot(CubeRoot(CubeRoot(np.add(CubeRoot(np.add(CubeRoot(np.add(np.subtract(CubeRoot(np.add(CubeRoot(np.add(np.subtract(X29, X1), X3)), X3)), X1), X29)), X3)), X3))))), X3)), np.tan(X6)), X3)))), X3)), X3))), X3)))), CubeRoot(np.add(CubeRoot(np.add(np.subtract(CubeRoot(np.add(CubeRoot(np.add(np.subtract(X29, X1), X3)), X3)), X1), np.add(CubeRoot(np.add(np.add(CubeRoot(CubeRoot(CubeRoot(np.add(CubeRoot(np.add(CubeRoot(np.add(np.subtract(CubeRoot(np.add(CubeRoot(np.add(np.subtract(X29, X1), X3)), X3)), X1), X3)), X3)), X3)))), np.add(CubeRoot(CubeRoot(np.add(CubeRoot(X26), CubeRoot(np.add(CubeRoot(CubeRoot(np.add(CubeRoot(np.add(np.subtract(X29, X1), X3)), X3))), X3))))), CubeRoot(np.add(np.subtract(CubeRoot(np.add(CubeRoot(CubeRoot(CubeRoot(CubeRoot(np.add(CubeRoot(np.add(CubeRoot(np.add(np.subtract(CubeRoot(np.add(CubeRoot(np.add(np.subtract(X29, X1), X3)), X3)), X1), X29)), X3)), X3))))), X3)), np.tan(X6)), X3)))), X3)), X3))), X3)))))))), X3))))), CubeRoot(np.add(CubeRoot(np.add(np.subtract(CubeRoot(np.add(np.add(CubeRoot(CubeRoot(CubeRoot(np.add(CubeRoot(np.add(np.subtract(CubeRoot(np.add(CubeRoot(np.add(np.subtract(X29, X1), X3)), X3)), X1), X3)), X3)))), np.add(CubeRoot(CubeRoot(np.add(np.sin(np.multiply(X9, np.multiply(log(X17), X28))), CubeRoot(np.add(CubeRoot(CubeRoot(CubeRoot(np.add(CubeRoot(np.add(CubeRoot(np.add(np.subtract(CubeRoot(np.add(CubeRoot(np.add(np.subtract(X29, X1), X3)), X3)), X1), X3)), X3)), X3)))), X3))))), np.add(CubeRoot(CubeRoot(CubeRoot(np.add(CubeRoot(np.add(CubeRoot(np.add(np.subtract(CubeRoot(np.add(CubeRoot(np.add(np.subtract(X29, X1), X3)), X3)), X1), X3)), X3)), X3)))), np.add(np.add(CubeRoot(CubeRoot(CubeRoot(np.add(CubeRoot(np.add(CubeRoot(np.add(np.subtract(CubeRoot(np.add(CubeRoot(np.add(np.subtract(X29, X1), X3)), X3)), X1), X3)), X3)), X3)))), np.add(CubeRoot(CubeRoot(np.add(np.sin(X29), CubeRoot(np.add(CubeRoot(CubeRoot(np.add(CubeRoot(np.add(np.subtract(X29, X1), X3)), X3))), X3))))), CubeRoot(np.add(CubeRoot(np.add(np.subtract(CubeRoot(np.add(CubeRoot(np.add(np.subtract(X29, X1), X3)), X3)), X1), np.add(CubeRoot(np.add(np.add(CubeRoot(CubeRoot(CubeRoot(np.add(CubeRoot(np.add(CubeRoot(np.add(np.subtract(CubeRoot(np.add(CubeRoot(np.add(np.subtract(X29, X1), X3)), X3)), X1), X3)), X3)), X3)))), np.add(CubeRoot(CubeRoot(np.add(div(X28, X8), CubeRoot(CubeRoot(CubeRoot(np.add(CubeRoot(np.add(np.subtract(X29, X1), X3)), X3))))))), CubeRoot(np.add(np.subtract(CubeRoot(np.add(CubeRoot(CubeRoot(CubeRoot(CubeRoot(np.add(CubeRoot(np.add(CubeRoot(np.add(np.subtract(CubeRoot(np.add(CubeRoot(np.add(np.subtract(X29, X1), X3)), X3)), X1), X29)), X3)), X3))))), X3)), np.tan(X6)), X3)))), X3)), X3))), X3)))), CubeRoot(np.add(CubeRoot(np.add(np.subtract(CubeRoot(np.add(CubeRoot(np.add(np.subtract(X29, X1), X3)), X3)), X1), np.add(CubeRoot(np.add(np.add(CubeRoot(CubeRoot(CubeRoot(np.add(CubeRoot(np.add(CubeRoot(np.add(np.subtract(CubeRoot(np.add(CubeRoot(np.add(np.subtract(X29, X1), X3)), X3)), X1), X3)), X3)), X3)))), np.add(CubeRoot(CubeRoot(np.add(CubeRoot(X26), CubeRoot(np.add(CubeRoot(CubeRoot(np.add(CubeRoot(np.add(np.subtract(X29, X1), X3)), X3))), X3))))), CubeRoot(np.add(np.subtract(CubeRoot(np.add(CubeRoot(CubeRoot(CubeRoot(CubeRoot(np.add(CubeRoot(np.add(CubeRoot(np.add(np.subtract(CubeRoot(np.add(CubeRoot(np.add(np.subtract(X29, X1), X3)), X3)), X1), X29)), X3)), X3))))), X3)), np.tan(X6)), X3)))), X3)), X3))), X3)))))), X3)), X1), np.add(CubeRoot(np.add(np.add(CubeRoot(CubeRoot(CubeRoot(np.add(CubeRoot(np.add(CubeRoot(np.add(np.subtract(CubeRoot(np.add(CubeRoot(np.add(np.subtract(X29, X1), X3)), X3)), X1), X3)), X3)), X3)))), np.add(CubeRoot(CubeRoot(np.add(div(X28, X8), CubeRoot(CubeRoot(CubeRoot(np.add(CubeRoot(np.add(np.subtract(X29, X1), X3)), X3))))))), CubeRoot(np.add(np.subtract(CubeRoot(np.add(CubeRoot(CubeRoot(CubeRoot(CubeRoot(np.add(CubeRoot(np.add(CubeRoot(np.add(np.subtract(CubeRoot(np.add(CubeRoot(np.add(np.subtract(X29, X1), X3)), X3)), X1), X29)), X3)), X3))))), X3)), np.tan(X6)), X3)))), X3)), X3))), X3)))), CubeRoot(np.add(CubeRoot(np.add(np.subtract(CubeRoot(np.add(CubeRoot(np.add(np.subtract(X29, X1), X3)), X3)), X1), np.add(CubeRoot(np.add(np.add(CubeRoot(CubeRoot(CubeRoot(np.add(CubeRoot(np.add(CubeRoot(np.add(np.subtract(CubeRoot(np.add(CubeRoot(np.add(np.subtract(X29, X1), X3)), X3)), X1), X3)), X3)), X3)))), np.add(CubeRoot(CubeRoot(np.add(CubeRoot(X26), CubeRoot(np.add(CubeRoot(CubeRoot(np.add(CubeRoot(np.add(np.subtract(X29, X1), X3)), X3))), X3))))), CubeRoot(np.add(np.subtract(CubeRoot(np.add(CubeRoot(CubeRoot(CubeRoot(CubeRoot(np.add(CubeRoot(np.add(CubeRoot(np.add(np.subtract(CubeRoot(np.add(CubeRoot(np.add(np.subtract(X29, X1), X3)), X3)), X1), X29)), X3)), X3))))), X3)), np.tan(X6)), X3)))), X3)), X3))), X3))))))',
####################################################################################################
# Standard Scaler + BorderlineSMOTE Class 3 
####################################################################################################
'np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.cos(X0), X29), X29), X29), X29), X29), np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.add(sqrt(log2(np.subtract(X16, -833.589))), X29), np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.add(CubeRoot(X29), X29), np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.cos(X0), X29), X29), X29), X29), X29), np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.add(sqrt(log2(np.subtract(X16, -833.589))), X29), np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.add(CubeRoot(X29), X29), np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.cos(X0), X29), X29), X29), X29), X29), np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.cos(X0), X29), X29), X29), X29), X29), CubeRoot(X29)), X29), X29), X29)), X29)), X29), X29), X29), np.abs(np.add(sqrt(X27), np.cos(log10(X26))))), X29), X29), X29), X29)), X29), X29), X29), min(CubeRoot(np.multiply(max(np.multiply(log(np.tan(log10(np.add(log10(X22), max(X11, X13))))), div(np.subtract(np.cos(log10(np.sin(X10))), sqrt(CubeRoot(log(log10(X9))))), log10(np.tan(log10(np.sin(X27)))))), np.multiply(np.cos(np.sin(log2(np.subtract(np.abs(X8), np.add(X6, X1))))), log10(np.sin(np.multiply(CubeRoot(min(X23, X26)), sqrt(max(X29, X15))))))), max(log(np.tan(np.multiply(np.tan(-800.359), np.sin(CubeRoot(CubeRoot(X18)))))), np.sin(np.tan(log2(X15)))))), np.abs(np.tan(sqrt(min(np.abs(np.add(log(log(np.subtract(-379.954, X15))), np.tan(log10(sqrt(X13))))), CubeRoot(log2(min(np.add(div(X20, X9), np.add(X4, X5)), np.add(log10(X22), np.abs(X21))))))))))), X29), X29), X29), X29), X29), CubeRoot(X29)), X29), X29), X29)), X29)), X29), X29), X29), np.abs(np.add(sqrt(X27), np.cos(log10(X26))))), X29), X29), X29), X29)), X29), X29), X29), min(CubeRoot(np.multiply(max(np.multiply(log(np.tan(log10(np.add(log10(X22), max(X11, X13))))), div(np.subtract(np.cos(log10(np.sin(X10))), sqrt(CubeRoot(log(log10(X9))))), log10(np.tan(log10(np.sin(X27)))))), np.multiply(np.cos(np.sin(log2(np.subtract(np.abs(X8), np.add(X6, X1))))), log10(np.sin(np.multiply(CubeRoot(min(X23, X26)), sqrt(max(X29, X15))))))), max(log(np.tan(np.multiply(np.tan(-800.359), np.sin(CubeRoot(CubeRoot(X18)))))), np.sin(np.tan(log2(X15)))))), np.abs(np.tan(sqrt(min(np.abs(np.add(log(log(np.subtract(-379.954, X15))), np.tan(log10(sqrt(X13))))), CubeRoot(log2(min(np.add(div(X20, X9), np.add(X4, X5)), np.add(log10(X22), np.abs(X21))))))))))), X29), X29), X29)), X29), X29), X29), X29)',
'np.add(X29, np.add(np.add(X29, np.add(np.add(CubeRoot(CubeRoot(CubeRoot(CubeRoot(CubeRoot(CubeRoot(CubeRoot(CubeRoot(np.add(X29, X28))))))))), CubeRoot(np.add(CubeRoot(CubeRoot(np.add(CubeRoot(CubeRoot(CubeRoot(CubeRoot(np.add(X29, X28))))), np.add(CubeRoot(CubeRoot(CubeRoot(CubeRoot(np.add(X29, X28))))), CubeRoot(np.add(CubeRoot(CubeRoot(CubeRoot(CubeRoot(np.add(X29, X28))))), CubeRoot(max(X29, log(X26))))))))), np.add(X29, np.add(np.add(CubeRoot(CubeRoot(CubeRoot(CubeRoot(np.add(X29, X28))))), CubeRoot(np.add(CubeRoot(CubeRoot(np.add(CubeRoot(CubeRoot(CubeRoot(CubeRoot(np.add(X29, X28))))), np.add(CubeRoot(CubeRoot(CubeRoot(CubeRoot(np.add(X29, X28))))), CubeRoot(np.add(log2(np.abs(np.sin(sqrt(np.subtract(X10, X15))))), np.add(X29, X28))))))), CubeRoot(np.add(CubeRoot(CubeRoot(CubeRoot(CubeRoot(np.add(X29, X28))))), CubeRoot(np.multiply(X22, X3))))))), np.add(CubeRoot(CubeRoot(CubeRoot(CubeRoot(CubeRoot(CubeRoot(CubeRoot(np.add(X29, X28)))))))), CubeRoot(CubeRoot(np.add(CubeRoot(np.add(CubeRoot(CubeRoot(CubeRoot(np.add(CubeRoot(CubeRoot(CubeRoot(CubeRoot(CubeRoot(CubeRoot(sqrt(X1))))))), CubeRoot(CubeRoot(np.add(CubeRoot(CubeRoot(CubeRoot(CubeRoot(np.add(X29, X28))))), CubeRoot(np.add(np.add(X1, np.subtract(X6, X7)), CubeRoot(max(X29, log(np.add(log10(np.sin(np.cos(X7))), max(np.multiply(np.sin(sqrt(X20)), sqrt(np.cos(X19))), min(min(sqrt(X23), np.abs(X15)), np.sin(np.multiply(X24, X5))))))))))))))))), np.add(X29, np.add(CubeRoot(CubeRoot(np.add(CubeRoot(CubeRoot(CubeRoot(CubeRoot(np.add(X29, X28))))), CubeRoot(np.add(CubeRoot(np.sin(X6)), CubeRoot(max(X29, np.cos(div(X25, log2(X11)))))))))), np.add(CubeRoot(CubeRoot(CubeRoot(CubeRoot(CubeRoot(CubeRoot(CubeRoot(np.add(X29, X28)))))))), CubeRoot(CubeRoot(np.add(CubeRoot(CubeRoot(CubeRoot(CubeRoot(np.add(X29, X28))))), CubeRoot(np.add(np.sin(X28), CubeRoot(max(X29, log(np.add(log10(np.sin(log(X14))), CubeRoot(CubeRoot(np.abs(X23))))))))))))))))), CubeRoot(np.add(CubeRoot(CubeRoot(CubeRoot(CubeRoot(np.add(X29, X28))))), np.add(X29, np.add(CubeRoot(CubeRoot(np.add(CubeRoot(CubeRoot(CubeRoot(CubeRoot(np.add(X29, X28))))), CubeRoot(np.multiply(X22, X3))))), np.add(CubeRoot(CubeRoot(CubeRoot(CubeRoot(CubeRoot(CubeRoot(CubeRoot(np.add(X29, X28)))))))), CubeRoot(CubeRoot(np.add(CubeRoot(CubeRoot(CubeRoot(CubeRoot(np.add(X29, X28))))), min(np.cos(np.abs(X29)), np.sin(log10(-60.398)))))))))))))))))))), np.add(CubeRoot(CubeRoot(CubeRoot(CubeRoot(CubeRoot(CubeRoot(CubeRoot(np.add(X29, X28)))))))), CubeRoot(CubeRoot(np.add(CubeRoot(np.add(CubeRoot(CubeRoot(CubeRoot(np.add(CubeRoot(CubeRoot(CubeRoot(CubeRoot(CubeRoot(CubeRoot(CubeRoot(np.add(X29, X28)))))))), CubeRoot(CubeRoot(np.add(CubeRoot(CubeRoot(CubeRoot(CubeRoot(np.add(X29, X28))))), CubeRoot(np.add(np.add(X1, np.subtract(X6, X7)), CubeRoot(max(X29, log(np.add(log10(np.cos(X25)), CubeRoot(CubeRoot(np.abs(X23)))))))))))))))), np.add(X29, np.add(CubeRoot(CubeRoot(CubeRoot(CubeRoot(np.add(X29, X28))))), np.add(X29, np.add(CubeRoot(CubeRoot(np.add(CubeRoot(CubeRoot(CubeRoot(CubeRoot(np.add(X29, X28))))), CubeRoot(np.multiply(X22, X3))))), np.add(CubeRoot(CubeRoot(CubeRoot(CubeRoot(CubeRoot(CubeRoot(CubeRoot(np.add(X29, X28)))))))), CubeRoot(CubeRoot(np.add(CubeRoot(CubeRoot(CubeRoot(CubeRoot(np.add(X29, X28))))), CubeRoot(np.add(CubeRoot(CubeRoot(CubeRoot(CubeRoot(np.add(X29, X28))))), min(np.subtract(log(max(-536.978, X25)), max(X22, X22)), CubeRoot(sqrt(X1))))))))))))))), CubeRoot(np.add(CubeRoot(CubeRoot(CubeRoot(CubeRoot(np.add(X29, X28))))), np.add(X29, np.add(CubeRoot(CubeRoot(np.add(CubeRoot(CubeRoot(CubeRoot(CubeRoot(np.add(X29, X28))))), CubeRoot(np.multiply(X22, X3))))), np.add(CubeRoot(CubeRoot(CubeRoot(CubeRoot(CubeRoot(CubeRoot(CubeRoot(np.add(X29, X28)))))))), CubeRoot(CubeRoot(np.add(CubeRoot(CubeRoot(CubeRoot(CubeRoot(np.add(X29, X28))))), CubeRoot(np.add(CubeRoot(CubeRoot(CubeRoot(CubeRoot(np.add(X29, X28))))), min(np.subtract(log(max(-536.978, X25)), log(X1)), CubeRoot(sqrt(X1))))))))))))))))))), np.add(X29, np.add(CubeRoot(CubeRoot(np.add(CubeRoot(CubeRoot(CubeRoot(CubeRoot(np.add(X29, X28))))), CubeRoot(np.multiply(X22, X3))))), np.add(CubeRoot(CubeRoot(CubeRoot(CubeRoot(CubeRoot(CubeRoot(CubeRoot(np.add(X29, X28)))))))), CubeRoot(CubeRoot(np.add(CubeRoot(CubeRoot(CubeRoot(CubeRoot(np.add(X29, X28))))), CubeRoot(np.add(CubeRoot(CubeRoot(CubeRoot(CubeRoot(np.add(X29, X28))))), min(np.subtract(log(max(-536.978, X25)), log(X1)), CubeRoot(sqrt(X1)))))))))))))',
'np.multiply(CubeRoot(np.subtract(np.subtract(np.subtract(np.subtract(np.subtract(np.add(div(log2(-472.000), np.multiply(-4.257, X15)), div(log10(np.add(div(np.add(sqrt(np.tan(X1)), log2(sqrt(X26))), log2(X7)), min(np.add(div(max(log(div(log10(log10(np.multiply(X29, X15))), sqrt(sqrt(div(X14, X15))))), CubeRoot(np.add(log(min(div(X17, X22), np.subtract(X19, X25))), np.subtract(div(div(X26, X8), CubeRoot(X16)), np.add(np.cos(X20), np.sin(X22)))))), np.multiply(CubeRoot(div(div(X3, X3), div(sqrt(X23), np.multiply(X2, X17)))), log2(np.sin(X27)))), np.add(div(np.sin(log2(min(np.tan(np.multiply(X27, X2)), log(CubeRoot(X4))))), np.tan(np.abs(div(CubeRoot(CubeRoot(sqrt(np.subtract(np.tan(np.cos(log2(X18))), min(sqrt(np.sin(-543.068)), sqrt(np.cos(X4))))))), log2(log10(X4)))))), log(max(np.tan(log10(X29)), min(sqrt(sqrt(log(X22))), np.cos(X14)))))), np.subtract(X13, X24)))), max(X15, X25))), -869.364), -854.940), -869.364), -854.940), -854.940)), X29)',
'np.multiply(X29, np.multiply(np.subtract(min(np.multiply(X29, np.multiply(np.add(min(-357.080, min(X27, X3)), -540.876), log2(np.multiply(log10(np.sin(X29)), np.tan(log(log2(max(CubeRoot(np.tan(X1)), 261.626)))))))), log(div(np.add(np.cos(np.tan(X12)), np.multiply(log2(X29), np.abs(log2(max(log(np.cos(max(X27, min(log2(X17), np.tan(min(min(CubeRoot(np.cos(np.multiply(CubeRoot(np.add(div(X27, X13), min(X10, X7))), CubeRoot(log10(np.add(X1, X9)))))), sqrt(sqrt(np.tan(np.subtract(np.tan(min(X16, X3)), np.abs(sqrt(X7))))))), log(div(np.sin(min(np.multiply(np.abs(np.cos(X28)), np.cos(CubeRoot(X0))), CubeRoot(log2(np.subtract(-370.600, X7))))), div(np.multiply(np.abs(log10(np.subtract(X29, -514.582))), np.sin(np.multiply(np.sin(X19), log(X14)))), np.sin(log2(np.abs(log2(X17))))))))))))), min(X24, div(np.cos(X26), -718.562))))))), np.abs(np.abs(CubeRoot(X27)))))), sqrt(div(min(np.add(15.175, X4), np.tan(np.subtract(max(CubeRoot(max(X5, X23)), max(max(np.cos(X29), CubeRoot(X8)), log10(X11))), CubeRoot(div(X4, X29))))), div(X4, X29)))), np.add(X24, -409.279)))',
'np.tan(np.tan(CubeRoot(CubeRoot(CubeRoot(CubeRoot(CubeRoot(CubeRoot(np.subtract(X29, X26)))))))))',
####################################################################################################
# Standard Scaler + SMOTE Class 3 
####################################################################################################
'np.add(np.add(np.add(np.add(np.add(CubeRoot(CubeRoot(np.add(X29, X28))), CubeRoot(CubeRoot(np.cos(X28)))), CubeRoot(CubeRoot(np.add(X29, X28)))), np.add(CubeRoot(CubeRoot(np.add(X29, np.add(np.add(np.add(CubeRoot(CubeRoot(np.add(X29, X28))), CubeRoot(CubeRoot(max(X29, np.add(np.add(CubeRoot(X29), np.add(np.add(CubeRoot(max(X29, np.add(CubeRoot(np.multiply(X1, X6)), CubeRoot(CubeRoot(np.add(X29, X28)))))), CubeRoot(CubeRoot(np.add(X29, X28)))), CubeRoot(CubeRoot(max(X29, np.add(CubeRoot(X29), CubeRoot(CubeRoot(np.add(X29, X28))))))))), CubeRoot(CubeRoot(np.add(X29, X28)))))))), np.add(CubeRoot(CubeRoot(np.add(X29, X28))), np.cos(np.multiply(np.multiply(X12, log10(X27)), X8)))), CubeRoot(X29))))), CubeRoot(X29))), CubeRoot(CubeRoot(max(X29, np.add(CubeRoot(X29), CubeRoot(CubeRoot(np.add(X29, X28)))))))), np.add(np.add(np.add(CubeRoot(CubeRoot(np.add(X29, X28))), CubeRoot(CubeRoot(max(X29, np.add(np.add(CubeRoot(X29), CubeRoot(CubeRoot(max(X29, np.add(CubeRoot(X29), CubeRoot(CubeRoot(np.sin(sqrt(X0))))))))), CubeRoot(CubeRoot(np.add(X29, X28)))))))), np.add(CubeRoot(CubeRoot(np.add(X29, np.add(np.add(np.add(CubeRoot(CubeRoot(np.add(X29, X28))), CubeRoot(CubeRoot(max(X29, np.add(np.add(CubeRoot(X29), np.add(np.add(CubeRoot(max(X29, np.add(CubeRoot(np.multiply(X1, X6)), CubeRoot(CubeRoot(np.add(X29, X28)))))), CubeRoot(CubeRoot(np.add(X29, X28)))), CubeRoot(CubeRoot(max(X29, np.add(CubeRoot(X29), CubeRoot(CubeRoot(np.add(X29, X28))))))))), CubeRoot(CubeRoot(np.add(np.sin(np.add(min(CubeRoot(X14), div(X11, -635.367)), sqrt(np.subtract(X13, X10)))), X28)))))))), np.add(CubeRoot(CubeRoot(np.add(X29, X28))), CubeRoot(X29))), CubeRoot(X29))))), CubeRoot(X29))), CubeRoot(X29)))',
'np.subtract(X29, np.subtract(np.subtract(np.subtract(np.subtract(log10(X29), np.subtract(X29, np.subtract(np.subtract(X13, np.subtract(X29, np.subtract(div(CubeRoot(np.add(CubeRoot(X0), np.subtract(X29, np.subtract(X27, X29)))), log2(sqrt(X3))), X29))), X29))), X29), np.subtract(X29, np.subtract(np.subtract(X13, np.subtract(X29, np.subtract(div(CubeRoot(np.add(CubeRoot(np.add(CubeRoot(X0), np.subtract(X29, np.subtract(log10(X29), np.subtract(X29, np.subtract(np.subtract(np.sin(min(X4, X12)), np.subtract(X29, np.subtract(X27, X29))), X29)))))), np.subtract(X29, np.subtract(X27, X29)))), log2(sqrt(min(np.multiply(X16, X16), np.cos(np.add(np.cos(log(np.cos(max(CubeRoot(min(log(div(X22, X14)), CubeRoot(np.subtract(X27, X13)))), sqrt(CubeRoot(log2(np.sin(X24)))))))), min(CubeRoot(X4), log2(X4)))))))), X29))), X29))), X29))',
'np.add(np.add(np.add(X28, np.add(X28, np.add(np.add(np.sin(X29), np.add(np.add(X28, np.subtract(np.cos(X22), np.subtract(X14, X5))), X29)), X29))), np.add(np.add(np.add(X28, np.add(X28, np.add(np.add(np.sin(X29), np.add(X28, np.add(X28, np.add(np.add(np.sin(X29), np.add(np.add(X28, np.subtract(X29, np.tan(X12))), X29)), X29)))), X29))), np.add(np.add(X28, np.add(np.add(np.add(X28, np.add(X28, np.add(np.add(np.sin(X29), np.add(np.add(X28, np.subtract(X29, np.subtract(X14, X5))), X29)), X29))), np.add(np.add(np.add(X28, np.add(X28, np.add(np.add(np.sin(X29), np.add(np.add(X28, np.add(X28, np.add(np.add(np.sin(X29), np.add(np.add(X28, np.add(X28, np.add(X28, np.add(np.add(np.sin(X29), np.add(np.add(X28, np.add(X28, np.add(np.add(np.sin(X29), np.add(np.add(X28, np.add(X28, np.add(np.add(np.sin(X29), np.add(np.add(X28, np.subtract(X29, np.subtract(X14, X5))), X29)), X29))), np.add(np.add(np.add(X28, np.add(X28, np.add(np.add(np.sin(X29), np.add(X28, np.add(X28, np.add(np.add(np.sin(X29), np.add(np.add(X28, np.subtract(X29, np.subtract(X14, np.sin(np.sin(X8))))), X29)), X29)))), X29))), np.add(np.add(X28, np.add(np.add(X28, np.add(X28, np.add(np.add(np.sin(X29), np.add(np.cos(np.sin(min(np.sin(max(CubeRoot(X0), log2(X7))), log(np.subtract(CubeRoot(X17), np.multiply(X15, X7)))))), X29)), X29))), np.add(np.add(np.add(X28, np.add(X28, np.add(np.add(np.sin(X29), np.add(X28, np.add(X28, np.add(np.add(np.sin(X29), np.add(np.add(X28, np.subtract(X29, np.subtract(X14, X5))), X29)), X29)))), X29))), np.add(np.add(X28, np.add(X28, X29)), X29)), X29))), X29)), X29))), X29))), np.add(np.add(np.sin(X29), np.add(X28, np.add(X28, np.add(np.add(np.sin(X29), np.add(np.add(np.cos(np.multiply(min(X20, X5), sqrt(X27))), np.subtract(X29, np.subtract(X14, X5))), X29)), X29)))), X29))), X29)))), np.add(np.add(np.add(X28, np.add(X28, np.add(np.add(np.sin(X29), np.add(X28, np.add(X28, np.add(np.add(np.sin(X29), np.add(np.add(X28, np.subtract(X29, np.subtract(X14, X5))), X29)), X29)))), X29))), np.add(np.add(X28, np.add(np.add(X28, np.add(X28, np.add(np.add(np.sin(X29), np.add(np.cos(np.sin(min(np.sin(max(CubeRoot(X0), log2(X7))), log(np.subtract(CubeRoot(X17), np.multiply(X15, X7)))))), X29)), X29))), np.add(np.add(np.add(X28, np.add(X28, np.add(np.add(np.sin(X29), np.add(X28, np.add(X28, np.add(np.add(np.sin(X29), np.add(np.add(X28, np.subtract(X29, np.subtract(X14, X5))), X29)), X29)))), X29))), np.add(np.add(X28, np.add(X28, X29)), X29)), X29))), X29)), X29))), X29))), np.add(np.add(np.sin(X29), np.add(X28, np.add(X28, np.add(np.add(np.sin(X29), np.add(np.add(np.cos(np.multiply(min(X20, X5), sqrt(X27))), np.subtract(X29, np.subtract(X14, X5))), X29)), X29)))), X29))), X29))), np.add(np.add(X28, np.add(X28, X29)), X29)), X29)), X29)), X29)), X29)), X29)',
'div(X29, np.abs(min(np.abs(log2(np.cos(log2(np.add(log10(log(log2(X15))), np.multiply(np.cos(np.abs(X29)), X15)))))), X7)))',
'np.tan(np.tan(CubeRoot(CubeRoot(CubeRoot(CubeRoot(CubeRoot(CubeRoot(CubeRoot(np.subtract(X29, X26))))))))))',
####################################################################################################
# Stnadard Scaler + SVMSMOTE Class 3
####################################################################################################
'np.tan(CubeRoot(np.add(sqrt(np.abs(min(log2(log2(np.add(np.cos(log2(X29)), X29))), min(min(np.abs(CubeRoot(np.sin(sqrt(np.tan(np.sin(np.cos(np.multiply(np.multiply(log2(max(np.abs(np.sin(X15)), X10)), CubeRoot(X11)), np.add(np.multiply(div(X13, X10), np.abs(X15)), min(np.tan(X21), np.sin(np.add(log(np.multiply(X9, X0)), np.subtract(X19, X5))))))))))))), min(log2(log(np.add(np.cos(np.tan(np.tan(log2(np.add(np.cos(CubeRoot(X29)), X29))))), np.subtract(X9, X16)))), log10(log2(X29)))), np.subtract(X12, X12))))), CubeRoot(X29))))',
'np.tan(np.tan(np.cos(log2(log10(min(np.multiply(div(np.multiply(X21, X15), np.multiply(log2(log(X16)), min(144.987, X6))), np.subtract(X4, X4)), CubeRoot(X29)))))))',
'np.add(np.add(np.add(np.add(np.add(np.add(sqrt(X14), np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.subtract(np.tan(np.abs(np.subtract(np.sin(np.tan(log2(max(X14, X6)))), X27))), np.sin(div(np.subtract(log2(X4), X27), np.subtract(X29, X17)))), np.add(np.add(np.add(np.add(np.sin(np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.subtract(np.tan(np.abs(np.cos(min(np.abs(sqrt(np.add(X27, log(X5)))), np.subtract(log10(div(np.tan(np.add(X13, log10(np.add(X0, X15)))), np.abs(np.cos(np.abs(X0))))), np.multiply(np.sin(X26), np.sin(X28))))))), log10(np.multiply(sqrt(CubeRoot(div(max(X17, X8), div(X11, X8)))), log(min(sqrt(np.add(log10(X13), X7)), np.abs(min(X0, X22))))))), np.add(np.add(np.add(np.add(np.sin(CubeRoot(sqrt(X29))), np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.add(max(X3, X29), X29), X29), log2(log(X29))), X29), log2(log(X29))), X29), X29), log2(log(X29))), X29), X29)), X29), X29), X29)), X29), X29), X29), log2(log(X29))), X29), X29)), np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.subtract(np.tan(np.abs(np.subtract(np.sin(np.abs(log2(log2(X23)))), X27))), np.sin(div(np.add(X26, X27), np.subtract(X29, X17)))), np.add(np.add(np.add(np.add(np.sin(np.abs(np.tan(sqrt(log2(np.tan(log2(log(X29)))))))), np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.add(log2(log(X29)), X29), X29), X29), np.add(np.add(np.add(np.add(np.add(np.subtract(np.tan(np.abs(np.subtract(np.sin(np.tan(log2(max(X14, X6)))), X27))), np.sin(div(np.subtract(log2(X4), X27), np.subtract(X29, X17)))), np.add(np.add(np.add(np.add(np.sin(np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.subtract(np.tan(np.abs(np.cos(min(np.abs(sqrt(sqrt(log10(X2)))), np.subtract(log10(div(np.tan(np.add(X13, log10(np.add(X0, X15)))), np.abs(np.cos(np.abs(X0))))), np.multiply(log10(X21), np.sin(X28))))))), log10(np.multiply(sqrt(CubeRoot(div(max(X17, X8), div(X11, X8)))), log(min(sqrt(np.add(log10(X13), X7)), np.abs(min(X0, X22))))))), np.add(np.add(np.add(np.add(np.sin(CubeRoot(sqrt(X29))), np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.add(max(X3, X29), X29), X29), log2(log(X29))), X29), log2(log(X29))), X29), X29), log2(log(X29))), X29), X29)), X29), X29), X29)), X29), X29), X29), log2(log(X29))), X29), X29)), np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.subtract(np.tan(np.abs(np.subtract(np.sin(np.abs(log2(log2(X23)))), X27))), np.sin(div(np.add(X26, X27), np.subtract(X29, X17)))), np.add(np.add(np.add(np.add(np.sin(np.abs(np.tan(sqrt(log2(np.tan(log2(log(X29)))))))), np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.add(log2(log(X29)), X29), X29), X29), log2(log(X29))), X29), np.cos(np.subtract(log2(np.multiply(np.abs(log(X28)), np.cos(log10(X5)))), div(CubeRoot(np.multiply(np.sin(CubeRoot(X4)), div(np.multiply(X1, X29), np.subtract(X5, X18)))), log10(np.tan(np.cos(np.sin(X20)))))))), X29), X29), np.abs(log(X29))), X29), X29)), X29), X29), X29)), max(log10(X27), np.multiply(X16, X12))), X29), X29), log2(min(sqrt(np.sin(X0)), log(X0)))), X29), log(np.subtract(np.abs(np.abs(X26)), np.sin(log2(log2(X5))))))), X29), X29), X29)), X29), X29), X29), X29)), X29), np.cos(np.subtract(log2(np.multiply(np.abs(log(X28)), np.cos(log10(X5)))), div(CubeRoot(np.multiply(np.sin(CubeRoot(X4)), div(np.multiply(X1, X29), np.subtract(X5, X18)))), log10(np.tan(np.cos(np.sin(X20)))))))), X29), X29), np.abs(log(X29))), X29), X29)), X29), X29), X29)), max(log10(X27), np.multiply(X16, X12))), X29), X29), log2(min(sqrt(np.sin(X0)), log(X0)))), X29), log(np.subtract(np.abs(np.abs(X26)), np.sin(log2(log2(X5))))))), X29), X29), X29)), X29), X29), X29), X29), log(np.cos(log10(np.multiply(X16, X8))))), X29), X29), X29), log10(log10(X25))), log10(log(min(sqrt(log2(X20)), np.cos(X28)))))), X29), X29), X29), X29), CubeRoot(np.abs(log2(max(np.add(np.add(np.tan(min(max(sqrt(X6), sqrt(X26)), np.abs(np.abs(X4)))), np.tan(sqrt(log2(X28)))), X29), np.sin(np.abs(sqrt(np.tan(np.abs(np.tan(np.abs(X24))))))))))))',
'log2(np.multiply(div(sqrt(np.subtract(np.abs(np.tan(max(max(CubeRoot(X0), np.multiply(min(np.tan(log10(log10(div(sqrt(X24), max(X0, X4))))), X10), X9)), div(np.multiply(max(np.sin(log10(log10(CubeRoot(X27)))), CubeRoot(sqrt(np.cos(log10(X11))))), X28), np.tan(X24))))), min(min(np.multiply(X7, X4), sqrt(X12)), np.cos(np.add(X10, np.add(X28, 39.759)))))), max(np.abs(X4), np.multiply(log2(np.multiply(X28, X4)), np.add(np.sin(X28), X29)))), np.multiply(np.subtract(CubeRoot(CubeRoot(CubeRoot(np.multiply(np.subtract(log2(X4), np.multiply(sqrt(CubeRoot(X26)), np.sin(sqrt(np.sin(log10(X12)))))), np.add(np.sin(X28), X29))))), max(np.cos(X28), np.multiply(sqrt(log10(sqrt(log10(X20)))), sqrt(log10(log10(div(X6, X15))))))), sqrt(min(log2(X28), div(np.multiply(np.multiply(np.multiply(np.subtract(np.sin(min(np.cos(np.tan(X24)), np.sin(np.abs(X22)))), np.multiply(div(sqrt(X18), np.sin(X10)), min(np.abs(-399.810), np.sin(np.multiply(log(X29), np.sin(X26)))))), np.add(np.subtract(sqrt(CubeRoot(np.multiply(X27, X1))), np.tan(np.tan(np.multiply(max(sqrt(X14), np.abs(X19)), np.tan(np.subtract(X16, X17)))))), log(np.cos(np.subtract(X25, X12))))), np.subtract(log10(sqrt(sqrt(log2(X9)))), log10(log2(np.sin(log2(log2(X8))))))), np.tan(np.multiply(X24, min(X8, log(X16))))), div(X0, div(sqrt(np.abs(np.add(np.multiply(div(np.multiply(sqrt(log(np.multiply(sqrt(sqrt(div(X5, X7))), np.cos(np.sin(log10(X2)))))), div(log(min(CubeRoot(np.sin(log(X7))), log10(CubeRoot(div(X17, X24))))), np.cos(np.cos(sqrt(np.add(log2(X22), max(X0, X12))))))), np.sin(div(X4, X11))), np.multiply(CubeRoot(sqrt(X5)), div(div(X27, X10), np.sin(X24)))), np.cos(np.tan(CubeRoot(np.sin(np.add(max(log10(X12), div(X29, log(X26))), np.abs(log(X27)))))))))), sqrt(np.subtract(log2(CubeRoot(max(max(np.add(div(np.add(X3, X28), max(log(log10(np.multiply(X23, X8))), np.cos(max(np.tan(-215.477), log(np.abs(div(min(X22, X27), np.tan(X13)))))))), log2(CubeRoot(X4))), sqrt(np.multiply(max(np.cos(np.cos(np.multiply(log2(X16), -462.760))), np.add(log(X14), np.add(X17, X27))), max(np.tan(sqrt(log10(max(np.subtract(log(np.sin(np.multiply(X3, X20))), X28), np.cos(X5))))), log2(X20))))), div(X17, X4)))), np.abs(X8)))))))))))',
'np.add(np.add(CubeRoot(np.sin(np.multiply(np.subtract(np.sin(np.cos(log(X29))), min(sqrt(min(X6, 305.050)), np.tan(log2(X17)))), np.multiply(log(np.add(div(X27, X28), log(X2))), CubeRoot(div(sqrt(X28), np.subtract(X24, X12))))))), np.add(np.add(X29, np.add(X29, np.tan(CubeRoot(CubeRoot(max(log10(X4), np.add(X29, np.sin(CubeRoot(CubeRoot(CubeRoot(CubeRoot(np.add(X28, CubeRoot(CubeRoot(np.add(X29, np.add(np.cos(np.subtract(np.tan(X3), log(X26))), CubeRoot(CubeRoot(CubeRoot(np.add(X29, CubeRoot(np.abs(div(log2(np.add(np.tan(X26), X28)), np.add(div(log2(min(log10(np.subtract(np.sin(min(max(np.sin(X24), log2(X4)), log10(min(X5, X14)))), np.multiply(np.sin(div(sqrt(X16), np.subtract(X0, X7))), max(np.add(max(X19, X16), log(X22)), log2(np.subtract(X17, X2)))))), X7)), log2(div(log(div(X25, X3)), X21))), 235.781))))))))))))))))))))))))), np.add(np.tan(CubeRoot(CubeRoot(np.add(X29, CubeRoot(np.abs(div(log2(log2(log10(np.subtract(np.abs(X8), np.cos(X2))))), np.add(div(log2(min(X3, X7)), log2(div(np.add(X7, X20), X21))), 235.781)))))))), np.add(X29, np.add(X29, np.tan(CubeRoot(CubeRoot(max(X29, np.add(X29, np.add(np.cos(log(CubeRoot(np.tan(np.multiply(np.sin(max(log(X17), max(X4, X9))), CubeRoot(np.cos(np.abs(X6)))))))), CubeRoot(np.add(X29, np.tan(CubeRoot(CubeRoot(max(X29, np.add(X29, np.add(np.cos(X2), CubeRoot(CubeRoot(CubeRoot(np.add(X29, CubeRoot(np.abs(div(log2(np.add(np.tan(X26), X28)), np.add(div(log2(min(div(X12, min(X17, X9)), X7)), log2(div(log(np.tan(np.subtract(X13, div(sqrt(X4), CubeRoot(X13))))), X21))), 235.781))))))))))))))))))))))))))), CubeRoot(CubeRoot(CubeRoot(np.add(X28, CubeRoot(CubeRoot(np.add(X29, np.add(np.cos(np.subtract(np.tan(X3), log(X26))), CubeRoot(CubeRoot(CubeRoot(np.add(X29, CubeRoot(np.abs(div(log2(np.add(np.tan(X26), X28)), np.add(div(log2(min(np.cos(X18), X7)), log2(div(log(div(X25, X3)), X21))), 235.781)))))))))))))))))'

 ]
                                                                                                                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                                                                          
RobustScaler_raw = [####################################################################################################
# Robust Scaler + ADASYN CLASS 3 
####################################################################################################
'log2(np.subtract(np.subtract(np.subtract(np.subtract(np.subtract(np.subtract(np.subtract(X8, X16), log2(np.multiply(div(CubeRoot(max(min(np.cos(max(np.multiply(log(np.sin(X1)), max(X28, X14)), np.cos(np.subtract(X24, X1)))), max(np.add(log(np.cos(np.cos(np.tan(X9)))), CubeRoot(np.abs(sqrt(log(X23))))), div(X2, log(log2(np.cos(max(X3, np.tan(log10(np.add(X19, max(X21, X20))))))))))), CubeRoot(log10(np.abs(np.sin(np.sin(X15))))))), -209.492), np.sin(div(CubeRoot(X29), np.add(np.add(343.596, X20), X25)))))), log2(np.multiply(div(CubeRoot(max(X9, np.cos(sqrt(np.add(np.subtract(X12, -185.340), X1))))), np.subtract(log(71.654), -209.492)), np.sin(div(CubeRoot(X29), np.add(np.add(343.596, np.subtract(X18, X7)), X25)))))), log2(np.multiply(div(CubeRoot(np.multiply(np.sin(log10(np.sin(sqrt(np.sin(X29))))), np.sin(log10(np.sin(np.cos(np.subtract(X24, X10))))))), np.add(343.596, X19)), np.abs(np.sin(361.306))))), log2(np.multiply(div(CubeRoot(max(min(np.cos(max(np.multiply(log(np.multiply(sqrt(max(min(X27, X13), log10(X10))), np.add(min(np.subtract(np.sin(X1), max(X10, X20)), np.cos(sqrt(log(min(sqrt(np.subtract(X26, X7)), np.subtract(X20, X29)))))), np.add(np.cos(div(np.cos(div(np.cos(X15), np.tan(np.tan(np.add(np.sin(X11), max(X11, X13)))))), log10(CubeRoot(np.subtract(X2, X16))))), np.sin(log10(sqrt(CubeRoot(min(X15, X15))))))))), max(X28, X14)), np.cos(np.subtract(X24, X1)))), max(np.add(log(np.cos(np.cos(np.tan(X9)))), CubeRoot(np.abs(sqrt(log(X23))))), div(X2, log(log2(np.cos(max(X3, np.tan(log10(np.add(X19, log2(log10(max(np.abs(X27), max(div(X1, np.add(X24, log(X15))), np.subtract(min(CubeRoot(X26), CubeRoot(X11)), sqrt(np.cos(X22))))))))))))))))), CubeRoot(log10(np.cos(X27))))), -209.492), np.sin(div(CubeRoot(X29), np.add(np.add(343.596, X20), X25)))))), log2(np.multiply(div(CubeRoot(max(X9, np.cos(sqrt(np.add(np.subtract(X12, -185.340), X1))))), np.subtract(log(71.654), -209.492)), np.sin(div(CubeRoot(X29), np.add(np.add(343.596, np.subtract(X18, X7)), X25)))))), log2(np.multiply(div(CubeRoot(np.multiply(np.sin(log10(np.sin(sqrt(np.sin(X29))))), np.sin(log10(np.sin(np.cos(np.subtract(X24, X10))))))), np.add(343.596, X19)), np.abs(np.sin(361.306))))))',
'np.multiply(max(X11, 213.195), log10(np.add(np.cos(np.cos(np.sin(CubeRoot(X13)))), CubeRoot(X29))))',
'np.add(min(np.subtract(np.abs(min(np.tan(np.sin(np.tan(X29))), div(max(CubeRoot(X12), log(X4)), div(np.cos(X6), CubeRoot(X24))))), np.add(sqrt(max(CubeRoot(np.sin(X27)), CubeRoot(max(X23, X26)))), np.abs(np.add(log(np.sin(X20)), np.tan(np.sin(X19)))))), log2(log2(sqrt(sqrt(np.tan(X12)))))), np.add(X29, np.add(np.add(log10(log(sqrt(log(np.sin(np.abs(X23)))))), np.add(X29, np.add(X29, np.add(X29, np.add(np.sin(log(X23)), np.add(X29, np.add(np.sin(log(X16)), np.add(X29, np.add(X29, np.add(X29, log2(CubeRoot(np.tan(np.abs(X29)))))))))))))), np.add(X29, np.add(X29, np.add(X29, np.add(X29, np.add(X29, np.add(X29, np.add(X29, log10(CubeRoot(np.tan(np.abs(X29))))))))))))))',
'log10(np.multiply(div(CubeRoot(X29), np.tan(X17)), max(np.abs(np.tan(X17)), np.abs(max(sqrt(X29), np.multiply(np.multiply(div(CubeRoot(X29), np.multiply(log10(np.tan(np.cos(X27))), np.sin(np.cos(np.cos(log(log10(min(CubeRoot(sqrt(np.cos(min(min(np.abs(log10(min(np.add(np.subtract(X24, X9), log(div(CubeRoot(CubeRoot(np.abs(np.abs(np.multiply(sqrt(X25), CubeRoot(np.add(np.subtract(min(X12, 233.251), CubeRoot(X22)), np.subtract(log(X12), np.subtract(X7, X21))))))))), np.subtract(log(X12), max(X23, X10))))), log(div(CubeRoot(CubeRoot(np.abs(np.abs(np.multiply(np.add(X27, X22), np.subtract(np.cos(div(X13, X10)), sqrt(np.sin(np.tan(X18))))))))), np.subtract(np.add(np.tan(X28), log(352.723)), max(CubeRoot(np.add(np.sin(log(log2(X21))), np.sin(np.cos(X15)))), np.abs(log2(max(np.abs(np.tan(X17)), np.abs(X29))))))))))), np.subtract(sqrt(X17), sqrt(X12))), np.cos(min(X4, X23)))))), np.abs(X18))))))))), np.add(np.multiply(div(CubeRoot(X29), np.multiply(log10(np.tan(np.cos(X27))), np.sin(np.cos(np.cos(log(log10(sqrt(sqrt(np.cos(np.cos(min(min(max(np.sin(np.cos(sqrt(CubeRoot(X21)))), np.abs(np.tan(np.subtract(X6, X11)))), np.subtract(np.abs(X5), np.tan(X4))), np.abs(min(np.sin(np.subtract(np.cos(np.subtract(X21, X14)), CubeRoot(max(X12, X21)))), log2(log10(CubeRoot(X3))))))))))))))))), 240.340), X18)), np.subtract(max(np.add(X29, X29), np.multiply(X29, 240.340)), np.sin(np.sin(X25)))))))))',
'np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.add(log2(CubeRoot(X29)), X29), X29), X29), X29), np.subtract(X6, max(X26, X27))), X29), X29), min(log(log10(X28)), log2(div(np.tan(X17), min(37.439, X28))))), X29), X29), X29), X29)',






####################################################################################################
# Robust Scaler + BorderlineSMOTE Class 3 
####################################################################################################
'log2(sqrt(div(np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.multiply(np.add(div(np.add(np.abs(X11), -400.522), div(min(np.subtract(np.sin(X3), np.subtract(max(log(min(np.tan(log2(log(X9))), log(np.add(log10(X0), np.cos(X28))))), np.abs(log10(min(div(np.multiply(X23, X21), max(X21, X26)), np.sin(np.subtract(X17, X13)))))), np.abs(X5))), CubeRoot(np.sin(236.762))), -157.675)), X29), X29), X29), X29), X29), X29), X29), X29), X29), div(np.sin(log10(np.sin(log2(207.848)))), -157.675)), X29), X29), X29), X29), X29), log(-403.435))))',
'np.add(X29, np.add(X29, np.add(np.multiply(CubeRoot(np.cos(np.sin(X23))), np.add(X29, np.add(X13, np.add(log10(np.subtract(np.abs(X24), max(X15, X21))), np.subtract(np.subtract(log10(CubeRoot(X29)), log2(log10(CubeRoot(X29)))), X26))))), np.add(log10(np.subtract(CubeRoot(X21), max(X15, X21))), np.subtract(np.subtract(log10(CubeRoot(X29)), log2(log10(CubeRoot(X29)))), X26)))))',
'np.add(X29, np.subtract(log2(np.abs(max(sqrt(np.multiply(sqrt(div(X29, min(max(np.subtract(max(X18, X21), log10(X9)), sqrt(log(X9))), X28))), div(np.multiply(np.subtract(log2(log(np.add(X17, X27))), np.subtract(np.add(min(X21, X13), np.abs(415.791)), min(min(X21, X8), np.cos(X4)))), np.subtract(np.abs(log10(np.subtract(np.sin(X20), max(X26, X2)))), 404.393)), log(X29)))), log2(np.multiply(np.subtract(-572.543, log2(X26)), log(np.multiply(X21, np.multiply(X29, X18)))))))), X27))',
'np.subtract(min(log2(CubeRoot(X29)), log2(X5)), CubeRoot(div(min(max(-538.893, X28), np.multiply(np.abs(76.325), -592.840)), X29)))',
'np.subtract(CubeRoot(min(log2(np.add(max(np.multiply(np.abs(np.subtract(-428.681, X14)), min(X29, np.add(385.934, X25))), np.sin(X10)), np.cos(np.tan(X16)))), log2(np.add(np.multiply(np.abs(np.add(-428.681, X14)), min(X29, np.add(385.934, X25))), np.abs(min(div(X12, -626.484), np.abs(log2(X13)))))))), np.subtract(np.add(np.cos(np.add(X29, X29)), sqrt(np.add(log2(div(np.multiply(CubeRoot(min(-402.499, X21)), np.abs(X17)), min(np.cos(X14), np.sin(X16)))), log10(X12)))), log(np.subtract(np.add(np.cos(max(np.cos(div(X22, 183.503)), log(sqrt(X7)))), np.subtract(CubeRoot(X29), np.add(np.cos(log(div(X5, X29))), np.multiply(X29, np.multiply(div(-179.534, X20), CubeRoot(np.abs(np.multiply(np.add(np.tan(sqrt(X2)), log2(div(X13, X23))), np.subtract(CubeRoot(div(X0, X29)), np.add(sqrt(X26), np.abs(-289.752))))))))))), np.cos(CubeRoot(log2(-379.101)))))))',




####################################################################################################
# Robust Scaler + SMOTE Class 3 
####################################################################################################
'np.subtract(X29, div(np.add(-766.901, X24), np.subtract(X29, np.cos(X29))))',
'np.add(CubeRoot(log(X23)), np.add(np.add(X29, np.add(X29, np.add(min(np.subtract(CubeRoot(min(np.tan(X3), np.tan(234.703))), X27), div(X24, np.tan(X15))), np.add(X29, np.add(X29, np.add(X29, np.add(np.add(X29, np.add(X29, np.add(min(np.subtract(CubeRoot(min(np.tan(X3), np.tan(234.703))), X27), div(X24, np.tan(X15))), np.add(X29, np.add(X29, np.add(X29, min(np.add(X29, min(np.add(X29, min(np.tan(-221.020), X14)), div(X24, np.tan(X15)))), X14))))))), np.add(X29, np.add(np.add(X29, min(np.add(X29, min(np.subtract(CubeRoot(log(np.abs(np.subtract(X15, X24)))), X27), div(X24, np.tan(X15)))), X14)), np.add(X29, min(np.add(X29, min(np.subtract(CubeRoot(log(np.abs(np.subtract(X15, X24)))), X27), div(X24, np.tan(X15)))), X14))))))))))), np.add(X29, np.add(X29, np.add(X29, min(np.add(X29, min(np.subtract(CubeRoot(log(np.abs(np.subtract(X15, X24)))), X27), div(X24, np.tan(X15)))), X14))))))',
'np.subtract(X29, np.add(np.add(max(max(max(max(max(max(max(max(log2(np.multiply(np.add(X28, X6), X26)), log2(np.multiply(np.abs(X27), X26))), log2(X27)), log2(np.multiply(min(div(np.abs(-425.119), np.add(X29, np.sin(66.742))), X5), X26))), log2(np.multiply(np.sin(X5), X26))), log2(X27)), log2(np.multiply(sqrt(X27), X26))), log2(X27)), log2(np.multiply(min(div(np.abs(-425.119), np.add(X29, np.subtract(X11, X3))), np.abs(X5)), X26))), max(max(max(max(max(max(max(max(np.add(max(div(-750.334, X29), log2(np.multiply(np.sin(144.338), X26))), log2(sqrt(np.cos(X22)))), log2(X27)), log2(min(np.abs(X27), X26))), log2(X27)), log2(np.multiply(log10(max(log10(log(CubeRoot(np.cos(min(np.subtract(np.multiply(max(CubeRoot(CubeRoot(np.sin(np.abs(X21)))), div(np.multiply(X13, X13), np.sin(X23))), div(X11, X0)), log2(X2)), np.sin(log(np.add(log10(X25), np.multiply(X20, X1))))))))), np.add(np.abs(max(X14, sqrt(X27))), min(log(sqrt(X9)), np.sin(log2(np.tan(X7))))))), X27))), log2(np.multiply(np.tan(X28), X26))), log2(X27)), log2(np.multiply(X5, X26))), log2(X27))), log2(X28)))',
'log2(np.add(div(sqrt(X29), log(CubeRoot(CubeRoot(CubeRoot(CubeRoot(sqrt(np.sin(CubeRoot(np.cos(div(np.sin(X26), X29))))))))))), np.add(sqrt(div(sqrt(X29), log(CubeRoot(CubeRoot(CubeRoot(CubeRoot(sqrt(np.sin(CubeRoot(CubeRoot(CubeRoot(CubeRoot(np.cos(CubeRoot(np.sin(X7)))))))))))))))), log2(np.cos(log10(max(log2(np.cos(X25)), sqrt(np.cos(sqrt(sqrt(CubeRoot(log10(max(734.935, X1))))))))))))))',
'np.add(np.add(div(CubeRoot(np.sin(log2(np.cos(X12)))), div(sqrt(np.sin(np.multiply(CubeRoot(log(np.abs(np.sin(max(div(X26, 530.792), log10(X28)))))), np.abs(np.sin(X11))))), div(CubeRoot(np.multiply(log2(log10(np.multiply(X29, X17))), div(log(X12), div(247.413, X18)))), np.subtract(sqrt(X19), log2(np.abs(np.multiply(X22, X22))))))), np.subtract(np.subtract(log10(np.multiply(min(X13, -512.026), CubeRoot(np.add(log(X29), np.sin(X29))))), np.multiply(np.sin(np.subtract(sqrt(np.tan(X28)), log(CubeRoot(X29)))), sqrt(np.sin(np.subtract(np.multiply(X25, X16), np.tan(X4)))))), max(np.sin(np.sin(np.tan(log10(X1)))), np.subtract(np.subtract(X26, X1), sqrt(log2(X29)))))), np.subtract(np.subtract(log10(np.multiply(min(X13, -512.026), sqrt(X29))), np.multiply(np.multiply(np.subtract(X27, X10), X27), sqrt(log2(X4)))), CubeRoot(np.subtract(np.tan(np.cos(log10(X5))), np.subtract(log2(X29), np.multiply(X29, min(X24, log(log10(CubeRoot(X23))))))))))',



####################################################################################################
# Robust Scaler + SVMSMOTE Class 3
####################################################################################################
'log2(np.add(div(np.add(np.add(div(np.add(np.sin(sqrt(sqrt(np.multiply(div(np.sin(X29), np.sin(X27)), np.multiply(sqrt(X25), log2(np.sin(np.multiply(np.cos(log2(X22)), log10(X1))))))))), log10(CubeRoot(np.cos(np.cos(np.abs(CubeRoot(min(X27, np.cos(X14))))))))), np.abs(-186.907)), np.abs(log2(div(np.tan(div(np.multiply(log(CubeRoot(sqrt(X27))), X26), max(np.subtract(np.abs(127.294), np.add(np.multiply(X8, X21), -179.114)), X29))), max(CubeRoot(sqrt(log2(np.abs(log2(div(np.tan(div(np.multiply(log(CubeRoot(sqrt(X27))), X26), max(np.subtract(np.subtract(X15, X3), np.add(np.multiply(X8, X21), -179.114)), X29))), max(CubeRoot(sqrt(log2(np.tan(X29)))), min(log(div(sqrt(X29), np.subtract(X6, X5))), max(np.tan(X25), div(X16, X3)))))))))), min(log(div(sqrt(X29), np.subtract(X6, X5))), max(np.subtract(X12, X7), div(X16, X3)))))))), div(sqrt(X29), np.abs(X24))), np.subtract(np.tan(log10(np.add(np.subtract(np.abs(np.abs(CubeRoot(-349.108))), log2(div(np.multiply(X28, X28), log2(X29)))), min(-20.111, X27)))), min(max(log2(log2(np.multiply(np.multiply(X1, X22), log10(X26)))), log(div(X29, log10(np.add(X10, X4))))), CubeRoot(X29)))), np.abs(np.sin(div(np.multiply(div(X0, np.tan(np.sin(X24))), X29), div(max(X6, np.abs(log10(X14))), log2(np.cos(X3))))))))',
'log2(np.add(X29, log2(max(max(log10(np.cos(np.subtract(np.subtract(X15, X8), np.cos(X29)))), np.add(max(max(X24, np.cos(np.subtract(X16, X9))), np.multiply(np.abs(np.add(np.multiply(sqrt(X29), np.abs(np.multiply(np.subtract(np.multiply(sqrt(X29), log2(-247.356)), np.multiply(np.add(-180.614, X24), X29)), log10(np.add(np.cos(X10), X21))))), np.multiply(-342.617, X29))), np.subtract(np.multiply(sqrt(X29), np.tan(np.add(X8, X9))), np.multiply(np.add(-180.614, X7), X29)))), np.subtract(np.subtract(np.multiply(sqrt(X29), log2(-247.356)), np.multiply(np.subtract(-180.614, -198.387), X29)), np.multiply(-342.617, X29)))), np.add(max(max(X24, np.cos(np.cos(min(X4, X13)))), np.add(np.abs(np.subtract(np.add(np.multiply(sqrt(X29), log10(-247.356)), np.multiply(div(X19, X12), X29)), np.multiply(-342.617, X29))), np.subtract(np.multiply(np.add(np.abs(np.multiply(np.subtract(np.multiply(max(125.388, X14), log2(-247.356)), np.multiply(np.add(-180.614, X24), X29)), np.multiply(-342.617, X29))), np.subtract(np.multiply(sqrt(X29), log2(np.add(X8, X9))), np.multiply(np.add(-180.614, X2), X29))), max(log10(np.cos(np.subtract(np.multiply(X15, X8), np.cos(X29)))), np.multiply(max(np.subtract(X24, np.cos(np.subtract(X16, X9))), np.multiply(np.subtract(np.multiply(np.add(np.abs(np.multiply(np.subtract(np.multiply(sqrt(X29), log2(-247.356)), np.multiply(np.add(-180.614, X24), X29)), np.multiply(-342.617, X29))), log2(div(min(np.subtract(X4, X0), np.tan(X5)), min(log(X10), min(X13, X21))))), max(log10(log10(log10(X25))), np.add(max(max(X24, np.cos(np.subtract(X13, X9))), np.multiply(div(X27, X29), np.subtract(np.multiply(sqrt(X29), np.tan(np.add(X8, X9))), np.multiply(np.add(-180.614, X2), X29)))), np.subtract(np.subtract(np.multiply(np.add(np.sin(np.multiply(np.subtract(np.multiply(sqrt(X29), log2(-247.356)), np.multiply(np.add(-180.614, X24), X29)), np.multiply(X15, X29))), np.subtract(np.multiply(sqrt(X29), log2(np.add(X8, X9))), np.multiply(np.add(-180.614, X2), X29))), max(log10(np.cos(np.subtract(np.multiply(X15, X8), np.cos(X29)))), np.add(max(max(X17, np.cos(max(X16, X9))), np.multiply(CubeRoot(log(X9)), np.subtract(np.abs(log2(np.add(X15, X13))), np.multiply(np.add(-180.614, X16), X29)))), np.subtract(np.subtract(np.multiply(sqrt(X29), log2(X6)), np.multiply(np.add(-180.614, -198.387), X29)), np.multiply(-342.617, X29))))), np.multiply(np.add(-180.614, X2), X29)), np.multiply(-342.617, X29))))), np.multiply(np.add(-180.614, X2), X29)), np.subtract(np.multiply(sqrt(X2), np.tan(np.add(X8, X9))), np.multiply(np.add(-180.614, X2), X29)))), np.subtract(np.subtract(np.add(np.add(np.abs(np.multiply(np.subtract(np.multiply(sqrt(X21), max(log10(np.cos(np.subtract(np.multiply(X15, X8), np.cos(X29)))), np.add(max(max(X24, np.cos(np.subtract(X16, X9))), np.multiply(np.subtract(np.multiply(np.add(np.abs(np.multiply(np.subtract(np.multiply(sqrt(X29), log2(-247.356)), np.multiply(np.add(-180.614, X24), X29)), np.multiply(X28, X29))), np.subtract(np.multiply(sqrt(X29), log2(np.add(X8, X9))), max(X4, X15))), max(div(X26, X1), np.add(max(div(X24, np.cos(np.subtract(X16, X9))), np.multiply(div(X27, X29), np.subtract(np.multiply(sqrt(X29), np.tan(np.add(X8, X9))), np.multiply(np.add(X11, X2), X29)))), np.subtract(np.subtract(np.multiply(np.add(np.abs(np.multiply(np.subtract(np.multiply(sqrt(X29), log2(-247.356)), np.multiply(np.add(-180.614, X24), X29)), div(-342.617, X29))), np.subtract(np.multiply(sqrt(X29), log2(np.add(X8, X9))), np.multiply(np.add(-180.614, X2), X29))), max(log10(np.cos(np.subtract(np.multiply(X15, X8), np.cos(X29)))), np.add(max(max(X24, np.cos(np.subtract(X16, X9))), np.multiply(div(X27, X29), np.subtract(np.abs(log2(np.add(X15, X13))), np.subtract(np.add(-180.614, X2), X29)))), np.subtract(np.subtract(np.multiply(sqrt(X29), sqrt(-247.356)), np.multiply(np.add(-180.614, -198.387), X29)), np.multiply(-342.617, X10))))), np.multiply(np.add(-180.614, X2), X29)), np.multiply(-342.617, X29))))), np.multiply(div(-180.614, X2), X29)), np.subtract(np.multiply(sqrt(X29), np.tan(np.add(X8, X9))), np.multiply(np.add(-180.614, X2), X29)))), np.subtract(np.subtract(np.multiply(np.add(np.abs(np.multiply(np.subtract(np.multiply(sqrt(X29), log2(-247.356)), np.multiply(np.add(-180.614, X24), X29)), np.multiply(X28, X29))), np.subtract(np.multiply(sqrt(X29), log2(np.add(X8, X9))), np.multiply(np.add(-180.614, X2), X29))), max(log10(np.abs(np.subtract(np.multiply(X15, X8), np.cos(X29)))), np.add(max(max(X24, np.cos(np.subtract(X16, X9))), np.multiply(div(X27, X29), np.subtract(np.multiply(np.multiply(X4, X3), np.sin(np.add(X25, X23))), np.multiply(np.add(-180.614, X2), X29)))), np.subtract(np.subtract(np.multiply(sqrt(X29), log2(-247.356)), div(X29, X17)), np.multiply(-342.617, X29))))), np.multiply(log10(np.tan(X15)), X29)), np.multiply(-342.617, X29))))), np.multiply(np.add(-180.614, X24), X29)), max(-342.617, X29))), np.subtract(np.multiply(sqrt(X29), log2(np.add(X8, X9))), np.multiply(np.add(-180.614, X2), X29))), max(log10(np.cos(np.subtract(np.multiply(X15, X8), np.cos(X29)))), np.add(max(max(X24, np.cos(np.subtract(X16, X9))), np.multiply(div(X27, X29), np.subtract(np.multiply(np.multiply(X4, X3), np.sin(np.add(X25, X23))), np.multiply(np.add(-180.614, X7), X29)))), np.subtract(np.subtract(np.multiply(sqrt(X29), np.tan(X6)), np.multiply(np.add(-180.614, -198.387), X29)), np.multiply(-342.617, X27))))), np.multiply(np.add(-180.614, X2), X29)), np.multiply(-342.617, X29))))), np.multiply(min(-180.614, X2), X29)))), np.subtract(np.subtract(np.multiply(sqrt(X29), np.abs(np.multiply(np.subtract(np.add(sqrt(X29), np.tan(-247.356)), np.multiply(np.add(-180.614, np.add(sqrt(np.abs(X2)), log10(log(X21)))), X29)), min(-342.617, X29)))), np.multiply(np.subtract(X16, X28), X29)), np.multiply(-342.617, X29)))))))',
'np.add(np.add(np.abs(X29), np.subtract(np.add(np.add(np.abs(X29), np.subtract(log10(CubeRoot(X29)), CubeRoot(X27))), np.subtract(log10(CubeRoot(X29)), log(log10(X29)))), log(X27))), np.subtract(log10(CubeRoot(X29)), np.subtract(X27, np.tan(X24))))',
'log2(log2(np.abs(np.cos(div(sqrt(sqrt(sqrt(X29))), np.tan(np.cos(sqrt(np.sin(np.cos(np.sin(np.tan(log2(div(sqrt(sqrt(X29)), CubeRoot(np.cos(log10(min(np.add(np.multiply(X8, np.tan(log10(np.add(sqrt(log(min(X1, X9))), log2(log(np.tan(X6))))))), sqrt(log(log2(min(min(X18, X11), sqrt(np.abs(np.add(np.multiply(log2(np.abs(np.cos(div(sqrt(sqrt(sqrt(X29))), np.tan(np.cos(sqrt(np.sin(np.cos(np.sin(np.tan(log2(div(np.cos(np.sin(X3)), CubeRoot(np.cos(log10(min(np.add(np.cos(sqrt(np.abs(np.add(np.tan(np.subtract(np.subtract(np.abs(log2(X26)), log10(np.abs(max(min(max(np.abs(np.add(log(np.multiply(X4, X3)), X23)), np.tan(np.multiply(X20, X12))), np.multiply(log2(min(np.subtract(np.add(log10(X19), np.sin(X28)), np.cos(min(X9, X15))), np.cos(log(np.abs(div(np.sin(log(X4)), div(np.abs(X28), CubeRoot(X27)))))))), min(np.tan(np.cos(max(np.abs(X21), log(X15)))), np.multiply(np.multiply(np.add(X7, np.tan(np.abs(X24))), np.subtract(X29, X10)), np.sin(X7))))), np.tan(log(X6)))))), np.sin(min(np.add(min(-180.883, X26), max(X1, X14)), max(np.abs(X1), np.cos(X4)))))), np.add(np.cos(log2(np.abs(sqrt(np.cos(-118.186))))), log(np.abs(np.abs(X13)))))))), sqrt(log(min(6.714, np.sin(np.tan(log(sqrt(X16)))))))), CubeRoot(min(X22, X29))))))))))))))))))), CubeRoot(np.add(X2, X9))), log2(np.add(sqrt(np.abs(np.abs(np.abs(log2(min(np.add(X23, X24), np.subtract(log(max(min(np.abs(np.cos(div(np.sin(min(np.cos(CubeRoot(div(X3, X12))), np.subtract(np.sin(log10(min(np.sin(X22), np.subtract(X6, X15)))), X28))), max(np.multiply(np.subtract(np.abs(np.add(log2(X26), np.subtract(min(np.abs(X27), X6), div(log2(np.multiply(X4, X10)), log10(np.abs(X25)))))), np.subtract(np.tan(np.multiply(log10(div(log10(np.multiply(log2(CubeRoot(sqrt(log(log10(X1))))), np.sin(np.multiply(np.add(X23, X16), np.cos(X3))))), log10(np.sin(sqrt(np.subtract(X12, X20)))))), np.tan(log2(div(np.subtract(sqrt(np.multiply(X26, max(np.multiply(X5, X20), np.multiply(np.add(log(np.multiply(sqrt(log10(np.cos(X10))), np.abs(log10(log(X4))))), np.tan(np.subtract(div(CubeRoot(max(X14, X28)), np.sin(max(X29, X28))), np.cos(max(np.sin(X15), np.subtract(X21, X2)))))), log(np.add(X22, X14)))))), X14), sqrt(div(X15, X9))))))), min(max(np.tan(X3), X26), np.sin(np.tan(log(sqrt(X16))))))), sqrt(log2(np.subtract(np.cos(X17), log2(X6))))), log(np.multiply(np.tan(X10), log10(max(log10(np.tan(np.subtract(log10(X15), np.abs(min(CubeRoot(35.290), np.subtract(X13, -347.434)))))), np.subtract(np.tan(min(np.sin(log(np.subtract(min(np.subtract(min(np.multiply(np.multiply(np.sin(X9), log2(X28)), np.multiply(X18, X22)), np.multiply(log(X18), CubeRoot(X28))), np.tan(CubeRoot(X3))), np.tan(np.multiply(np.tan(log(X27)), np.subtract(log2(X5), np.cos(X16))))), np.add(CubeRoot(np.multiply(np.add(X20, X20), np.subtract(np.subtract(np.cos(log(min(np.subtract(X25, X2), np.multiply(X1, X5)))), np.multiply(np.add(np.add(X7, max(sqrt(X28), np.abs(X11))), min(log(np.multiply(X28, X2)), log10(min(X22, np.abs(-393.122))))), log(CubeRoot(np.sin(log10(X5)))))), np.multiply(X13, np.subtract(X5, X20))))), div(np.cos(CubeRoot(div(X6, X12))), max(np.add(np.multiply(X17, X12), np.tan(X6)), np.tan(np.cos(X22)))))))), X4)), CubeRoot(sqrt(X5))))))))))), CubeRoot(X26)), log(X24))), log2(CubeRoot(X15))))))))), np.tan(sqrt(np.cos(CubeRoot(X20)))))))))))))), CubeRoot(np.sin(log(div(np.sin(-78.708), X29))))))))))))))))))))))',
'np.add(np.add(log(min(np.cos(np.abs(max(np.tan(CubeRoot(sqrt(np.cos(X19)))), max(log(log2(np.abs(div(log10(np.multiply(X11, X18)), np.add(np.sin(-29.544), np.multiply(X7, X1)))))), np.sin(np.multiply(X3, X22)))))), log(max(np.tan(div(log10(X29), div(X29, X21))), np.subtract(X6, X1))))), log2(np.abs(div(max(CubeRoot(np.abs(X29)), log10(np.cos(X23))), np.cos(CubeRoot(log2(log2(np.multiply(div(-253.838, X9), X5))))))))), np.abs(div(max(CubeRoot(np.abs(X29)), log10(np.cos(X23))), np.cos(CubeRoot(log2(log2(np.multiply(div(-253.838, X9), X5))))))))'


]                                                                                                                                                                                                                                                                                                                          


PowerTransformer_raw = [####################################################################################################
# Power Transformer  + ADASYN CLASS 3 
####################################################################################################
# Nema dataseta pa nema ni rezultata




####################################################################################################
# Power Transformer  + BorderlineSMOTE Class 3 
####################################################################################################
'np.add(X29, np.add(X29, np.add(X29, np.add(X29, np.add(X29, np.add(np.add(X29, np.add(np.add(X29, np.add(np.add(X29, np.add(np.add(X29, np.add(X29, np.add(max(X29, X29), CubeRoot(log(np.sin(np.abs(np.sin(X22)))))))), X29)), X29)), X29)), X29))))))',
'div(np.add(X0, 705.262), X29)',
'np.add(np.add(max(244.553, X11), np.multiply(X29, 396.792)), X29)',
'np.add(np.add(np.add(np.multiply(np.add(np.multiply(log2(np.subtract(log(X18), np.sin(np.multiply(log10(X24), X16)))), np.add(max(X7, log2(np.add(log10(max(np.sin(X19), 184.353)), log10(np.sin(CubeRoot(log10(X15))))))), log10(np.abs(np.abs(min(np.tan(427.225), np.sin(np.sin(X29)))))))), log2(log10(min(log10(np.cos(np.tan(max(X17, X28)))), np.abs(np.sin(X9)))))), sqrt(div(CubeRoot(455.551), np.cos(X9)))), X29), X29), X29)',
'np.add(min(X29, np.tan(max(np.tan(X13), np.add(431.266, min(np.cos(X29), np.abs(X28)))))), np.add(CubeRoot(X28), np.add(np.multiply(X29, 35.180), X29)))',




####################################################################################################
# Power Transformer  + SMOTE Class 3 
####################################################################################################
'CubeRoot(div(X29, sqrt(CubeRoot(X18))))',
'div(max(np.sin(div(log10(CubeRoot(X25)), div(np.sin(X17), div(X22, X12)))), max(np.subtract(max(CubeRoot(X7), div(X4, X17)), np.abs(np.tan(X7))), np.subtract(sqrt(np.tan(X10)), np.add(-446.879, max(X1, X28))))), X29)',
'div(X29, np.abs(div(X29, -359.208)))',
'np.subtract(X29, log2(div(np.tan(np.tan(np.cos(np.add(np.cos(np.sin(X29)), np.sin(X29))))), log(np.cos(np.add(np.cos(np.sin(np.cos(CubeRoot(X7)))), np.sin(X29)))))))',
'np.add(np.add(np.add(np.add(np.add(X17, X18), np.add(np.add(np.add(log2(X29), np.add(np.add(np.add(np.add(log10(log2(X28)), X29), X29), X29), X29)), X29), X29)), X29), X29), X29)',



####################################################################################################
# Power Transformer  + SVMSMOTE Class 3
####################################################################################################
'np.tan(CubeRoot(np.tan(CubeRoot(np.tan(CubeRoot(CubeRoot(np.tan(CubeRoot(np.tan(CubeRoot(CubeRoot(np.tan(CubeRoot(np.tan(CubeRoot(CubeRoot(CubeRoot(X29))))))))))))))))))',
'np.subtract(X29, np.add(min(log(np.cos(log2(div(X4, X1)))), np.multiply(log2(log10(div(X24, X1))), np.cos(log2(CubeRoot(X22))))), np.add(np.subtract(np.multiply(np.sin(max(X29, X0)), X27), X29), np.subtract(np.subtract(np.abs(div(min(X17, min(np.sin(X17), X29)), max(CubeRoot(sqrt(sqrt(X17))), log10(np.cos(log(X22)))))), X29), X29))))',
'np.add(np.multiply(276.293, np.add(X29, X29)), np.sin(X11))',
'np.subtract(X29, div(np.subtract(np.add(X1, -201.746), np.sin(X17)), X29))',
'np.subtract(np.subtract(X29, div(np.sin(CubeRoot(X29)), min(CubeRoot(np.multiply(log2(X10), log10(CubeRoot(X17)))), np.tan(log10(np.sin(np.tan(np.sin(div(log(X14), sqrt(X17)))))))))), div(np.sin(CubeRoot(X29)), min(log10(np.sin(sqrt(min(X23, X29)))), X4)))'

]
                                                                                                                                                                                                                                                                                                                          
Normalizer_raw = [####################################################################################################
# Normalizer  + ADASYN CLASS 3 
####################################################################################################
'log2(np.subtract(log(sqrt(X29)), np.abs(np.subtract(log(sqrt(X29)), np.subtract(log(sqrt(X29)), np.abs(np.subtract(log(sqrt(X29)), np.abs(np.abs(np.subtract(log(sqrt(X29)), np.abs(np.subtract(log(sqrt(X29)), np.abs(np.subtract(log(sqrt(X29)), np.abs(np.subtract(log(sqrt(X29)), np.abs(np.subtract(log(sqrt(X29)), np.abs(np.subtract(log(sqrt(X29)), np.add(X16, X11)))))))))))))))))))))',
'np.subtract(div(sqrt(div(X20, X5)), div(sqrt(X29), div(X5, X27))), log2(529.909))',
'log2(log2(np.multiply(np.multiply(np.cos(-72.206), div(div(div(591.998, np.sin(X25)), CubeRoot(X4)), CubeRoot(np.multiply(X3, X29)))), div(div(div(591.998, np.sin(X14)), X14), CubeRoot(np.multiply(X3, X29))))))',
'np.add(np.add(CubeRoot(div(div(sqrt(X29), max(X13, X7)), X11)), log2(sqrt(np.abs(div(min(log10(sqrt(X29)), np.subtract(np.tan(X8), X13)), CubeRoot(X5)))))), log2(sqrt(np.abs(div(min(log10(sqrt(X29)), np.subtract(min(np.tan(X8), X9), X13)), X24)))))',
'log2(div(np.subtract(log2(div(617.501, sqrt(X29))), np.multiply(X11, CubeRoot(X6))), sqrt(X2)))',






####################################################################################################
# Normalizer  + BorderlineSMOTE Class 3 
####################################################################################################
'div(CubeRoot(log10(np.subtract(CubeRoot(CubeRoot(min(log2(np.subtract(CubeRoot(X7), np.cos(X22))), log2(sqrt(X29))))), min(np.sin(log2(log2(sqrt(sqrt(X21))))), min(np.cos(log(np.tan(np.add(X28, X21)))), min(np.subtract(max(log(X16), np.multiply(X26, X11)), log2(CubeRoot(X19))), log2(div(min(CubeRoot(X29), np.tan(X13)), CubeRoot(X27))))))))), CubeRoot(div(min(np.cos(X23), log10(CubeRoot(np.subtract(np.tan(X8), max(sqrt(CubeRoot(X9)), div(log(X23), np.multiply(X28, X2))))))), log(np.add(CubeRoot(np.tan(np.sin(X16))), log(np.tan(log10(np.add(np.multiply(X3, X2), np.subtract(X19, X10))))))))))',
'np.subtract(np.subtract(log(sqrt(np.add(log2(np.multiply(sqrt(X29), np.abs(X18))), div(X14, -500.985)))), np.cos(X8)), log2(np.multiply(sqrt(X29), X22)))',
'np.subtract(div(np.subtract(np.abs(log(CubeRoot(X29))), np.abs(np.cos(X2))), sqrt(sqrt(np.sin(X7)))), log10(div(sqrt(sqrt(np.subtract(log10(np.add(X4, X14)), min(np.sin(X29), max(X18, X20))))), max(np.cos(np.subtract(max(np.tan(X6), CubeRoot(X9)), log(np.cos(X16)))), CubeRoot(div(log10(np.sin(X27)), np.multiply(np.cos(X16), log(X22))))))))',
'min(np.multiply(log2(np.multiply(sqrt(X14), sqrt(X24))), log2(min(np.cos(X13), CubeRoot(X8)))), np.multiply(min(sqrt(log2(np.multiply(min(np.abs(sqrt(np.multiply(np.sin(max(X3, X16)), np.subtract(np.add(X29, X17), log10(X8))))), np.add(log10(np.cos(log10(np.sin(X6)))), np.sin(np.add(CubeRoot(log10(X17)), sqrt(max(X5, -183.832)))))), div(X9, X24)))), np.abs(log2(np.add(log2(np.abs(np.cos(log(np.cos(np.cos(X8)))))), np.abs(sqrt(max(X28, X29))))))), div(log2(np.multiply(CubeRoot(X8), div(log2(div(log10(np.abs(sqrt(max(X28, X29)))), sqrt(X23))), np.multiply(np.multiply(np.sin(min(CubeRoot(np.abs(min(X11, X29))), max(np.cos(X28), CubeRoot(-370.244)))), sqrt(np.abs(sqrt(CubeRoot(X17))))), log2(max(np.subtract(log10(sqrt(X24)), np.subtract(np.sin(X13), np.add(X5, X22))), CubeRoot(min(np.cos(X8), sqrt(X26))))))))), div(sqrt(np.multiply(X23, X0)), max(sqrt(np.sin(sqrt(X17))), np.add(np.cos(min(X23, 325.863)), CubeRoot(log(X12))))))))',
'div(np.add(np.cos(log(sqrt(X5))), np.add(min(log(log(np.tan(sqrt(np.multiply(X19, X29))))), sqrt(div(log2(log(CubeRoot(X22))), div(np.cos(np.cos(X1)), np.subtract(log10(X11), log2(470.433)))))), log10(min(log2(div(log10(sqrt(X17)), min(log(X12), np.cos(X29)))), np.subtract(X8, X10))))), sqrt(X6))',




####################################################################################################
# Normalizer  + SMOTE Class 3 
####################################################################################################
'np.add(log(log10(CubeRoot(np.cos(log10(CubeRoot(X22)))))), np.add(log(log10(CubeRoot(np.cos(log10(CubeRoot(X22)))))), np.abs(np.tan(log(max(div(np.tan(X2), CubeRoot(X29)), log10(log10(-41.364))))))))',
'np.subtract(np.abs(div(log2(div(X29, CubeRoot(X26))), X19)), log2(div(438.579, CubeRoot(X8))))',
'np.multiply(np.subtract(np.add(log2(X25), log2(div(log10(np.sin(sqrt(X29))), np.multiply(div(X16, X24), np.abs(log10(log10(55.563))))))), sqrt(log10(X15))), log2(div(log10(log(sqrt(384.230))), min(X22, X25))))',
'np.subtract(np.subtract(-76.129, X5), np.multiply(max(X9, 139.076), log(sqrt(np.add(np.multiply(X19, X29), log(X7))))))',
'np.tan(np.tan(np.sin(div(np.add(max(np.add(X21, X11), CubeRoot(np.multiply(np.abs(log2(sqrt(sqrt(X3)))), np.multiply(log2(np.sin(sqrt(X29))), log(CubeRoot(X22)))))), np.abs(log2(CubeRoot(log2(sqrt(2.415)))))), log10(np.sin(log10(np.add(np.tan(np.cos(X14)), np.cos(X10)))))))))',



####################################################################################################
# Normalizer  + SVMSMOTE Class 3
####################################################################################################
'div(np.subtract(np.tan(min(div(log(log10(np.subtract(np.add(CubeRoot(np.tan(X26)), log10(np.sin(sqrt(np.cos(np.tan(div(X17, X14))))))), log10(X2)))), sqrt(log2(div(np.sin(np.add(X0, X20)), CubeRoot(X29))))), log2(np.subtract(np.add(sqrt(CubeRoot(X11)), div(div(X7, X9), div(X21, X23))), log10(X16))))), CubeRoot(X21)), np.add(np.add(max(np.cos(X11), X29), min(log2(log(X15)), X19)), log2(div(np.cos(X13), log2(log(max(X15, X9)))))))',
'np.multiply(max(np.abs(777.092), np.abs(sqrt(CubeRoot(X18)))), log(np.add(log2(sqrt(X29)), np.sin(max(X20, X11)))))',
'log2(np.tan(log2(np.add(div(np.subtract(X8, X11), CubeRoot(X29)), log(np.cos(CubeRoot(div(np.subtract(np.subtract(min(sqrt(X29), X15), np.multiply(div(X21, 221.776), X25)), np.multiply(X25, X11)), np.add(X23, X16)))))))))',
'log2(np.subtract(div(127.600, sqrt(X29)), np.cos(np.tan(530.981))))',
'np.subtract(np.add(log2(X14), min(np.add(X2, np.multiply(np.cos(max(X26, X26)), -42.163)), log(X5))), div(div(log2(X19), CubeRoot(X29)), CubeRoot(X29)))'



    
    ]                                                                                                                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                                                                          
MaxAbsScaler_Raw = [####################################################################################################
# MaxAbsScaler  + ADASYN CLASS 3 
####################################################################################################
# Nema dataseta 





####################################################################################################
# MaxAbsScaler  + BorderlineSMOTE Class 3 
####################################################################################################
'log2(np.subtract(np.sin(np.add(div(X18, 588.031), log(log2(X29)))), min(div(log2(np.subtract(np.sin(np.add(div(X18, 588.031), log(log2(X29)))), min(div(CubeRoot(min(X15, np.tan(X29))), X28), log2(X29)))), X28), np.multiply(log2(X29), div(X9, X3)))))',
'log2(np.add(div(div(log(np.tan(X29)), np.tan(-179.063)), X24), log10(CubeRoot(np.cos(119.219)))))',
'np.tan(CubeRoot(max(np.tan(350.561), log(np.multiply(CubeRoot(div(np.subtract(np.subtract(np.sin(X18), div(log10(np.multiply(log(X10), np.sin(X29))), sqrt(X27))), X19), div(X3, X29))), min(np.subtract(np.multiply(np.add(X18, X8), np.add(X28, max(np.subtract(log10(X4), 194.046), np.add(log2(np.tan(np.abs(X29))), log2(np.cos(log10(X17))))))), log2(X7)), np.subtract(X11, X9)))))))',
'log2(np.multiply(sqrt(np.add(np.subtract(X27, np.multiply(X29, div(np.add(-218.390, X23), X28))), np.abs(np.tan(min(np.abs(log2(X29)), np.tan(np.cos(X26))))))), log2(div(sqrt(X27), np.add(log2(X29), max(X29, np.add(sqrt(X27), div(X17, 571.675))))))))',
'log2(max(np.multiply(np.multiply(X29, div(116.164, X5)), 704.733), div(X18, 705.991)))',




####################################################################################################
# MaxAbsScaler  + SMOTE Class 3 
####################################################################################################
'np.multiply(div(div(126.788, X1), X6), np.subtract(np.tan(X29), max(X12, X27)))',
'np.multiply(np.multiply(np.subtract(X29, np.subtract(X13, X8)), 160.041), 160.041)',
'log2(min(np.add(np.abs(X14), div(np.multiply(np.cos(X12), 497.906), np.subtract(log10(np.multiply(sqrt(np.abs(np.tan(X29))), CubeRoot(X6))), log(sqrt(sqrt(np.tan(X29))))))), np.multiply(np.abs(np.subtract(np.sin(X15), X18)), np.abs(X1))))',
'np.subtract(np.abs(min(div(-357.055, X29), CubeRoot(X3))), np.multiply(X21, 200.516))',
'np.add(CubeRoot(np.add(np.multiply(max(np.cos(np.cos(X10)), np.multiply(np.cos(X28), CubeRoot(346.352))), np.subtract(np.multiply(np.subtract(np.subtract(CubeRoot(np.tan(X19)), sqrt(np.multiply(164.270, X29))), sqrt(np.subtract(min(X0, X23), CubeRoot(X1)))), np.subtract(X10, 360.913)), sqrt(div(div(min(X1, X11), X6), X6)))), np.subtract(log2(np.sin(X2)), log(sqrt(CubeRoot(X15)))))), CubeRoot(np.tan(max(np.cos(log2(div(X25, X24))), np.abs(np.add(np.add(X18, X19), CubeRoot(X29)))))))',



####################################################################################################
# MaxAbsScaler  + SVMSMOTE Class 3
####################################################################################################
'log2(np.subtract(log(np.cos(336.106)), np.multiply(X29, np.multiply(837.395, 449.805))))',
'log(np.subtract(div(np.subtract(div(903.298, X29), np.sin(np.sin(log2(X9)))), X29), np.cos(sqrt(-357.004))))',
'log2(np.subtract(log2(np.sin(max(X15, np.sin(51.706)))), np.tan(log2(np.sin(max(log(log2(np.tan(log(-373.696)))), div(X28, X29)))))))',
'log2(np.add(min(np.cos(np.subtract(12.684, X29)), div(div(-445.086, X29), X29)), np.cos(443.072)))',
'np.subtract(np.abs(log2(np.multiply(div(880.945, X29), np.multiply(div(880.945, X29), np.tan(X29))))), np.add(CubeRoot(348.494), sqrt(X15)))'



    
    
    ]   
                                                                                                                                                                                                                                                                                                                          
MinMaxScaler_raw = [####################################################################################################
# MinMaxScaler  + ADASYN CLASS 3 
####################################################################################################
'log2(log2(np.add(max(np.abs(min(X29, X20)), sqrt(np.subtract(sqrt(X29), div(np.add(704.512, np.add(np.abs(max(-27.943, np.multiply(700.986, 138.629))), X11)), X29)))), np.cos(np.tan(min(X28, np.tan(np.add(np.subtract(log10(X29), sqrt(log(log(649.741)))), log(sqrt(313.499))))))))))',
'np.multiply(div(188.948, div(np.abs(X28), log(X29))), min(np.subtract(np.subtract(sqrt(X29), X9), np.tan(X9)), log2(log10(np.abs(np.add(np.multiply(np.abs(np.multiply(np.abs(max(np.sin(X21), max(X16, X26))), div(np.subtract(np.abs(X0), CubeRoot(X22)), log(max(X29, X16))))), np.multiply(np.cos(np.sin(np.subtract(np.multiply(X5, X25), log10(X1)))), log2(min(np.add(log(499.125), log(X18)), np.add(np.tan(X15), sqrt(X12)))))), log2(div(max(max(np.sin(max(X22, X18)), np.multiply(sqrt(X12), max(X22, X25))), div(div(np.tan(X17), np.subtract(345.055, X13)), np.multiply(np.tan(X19), np.sin(X24)))), max(div(sqrt(np.tan(X1)), div(CubeRoot(X17), min(X11, X12))), log10(np.sin(np.subtract(X19, X24))))))))))))',
'np.multiply(np.multiply(np.subtract(X29, X7), div(723.003, min(X2, X23))), div(723.003, CubeRoot(X7)))',
'log2(np.multiply(np.add(np.multiply(log2(X29), max(X1, 695.406)), log(CubeRoot(CubeRoot(np.cos(CubeRoot(779.588)))))), np.cos(X6)))',
'np.subtract(log(CubeRoot(X4)), np.subtract(np.add(np.add(np.cos(X10), CubeRoot(X25)), div(np.subtract(-250.503, X21), np.abs(X29))), np.subtract(log(X21), max(X20, log10(743.727)))))',






####################################################################################################
# MinMaxScaler  + BorderlineSMOTE Class 3 
####################################################################################################
'log2(np.tan(np.subtract(np.cos(np.subtract(np.sin(X10), X13)), CubeRoot(div(div(sqrt(max(np.sin(X14), log10(log2(X17)))), log2(np.multiply(X29, np.cos(log10(X26))))), np.abs(X6))))))',
'log2(np.add(np.add(np.add(np.add(div(div(X9, -274.084), X2), log2(log10(log2(CubeRoot(div(log(X29), X20)))))), log2(X29)), log2(X29)), np.multiply(log2(X29), -352.989)))',
'np.subtract(log10(CubeRoot(X4)), np.subtract(div(X8, X4), log(np.add(div(np.subtract(-720.672, div(-752.003, sqrt(X27))), X29), div(np.cos(X2), CubeRoot(min(np.subtract(X2, np.tan(CubeRoot(np.sin(13.288)))), X28)))))))',
'np.abs(log2(np.multiply(X29, div(np.cos(X21), np.subtract(log(np.cos(np.abs(X8))), log(-655.412))))))',
'log2(np.subtract(div(log2(X29), log(X0)), np.abs(log(np.sin(log(log2(min(np.cos(np.cos(np.cos(X27))), np.subtract(np.tan(X9), min(X7, np.sin(log(log2(min(np.cos(np.cos(np.cos(X27))), np.subtract(np.tan(X9), min(X7, log(np.sin(X11))))))))))))))))))',




####################################################################################################
# MinMaxScaler  + SMOTE Class 3 
####################################################################################################
'np.add(log(X19), np.add(min(X11, -644.298), max(log(log10(np.cos(X14))), div(np.add(log(min(X22, X22)), div(np.add(log10(X20), max(log10(sqrt(X15)), div(sqrt(X29), CubeRoot(X4)))), CubeRoot(X4))), CubeRoot(X4)))))',
'div(np.tan(np.subtract(max(min(np.tan(np.subtract(X23, div(np.tan(X4), X29))), X23), np.subtract(div(X17, X28), min(X13, X14))), div(X27, X29))), sqrt(X27))',
'np.tan(min(CubeRoot(np.tan(min(CubeRoot(log(max(np.tan(min(CubeRoot(np.tan(min(CubeRoot(log(max(np.abs(X9), sqrt(div(X6, div(min(sqrt(X29), np.tan(log(-594.362))), CubeRoot(-390.891))))))), sqrt(-325.686)))), np.tan(np.cos(min(log10(X1), log2(X6)))))), np.sin(log(-594.362))))), log(197.257)))), np.tan(X28)))',
'log(max(np.multiply(np.multiply(-859.171, -807.018), X29), div(np.cos(-357.823), np.subtract(np.tan(CubeRoot(np.tan(-586.204))), -879.626))))',
'np.subtract(np.add(X6, np.add(np.multiply(max(np.cos(X21), X9), np.subtract(log(log10(X11)), sqrt(CubeRoot(np.tan(X1))))), np.add(log(np.subtract(CubeRoot(div(min(X9, X1), np.sin(X9))), div(np.add(np.add(np.abs(X22), np.add(np.sin(X19), max(X2, np.subtract(np.multiply(div(-334.032, CubeRoot(X4)), np.subtract(log2(X16), log10(-588.527))), min(X17, sqrt(X7)))))), log2(np.tan(X7))), np.abs(X29)))), np.multiply(np.add(np.tan(X0), np.tan(X9)), np.multiply(log2(np.subtract(X27, X8)), X28))))), log10(159.923))',



####################################################################################################
# MinMaxScaler  + SVMSMOTE Class 3
####################################################################################################

'log2(np.add(div(np.cos(X29), -706.884), div(div(div(X29, div(np.sin(-42.865), -696.188)), CubeRoot(X7)), CubeRoot(X7))))',
'np.multiply(sqrt(max(div(67.133, sqrt(div(X27, X0))), np.add(159.919, X8))), log10(max(sqrt(np.subtract(log2(X29), log10(max(CubeRoot(np.add(np.sin(X20), np.abs(div(div(np.abs(div(div(sqrt(763.024), np.sin(X19)), sqrt(min(X21, X29)))), np.sin(X19)), sqrt(min(X21, X29)))))), sqrt(X18))))), CubeRoot(X29))))',
'log2(np.add(np.subtract(X29, div(620.172, sqrt(X29))), sqrt(np.cos(log(min(-576.955, X18))))))',
'np.subtract(CubeRoot(X6), np.subtract(np.subtract(np.subtract(np.subtract(log2(X29), X1), np.add(X6, X14)), np.tan(X0)), np.multiply(np.subtract(log2(X20), 53.116), sqrt(sqrt(div(log10(X7), log10(div(X29, log2(X3)))))))))',
'np.subtract(div(div(min(log2(X29), np.subtract(log10(div(log(X20), max(X21, X9))), np.sin(log2(np.abs(X18))))), sqrt(X29)), log2(X0)), log10(div(-711.662, log10(X11))))'
    
    
    ]                                                                                                                                                                                                                                                                                                                       
                                                                                                                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                                                                          
###############################################################################
from sklearn.preprocessing import (StandardScaler,
                                    RobustScaler,
                                    PowerTransformer,
                                    Normalizer,
                                    MaxAbsScaler,
                                    MinMaxScaler)
                                                                                                                                                                                                                                                                                                                          
SS = StandardScaler()
data_final_2_SS = pd.DataFrame(SS.fit_transform(data_final_2), columns = list(data_final_2.columns))
print(data_final_2_SS)

RS = RobustScaler()
data_final_2_RS = pd.DataFrame(RS.fit_transform(data_final_2), columns = list(data_final_2.columns))
print(data_final_2_RS)

PT = PowerTransformer()
data_final_2_PT = pd.DataFrame(PT.fit_transform(data_final_2), columns = list(data_final_2.columns))
print(data_final_2_PT)

Norm = Normalizer()
data_final_2_Norm = pd.DataFrame(Norm.fit_transform(data_final_2), columns = list(data_final_2.columns))
print(data_final_2_Norm)

MAS = MaxAbsScaler()
data_final_2_MAS = pd.DataFrame(MAS.fit_transform(data_final_2), columns = list(data_final_2.columns))
print(data_final_2_MAS)

MMS = MinMaxScaler()
data_final_2_MMS = pd.DataFrame(MMS.fit_transform(data_final_2), columns = list(data_final_2.columns))
print(data_final_2_MMS)

                                                                                                                                                                                                                                               
def replace_variable(match):
    variable_name = match.group(0)
    # print(variable_name)
    variable_index = int(re.search(r'\d+', variable_name).group())
    return f'data_final_2.loc[i][{variable_index}]'

def replace_variable_SS(match):
    variable_name = match.group(0)
    print("SHIT")
    print(variable_name)
    variable_index = int(re.search(r'\d+', variable_name).group())
    return f'data_final_2_SS.loc[i][{variable_index}]'

def replace_variable_RS(match):
    variable_name = match.group(0)
    variable_index = int(re.search(r'\d+', variable_name).group())
    return f'data_final_2_RS.loc[i][{variable_index}]'

def replace_variable_PT(match):
    variable_name = match.group(0)
    variable_index = int(re.search(r'\d+', variable_name).group())
    return f'data_final_2_PT.loc[i][{variable_index}]'

def replace_variable_Norm(match):
    variable_name = match.group(0)
    variable_index = int(re.search(r'\d+', variable_name).group())
    return f'data_final_2_Norm.loc[i][{variable_index}]'


def replace_variable_MAS(match):
    variable_name = match.group(0)
    variable_index = int(re.search(r'\d+', variable_name).group())
    return f'data_final_2_MAS.loc[i][{variable_index}]'


def replace_variable_MMS(match):
    variable_name = match.group(0)
    variable_index = int(re.search(r'\d+', variable_name).group())
    return f'data_final_2_MMS.loc[i][{variable_index}]'

# Create a regular expression pattern to match variable names
pattern = re.compile(r'X\d+')
# patternSS = re.compile(r'X\d+')
# Perform the variable replacement using re.sub with the callback function
# result = pattern.sub(replace_variable, text)

# print(result)
variable_mapping = {f'X{i}': f'data_final_2.loc[i][{i}]' for i in range(30)}
variable_mapping_SS = {f'X{i}': f'data_final_2_SS.loc[i][{i}]' for i in range(30)}
variable_mapping_RS = {f'X{i}': f'data_final_2.loc[i][{i}]' for i in range(30)}
variable_mapping_PT = {f'X{i}': f'data_final_2.loc[i][{i}]' for i in range(30)}
variable_mapping_Norm = {f'X{i}': f'data_final_2.loc[i][{i}]' for i in range(30)}
variable_mapping_MAS = {f'X{i}': f'data_final_2.loc[i][{i}]' for i in range(30)}
variable_mapping_MMS = {f'X{i}': f'data_final_2.loc[i][{i}]' for i in range(30)}
# print(variable_mapping)
for i in range(len(Original_all)):
    Original_all[i] = pattern.sub(replace_variable, Original_all[i])
    # print(Original_all[i])
for i in range(len(StandardScaler_raw)):
    StandardScaler_raw[i] = pattern.sub(replace_variable_SS, StandardScaler_raw[i])
for i in range(len(RobustScaler_raw)):
    RobustScaler_raw[i] = pattern.sub(replace_variable_RS, RobustScaler_raw[i])
for i in range(len(PowerTransformer_raw)):
    PowerTransformer_raw[i] = pattern.sub(replace_variable_PT, PowerTransformer_raw[i])
for i in range(len(Normalizer_raw)): 
    Normalizer_raw[i] = pattern.sub(replace_variable_Norm, Normalizer_raw[i])
for i in range(len(MaxAbsScaler_Raw)):
    MaxAbsScaler_Raw[i] = pattern.sub(replace_variable_MAS, MaxAbsScaler_Raw[i])
for i in range(len(MinMaxScaler_raw)):
    MinMaxScaler_raw[i] = pattern.sub(replace_variable_MMS, MinMaxScaler_raw[i])
def Sigmoid(x):
    return round(1/(1+np.exp(-x)),0)
AllEquations = Original_all + StandardScaler_raw + RobustScaler_raw + PowerTransformer_raw + \
    Normalizer_raw + MaxAbsScaler_Raw + MinMaxScaler_raw
y_pred = [[] for i in range(len(AllEquations))]
Final_pred = [[] for i in range(len(AllEquations))]
# for j in range(len(Original_all)):
for i in range(len(data_final_2)):
    res = [Sigmoid(eval(AllEquations[j])) for j in range(len(AllEquations))]
    # print(res)
    final_res = sum(res)
    for z in range(len(AllEquations)):
        if final_res >= z: 
            Final_pred[z].append(1.0)
        else:
            Final_pred[z].append(0.0)
    # print(Final_pred)
    print("Fuck you motherfucker")
    for k in range(len(res)):
        y_pred[k].append(res[k])

from sklearn.metrics import (accuracy_score,
                              roc_auc_score,
                              precision_score,
                              recall_score,
                              f1_score) 

Accuracy = [accuracy_score(y_real_3,Final_pred[i]) for i in range(len(Final_pred))]
AUC = [roc_auc_score(y_real_3, Final_pred[i]) for i in range(len(Final_pred))]
Precision = [precision_score(y_real_3,Final_pred[i]) for i in range(len(Final_pred))] 
Recall = [recall_score(y_real_3,Final_pred[i]) for i in range(len(Final_pred))]
F1_score = [f1_score(y_real_3,Final_pred[i]) for i in range(len(Final_pred))]

import matplotlib.pyplot as plt 
plt.figure(figsize=(12,8))
plt.plot([i for i in range(len(Accuracy))], Accuracy, color='red', label = "Accuracy")
plt.plot([i for i in range(len(AUC))], AUC, color='blue', label = "Accuracy")
plt.plot([i for i in range(len(Precision))], Precision, color='green', label="Precision")
plt.plot([i for i in range(len(Recall))], Recall, color='orange', label="Recall")
plt.plot([i for i in range(len(F1_score))], F1_score, color='yellow', label = "F1_Score")
plt.grid(True)
plt.legend()
plt.xticks(range(0,len(Final_pred)))

        
AllResults = pd.concat([pd.DataFrame(np.array(Accuracy), columns = ["Accuracy"]),
pd.DataFrame(np.array(AUC), columns = ["AUC"]),
pd.DataFrame(np.array(Precision), columns = ["Precision"]),
pd.DataFrame(np.array(Recall), columns = ["Recall"]),
pd.DataFrame(np.array(F1_score), columns = ["F1_score"])],axis=1)    

AllResults.to_csv("AllResults_class_3.csv",index=False)

pd.DataFrame(Final_pred).transpose().to_csv("FinalRES_class_3.csv",index=False)




