import  torch
import  torch.nn  as nn
import  math
import  numpy as np
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# batch1 = torch.ones(2,2, 2)
# print(batch1)
# b1 = batch1.reshape(2,4)
#
# print(b1)
# batch2 = torch.tensor([[[1, 2.],[3., 4.]],[[10, 20.],[30., 40.]]])
# b = batch2.reshape(2,4)
# print('****************')
# print(batch2)
# print(b)
#
# x = np.dot(b1,b.T)
# print('****************')
# print(x)
# print(res.shape)

#a=torch.tensor([[[[1, 2.0,3.0],[4, 5,6]],[[7, 8.,9],[10.,11.,12]]]])
a=torch.rand(2,2,2,2)
print(a)
preds=torch.softmax(a, dim=1)

print(preds)
pred,class_idx = torch.max(preds, dim=1)
print(class_idx)
print(pred)

# b = torch.tensor([[[1, 2.],[3., 4.],[5., 6.]]])
# print(b)
# c=torch.matmul(a,b)
# print(c)


# a = torch.randn(1,1000,3,4) #随机生成数组
#
# pred,class_idx=torch.max(a,dim=1) #针对第2个元素“3”，对应的是行
#
# #print("a:\n", a)
# print("************************************************")
# print("pred:\n", pred)
# print("class_idx:\n", class_idx)  #dim=1，列与列之间进行比较，所以返回每一张特征图，每一行的最大值
#
# row_max, row_idx = torch.max(pred, dim=1)
# print("************************************************")
# print("row_max:\n", row_max)
# print("************************************************")
# print("row_idx:\n", row_idx)
# col_max, col_idx = torch.max(row_max, dim=1)
# print("************************************************")
# print("col_max:\n", col_max)
# print("col_idx:\n", col_idx)
# predicted_class = class_idx[0, row_idx[0, col_idx], col_idx]
#
# print("************************************************")
# print("predicted_class:\n", predicted_class)
#predicted_class = class_idx[0, row_idx[0, col_idx], col_idx]