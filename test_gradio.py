import numpy as np

# 设置随机种子以确保结果可重复
np.random.seed(0)

# 初始值
initial_value = 0.7

# 生成50个随机数
random_numbers = []
for i in range(50):
    # 生成一个在0到0.02之间的随机数，用于波动
    fluctuation = np.random.uniform(-0.04, 0.05)
    # 计算新的值，确保不超过1
    new_value = min(initial_value + fluctuation, 1)
    random_numbers.append(new_value)
    # 更新初始值
    initial_value = new_value

# 打印结果
print(random_numbers)
# 打印结果
print(random_numbers)
for i in random_numbers:
    print(i)