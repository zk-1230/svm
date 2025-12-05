import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# ========== 1. 32x32 文本图片 -> 1x1024 向量 ==========
def img2vector(file_path):
    """
    将一个 32x32 的 0/1 文本图片转成 1x1024 的 numpy 向量
    """
    vec = np.zeros((1, 1024), dtype=np.float32)
    with open(file_path, 'r') as f:
        for i in range(32):
            line_str = f.readline().strip()
            # 防御：有时候行可能比 32 长/短
            line_str = line_str[:32].ljust(32, '0')
            for j in range(32):
                vec[0, 32 * i + j] = int(line_str[j])
    return vec

# ========== 2. 读取整个数据集 ==========
def load_dataset(dir_path):
    """
    遍历 dir_path 下所有 txt 文件
    文件名格式假定为：  digit_index.txt  例如：1_0.txt, 9_12.txt
    标签 = 文件名中 '_' 前面的数字
    """
    file_list = [f for f in os.listdir(dir_path) if f.endswith('.txt')]
    num_files = len(file_list)

    data_mat = np.zeros((num_files, 1024), dtype=np.float32)
    label_list = []

    for i, file_name in enumerate(file_list):
        full_path = os.path.join(dir_path, file_name)
        data_mat[i, :] = img2vector(full_path)

        # 提取标签
        class_str = file_name.split('_')[0]  # '1_7.txt' -> '1'
        label_list.append(int(class_str))

    return data_mat, np.array(label_list, dtype=np.int32)

# ========== 3. 指定你的训练集 / 测试集目录 ==========
train_dir = r"C:\Users\E507\Documents\GitHub\svm\dataset\trainingDigits"  # 你的trainingDigits路径
test_dir  = r"C:\Users\E507\Documents\GitHub\svm\dataset\testDigits" 

# 验证路径是否存在（新增：提前排查路径问题）
if not os.path.exists(train_dir):
    raise FileNotFoundError(f"训练集路径不存在：{train_dir}")
if not os.path.exists(test_dir):
    raise FileNotFoundError(f"测试集路径不存在：{test_dir}")

X_train, y_train = load_dataset(train_dir)
X_test,  y_test  = load_dataset(test_dir)

print("训练集形状：", X_train.shape, " 标签形状：", y_train.shape)
print("测试集形状：", X_test.shape,  " 标签形状：", y_test.shape)
# 新增：打印数据集真实类别信息（关键）
print("训练集包含的数字类别：", np.unique(y_train))
print("测试集包含的数字类别：", np.unique(y_test))
print("="*50)

# ========== 4. 配置 SVM + GridSearchCV ==========
# 任务 1：使用 SVC + GridSearchCV 在训练集上搜索最优参数
# 创建SVC基础模型（RBF核）
svc = SVC(kernel="rbf")

# 构造参数网格（C为惩罚系数，gamma为RBF核的带宽参数）
param_grid = {
    'C': [0.1, 1, 10, 100],    # 惩罚系数候选值
    'gamma': [0.001, 0.01, 0.1] # 核函数参数候选值
}

# 初始化GridSearchCV
grid_search = GridSearchCV(
    estimator=svc,               # 待优化的基础模型
    param_grid=param_grid,       # 参数搜索网格
    scoring="accuracy",          # 评估指标：准确率
    cv=5,                        # 5折交叉验证
    n_jobs=-1,                   # 使用所有可用CPU核心加速
    verbose=1                    # 输出搜索过程日志
)

# 在训练集上执行网格搜索
print("\n开始网格搜索最优参数...")
grid_search.fit(X_train, y_train)

# 输出最优参数和交叉验证得分
print("\n最优参数组合：", grid_search.best_params_)
print("5折交叉验证最佳平均准确率：", round(grid_search.best_score_, 4))
print("="*50)

# ========== 5. 使用最优模型在测试集上评估 ==========
# 任务 2：使用最优模型评估测试集性能
# 提取网格搜索得到的最优模型
best_clf = grid_search.best_estimator_

# 对测试集进行预测
y_pred = best_clf.predict(X_test)

# 计算测试集准确率
test_acc = accuracy_score(y_test, y_pred)
print("测试集准确率：", round(test_acc, 4))

# 优化：动态生成分类报告的标签（适配真实类别数量）
print("\n详细分类报告：")
# 获取测试集的唯一类别并排序
unique_labels = sorted(np.unique(y_test))
# 动态生成和真实类别匹配的标签名
target_names = [f"数字 {label}" for label in unique_labels]
# 生成分类报告（指定labels参数，避免类别数量不匹配）
print(classification_report(
    y_test, 
    y_pred, 
    labels=unique_labels,       # 指定真实存在的类别
    target_names=target_names,  # 匹配类别的名称
    zero_division=0             # 防止某些类别无预测结果时报错
))