# 第14章：Pandas数据处理

> **前置知识**：NumPy基础（第13章）、Python基础数据结构
>
> **本章目标**：掌握Pandas在数据处理与深度学习预处理中的核心用法

---

## 学习目标

完成本章学习后，你将能够：

1. 理解 Series 和 DataFrame 的结构与基本操作，熟练创建和索引数据
2. 使用 Pandas 读写 CSV、Excel 和 JSON 等常见数据格式
3. 灵活运用 `loc`、`iloc` 及条件筛选精确选取所需数据
4. 处理真实数据中常见的缺失值、重复值和类型错误问题
5. 使用 `groupby`、`agg` 和 `pivot_table` 进行分组统计与透视分析

---

## 14.1 Series 与 DataFrame 基础

### 14.1.1 Series：一维带标签数组

Series 是 Pandas 最基本的数据结构，可以看作一个带有标签（索引）的一维数组。

```python
import pandas as pd
import numpy as np

# 从列表创建 Series
scores = pd.Series([85, 92, 78, 95, 88])
print(scores)
# 0    85
# 1    92
# 2    78
# 3    95
# 4    88
# dtype: int64

# 指定自定义索引
scores = pd.Series(
    [85, 92, 78, 95, 88],
    index=['Alice', 'Bob', 'Charlie', 'Diana', 'Eve']
)
print(scores)
# Alice      85
# Bob        92
# Charlie    78
# Diana      95
# Eve        88
# dtype: int64

# 从字典创建 Series（键作为索引）
data = {'数学': 95, '英语': 88, '物理': 92, '化学': 79}
subject_scores = pd.Series(data)
print(subject_scores)
# 数学    95
# 英语    88
# 物理    92
# 化学    79
# dtype: int64
```

Series 的基本属性：

```python
s = pd.Series([10, 20, 30, 40, 50], index=list('abcde'))

print(s.values)      # array([10, 20, 30, 40, 50])
print(s.index)       # Index(['a', 'b', 'c', 'd', 'e'], dtype='object')
print(s.dtype)       # int64
print(s.shape)       # (5,)
print(s.size)        # 5
print(s.name)        # None（可通过 s.name = '分数' 设置名称）

# 基本统计
print(s.describe())
# count     5.0
# mean     30.0
# std      15.8
# min      10.0
# 25%      20.0
# 50%      30.0
# 75%      40.0
# max      50.0
```

### 14.1.2 DataFrame：二维表格数据

DataFrame 是 Pandas 的核心结构，类似于 Excel 表格或 SQL 数据表，由多个 Series 组成。

```python
# 从字典创建 DataFrame
data = {
    '姓名': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    '年龄': [25, 30, 35, 28, 22],
    '城市': ['北京', '上海', '广州', '深圳', '杭州'],
    '薪资': [15000, 25000, 20000, 30000, 18000]
}
df = pd.DataFrame(data)
print(df)
#       姓名  年龄  城市     薪资
# 0    Alice  25  北京  15000
# 1      Bob  30  上海  25000
# 2  Charlie  35  广州  20000
# 3    Diana  28  深圳  30000
# 4      Eve  22  杭州  18000

# 从 NumPy 数组创建 DataFrame
arr = np.random.randn(4, 3)
df2 = pd.DataFrame(arr, columns=['特征A', '特征B', '特征C'])
print(df2.round(3))
```

DataFrame 的基本属性与方法：

```python
df = pd.DataFrame({
    '姓名': ['Alice', 'Bob', 'Charlie', 'Diana'],
    '年龄': [25, 30, 35, 28],
    '薪资': [15000, 25000, 20000, 30000],
    '部门': ['技术', '产品', '技术', '设计']
})

# 基本信息
print(df.shape)       # (4, 4)
print(df.columns)     # Index(['姓名', '年龄', '薪资', '部门'])
print(df.index)       # RangeIndex(start=0, stop=4, step=1)
print(df.dtypes)
# 姓名    object
# 年龄     int64
# 薪资     int64
# 部门    object

# 快速概览
print(df.head(2))     # 前2行
print(df.tail(2))     # 后2行
df.info()             # 列名、非空数量、类型信息
print(df.describe())  # 数值列统计摘要
```

### 14.1.3 索引操作

```python
df = pd.DataFrame({
    '产品': ['A', 'B', 'C', 'D'],
    '销量': [100, 200, 150, 300],
    '价格': [9.9, 19.9, 14.9, 29.9]
})

# 访问单列（返回 Series）
print(df['销量'])
print(df.销量)         # 等价写法，但不推荐（可能与方法名冲突）

# 访问多列（返回 DataFrame）
print(df[['产品', '价格']])

# 新增列
df['收入'] = df['销量'] * df['价格']
print(df)

# 删除列
df_no_income = df.drop(columns=['收入'])

# 设置索引
df_indexed = df.set_index('产品')
print(df_indexed)
#    销量    价格    收入
# A  100   9.9   990.0
# B  200  19.9  3980.0
# C  150  14.9  2235.0
# D  300  29.9  8970.0

# 重置索引
df_reset = df_indexed.reset_index()
```

---

## 14.2 数据读写

### 14.2.1 CSV 文件

CSV 是数据分析最常见的格式，Pandas 提供了功能强大的读写接口。

```python
# 写入 CSV
df = pd.DataFrame({
    '日期': pd.date_range('2024-01-01', periods=5, freq='D'),
    '销量': [120, 135, 98, 156, 143],
    '城市': ['北京', '上海', '北京', '广州', '上海']
})
df.to_csv('sales.csv', index=False, encoding='utf-8-sig')

# 读取 CSV（基本用法）
df = pd.read_csv('sales.csv')
print(df.head())

# 常用参数详解
df = pd.read_csv(
    'sales.csv',
    encoding='utf-8',        # 编码格式
    sep=',',                 # 分隔符（默认逗号）
    header=0,                # 表头行号（默认第0行）
    index_col=None,          # 指定哪列作为索引
    usecols=['日期', '销量'], # 只读取指定列
    nrows=100,               # 只读取前100行
    dtype={'销量': int},      # 指定列类型
    parse_dates=['日期'],     # 解析日期列
    na_values=['N/A', '-']   # 额外的缺失值标识
)

# 大文件分块读取
chunk_size = 10000
chunks = []
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    # 每块进行预处理
    chunk_filtered = chunk[chunk['销量'] > 100]
    chunks.append(chunk_filtered)
df_large = pd.concat(chunks, ignore_index=True)
```

### 14.2.2 Excel 文件

```python
# 写入 Excel
with pd.ExcelWriter('report.xlsx', engine='openpyxl') as writer:
    df_sales.to_excel(writer, sheet_name='销售数据', index=False)
    df_summary.to_excel(writer, sheet_name='汇总统计', index=False)

# 读取 Excel
df = pd.read_excel(
    'report.xlsx',
    sheet_name='销售数据',   # 指定工作表（默认第一个）
    header=0,
    usecols='A:D',           # 按列字母范围选择
    skiprows=2               # 跳过前2行
)

# 读取所有工作表（返回字典）
all_sheets = pd.read_excel('report.xlsx', sheet_name=None)
for name, sheet_df in all_sheets.items():
    print(f"工作表 '{name}': {sheet_df.shape}")
```

### 14.2.3 JSON 文件

```python
# 写入 JSON
df = pd.DataFrame({
    'id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie'],
    'scores': [85, 92, 78]
})

# records 格式：[{col: val}, ...]（最常用）
df.to_json('data.json', orient='records', force_ascii=False, indent=2)

# 读取 JSON
df = pd.read_json('data.json', orient='records')

# 从 API 响应中解析嵌套 JSON
import json

nested_data = {
    'status': 'ok',
    'data': [
        {'id': 1, 'info': {'name': 'Alice', 'score': 85}},
        {'id': 2, 'info': {'name': 'Bob', 'score': 92}}
    ]
}

# 使用 json_normalize 展开嵌套结构
df = pd.json_normalize(
    nested_data['data'],
    sep='_'    # 嵌套键的连接符
)
print(df)
#    id  info_name  info_score
# 0   1      Alice          85
# 1   2        Bob          92
```

### 14.2.4 其他格式

```python
# Parquet（高效列存储，适合大数据）
df.to_parquet('data.parquet', index=False)
df = pd.read_parquet('data.parquet')

# SQL 数据库
import sqlite3
conn = sqlite3.connect('database.db')
df.to_sql('table_name', conn, if_exists='replace', index=False)
df = pd.read_sql('SELECT * FROM table_name WHERE score > 80', conn)
conn.close()

# 剪贴板（调试时很方便）
# df = pd.read_clipboard()
# df.to_clipboard(index=False)
```

---

## 14.3 数据选择与过滤

### 14.3.1 loc：基于标签的选择

`loc` 使用行/列的标签（索引名称）进行选择，范围是**闭区间**（包含两端）。

```python
df = pd.DataFrame({
    '数学': [85, 92, 78, 95, 88],
    '英语': [90, 85, 82, 78, 95],
    '物理': [75, 88, 90, 85, 80]
}, index=['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'])

# 选择单行（返回 Series）
print(df.loc['Alice'])

# 选择多行（返回 DataFrame）
print(df.loc[['Alice', 'Charlie']])

# 选择行范围（包含 Charlie）
print(df.loc['Alice':'Charlie'])

# 选择行和列
print(df.loc['Alice', '数学'])           # 单值：85
print(df.loc['Alice', ['数学', '英语']]) # 一行两列
print(df.loc['Alice':'Bob', '数学':'英语']) # 行范围 x 列范围

# 修改数据
df.loc['Alice', '数学'] = 90
df.loc['Bob', ['数学', '英语']] = [95, 88]
```

### 14.3.2 iloc：基于位置的选择

`iloc` 使用整数位置进行选择，范围是**左闭右开**（不包含右端），与 Python 切片一致。

```python
# 选择第0行
print(df.iloc[0])

# 选择第0到第2行（不含第2行）
print(df.iloc[0:2])

# 选择第1、3行
print(df.iloc[[1, 3]])

# 选择行和列
print(df.iloc[0, 0])         # 第0行第0列：单值
print(df.iloc[0:3, 0:2])     # 前3行，前2列
print(df.iloc[:, -1])        # 最后一列

# 每隔一行选取
print(df.iloc[::2])
```

### 14.3.3 条件筛选

条件筛选是数据分析中最常用的操作之一。

```python
df = pd.DataFrame({
    '姓名': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    '年龄': [25, 30, 35, 28, 22],
    '城市': ['北京', '上海', '广州', '深圳', '北京'],
    '薪资': [15000, 25000, 20000, 30000, 18000],
    '部门': ['技术', '产品', '技术', '设计', '技术']
})

# 单条件筛选
print(df[df['薪资'] > 20000])

# 多条件（AND：&，OR：|，NOT：~）
print(df[(df['薪资'] > 18000) & (df['城市'] == '北京')])
print(df[(df['城市'] == '北京') | (df['城市'] == '上海')])
print(df[~(df['部门'] == '技术')])  # 非技术部门

# isin：判断是否在列表中
target_cities = ['北京', '上海']
print(df[df['城市'].isin(target_cities)])

# between：范围筛选（含两端）
print(df[df['薪资'].between(18000, 25000)])

# 字符串方法筛选
print(df[df['姓名'].str.startswith('A')])
print(df[df['城市'].str.contains('京|海')])   # 正则支持

# query 方法（更简洁的写法）
print(df.query('薪资 > 20000 and 城市 == "北京"'))
print(df.query('年龄 between 25 and 30'))
```

### 14.3.4 where 与 mask

```python
# where：条件为 True 保留原值，否则替换
result = df['薪资'].where(df['薪资'] >= 20000, other=0)
print(result)
# 0        0
# 1    25000
# 2    20000
# 3    30000
# 4        0

# mask：条件为 True 时替换（where 的反向）
result = df['薪资'].mask(df['薪资'] < 20000, other=20000)
```

---

## 14.4 数据清洗

真实世界的数据几乎总是"脏"的，数据清洗通常占据数据科学项目70%以上的时间。

### 14.4.1 缺失值处理

```python
# 创建含缺失值的数据
df = pd.DataFrame({
    '姓名': ['Alice', 'Bob', None, 'Diana', 'Eve'],
    '年龄': [25, np.nan, 35, 28, np.nan],
    '薪资': [15000, 25000, np.nan, 30000, 18000],
    '部门': ['技术', '产品', '技术', None, '技术']
})

# 检测缺失值
print(df.isnull())           # 每个单元格是否为 NaN
print(df.isnull().sum())     # 每列缺失数量
print(df.isnull().mean())    # 每列缺失比例
print(df.notnull().all())    # 每列是否全部非空

# 查看包含缺失值的行
print(df[df.isnull().any(axis=1)])

# 删除缺失值
df.dropna()                    # 删除含任意 NaN 的行
df.dropna(axis=1)              # 删除含任意 NaN 的列
df.dropna(how='all')           # 删除全为 NaN 的行
df.dropna(subset=['年龄', '薪资'])  # 只检查指定列
df.dropna(thresh=3)            # 保留至少有3个非NaN值的行

# 填充缺失值
df.fillna(0)                        # 所有 NaN 填充为 0
df['年龄'].fillna(df['年龄'].mean())  # 用均值填充
df['部门'].fillna('未知')             # 用常量填充
df.fillna(method='ffill')           # 向前填充（用上一行值）
df.fillna(method='bfill')           # 向后填充（用下一行值）

# 按列指定不同的填充策略
fill_values = {
    '年龄': df['年龄'].median(),
    '薪资': df['薪资'].mean(),
    '部门': '未分配'
}
df_filled = df.fillna(fill_values)

# 插值填充（适用于时序数据）
ts = pd.Series([1.0, np.nan, np.nan, 4.0, 5.0])
print(ts.interpolate())
# 0    1.0
# 1    2.0
# 2    3.0
# 3    4.0
# 4    5.0
```

### 14.4.2 重复值处理

```python
df = pd.DataFrame({
    '用户ID': [1, 2, 2, 3, 4, 4],
    '姓名': ['Alice', 'Bob', 'Bob', 'Charlie', 'Diana', 'Diana'],
    '订单金额': [100, 200, 200, 150, 300, 280]
})

# 检测重复行
print(df.duplicated())         # 每行是否重复
print(df.duplicated().sum())   # 重复行数量

# 按指定列检测重复
print(df.duplicated(subset=['用户ID']))  # 只看用户ID是否重复

# 删除重复行
df.drop_duplicates()                          # 保留第一次出现
df.drop_duplicates(keep='last')               # 保留最后一次出现
df.drop_duplicates(keep=False)                # 删除所有重复项
df.drop_duplicates(subset=['用户ID'], keep='first')  # 按列去重
```

### 14.4.3 类型转换

```python
df = pd.DataFrame({
    '日期': ['2024-01-01', '2024-01-02', '2024-01-03'],
    '销量': ['100', '200', '150'],
    '单价': ['9.9', '19.9', '14.9'],
    '是否促销': ['True', 'False', 'True'],
    '等级': ['A', 'B', 'A']
})

print(df.dtypes)  # 初始类型都是 object

# 转换为数值类型
df['销量'] = df['销量'].astype(int)
df['单价'] = df['单价'].astype(float)
df['是否促销'] = df['是否促销'].map({'True': True, 'False': False})

# 转换为日期类型
df['日期'] = pd.to_datetime(df['日期'])
df['日期'] = pd.to_datetime(df['日期'], format='%Y-%m-%d')

# 提取日期组件
df['年'] = df['日期'].dt.year
df['月'] = df['日期'].dt.month
df['星期'] = df['日期'].dt.day_name()

# 转换为分类类型（节省内存，提高性能）
df['等级'] = df['等级'].astype('category')
print(df['等级'].cat.categories)   # Index(['A', 'B'], dtype='object')

# 处理转换错误
bad_data = pd.Series(['100', '200', 'abc', '300'])
result = pd.to_numeric(bad_data, errors='coerce')  # 错误变为 NaN
print(result)
# 0    100.0
# 1    200.0
# 2      NaN
# 3    300.0
```

### 14.4.4 字符串处理

```python
df = pd.DataFrame({
    '姓名': ['  Alice  ', 'BOB', 'charlie', 'DIANA'],
    '邮箱': ['alice@example.com', 'bob@test.org', None, 'diana@example.com'],
    '手机': ['138-1234-5678', '139 8765 4321', '137-9999-0000', None]
})

# 字符串方法（通过 .str 访问器）
df['姓名'] = df['姓名'].str.strip()          # 去除首尾空格
df['姓名'] = df['姓名'].str.title()          # 首字母大写
df['邮箱_域名'] = df['邮箱'].str.split('@').str[1]  # 提取域名

# 正则表达式处理
df['手机_清洗'] = df['手机'].str.replace(r'[-\s]', '', regex=True)

# 字符串包含/匹配
print(df[df['邮箱'].str.contains('@example.com', na=False)])

# 字符串长度
df['姓名长度'] = df['姓名'].str.len()
```

---

## 14.5 分组聚合

### 14.5.1 groupby 基础

`groupby` 实现了"分割-应用-合并"（Split-Apply-Combine）模式：将数据按某列分组，对每组应用函数，再合并结果。

```python
df = pd.DataFrame({
    '部门': ['技术', '产品', '技术', '设计', '产品', '技术', '设计'],
    '城市': ['北京', '上海', '北京', '广州', '北京', '上海', '广州'],
    '薪资': [20000, 18000, 25000, 15000, 22000, 30000, 17000],
    '年龄': [28, 26, 32, 24, 30, 35, 27]
})

# 按单列分组
grouped = df.groupby('部门')

# 遍历分组
for name, group in grouped:
    print(f"\n{name} 部门：")
    print(group)

# 常用聚合函数
print(grouped['薪资'].mean())    # 各部门平均薪资
print(grouped['薪资'].sum())     # 各部门薪资总和
print(grouped['薪资'].max())     # 各部门最高薪资
print(grouped['薪资'].count())   # 各部门人数
print(grouped['薪资'].std())     # 各部门薪资标准差

# 多列聚合
print(grouped[['薪资', '年龄']].mean())
```

### 14.5.2 agg：多种聚合方式

```python
# 对同一列应用多个函数
result = grouped['薪资'].agg(['mean', 'max', 'min', 'std'])
print(result)
#       mean    max    min          std
# 部门
# 产品  20000  22000  18000  2828.427125
# 技术  25000  30000  20000  5000.000000
# 设计  16000  17000  15000  1414.213562

# 对不同列应用不同函数
result = df.groupby('部门').agg({
    '薪资': ['mean', 'max', 'min'],
    '年龄': ['mean', 'count']
})
print(result)

# 使用具名聚合（推荐，结果列名更清晰）
result = df.groupby('部门').agg(
    平均薪资=('薪资', 'mean'),
    最高薪资=('薪资', 'max'),
    人员数量=('薪资', 'count'),
    平均年龄=('年龄', 'mean')
)
print(result)
#       平均薪资  最高薪资  人员数量  平均年龄
# 部门
# 产品   20000  22000      2    28.0
# 技术   25000  30000      3    31.67
# 设计   16000  17000      2    25.5

# 自定义聚合函数
def range_func(x):
    return x.max() - x.min()

result = df.groupby('部门')['薪资'].agg(
    薪资范围=range_func,
    薪资中位数='median'
)
```

### 14.5.3 多级分组与 transform

```python
# 多列分组
result = df.groupby(['部门', '城市'])['薪资'].mean()
print(result)
# 部门  城市
# 产品  上海    18000
#      北京    22000
# 技术  上海    30000
#      北京    22500
# 设计  广州    16000

# transform：返回与原 DataFrame 等长的结果（不聚合）
# 用于添加分组统计信息到原数据
df['部门平均薪资'] = df.groupby('部门')['薪资'].transform('mean')
df['薪资偏差'] = df['薪资'] - df['部门平均薪资']
print(df[['部门', '薪资', '部门平均薪资', '薪资偏差']])

# 分组内排名
df['部门薪资排名'] = df.groupby('部门')['薪资'].rank(ascending=False)

# filter：筛选满足条件的完整分组
# 只保留部门平均薪资超过20000的分组
high_pay_depts = df.groupby('部门').filter(
    lambda x: x['薪资'].mean() > 20000
)
```

### 14.5.4 pivot_table：数据透视表

```python
# 创建销售数据
sales = pd.DataFrame({
    '月份': ['1月', '1月', '2月', '2月', '3月', '3月'],
    '产品': ['A', 'B', 'A', 'B', 'A', 'B'],
    '地区': ['北区', '南区', '北区', '南区', '北区', '南区'],
    '销量': [100, 150, 120, 180, 90, 200],
    '收入': [1000, 2250, 1200, 2700, 900, 3000]
})

# 基本透视表
pivot = pd.pivot_table(
    sales,
    values='销量',         # 聚合的数值列
    index='月份',          # 行分组
    columns='产品',        # 列分组
    aggfunc='sum',         # 聚合方式
    fill_value=0           # 缺失值填充
)
print(pivot)
# 产品     A    B
# 月份
# 1月    100  150
# 2月    120  180
# 3月     90  200

# 多值聚合
pivot2 = pd.pivot_table(
    sales,
    values=['销量', '收入'],
    index='月份',
    columns='产品',
    aggfunc={'销量': 'sum', '收入': 'mean'},
    margins=True            # 添加行列小计（"All" 行/列）
)
print(pivot2)

# crosstab：快速频次统计
ct = pd.crosstab(sales['月份'], sales['产品'])
ct_pct = pd.crosstab(sales['月份'], sales['产品'], normalize='index')  # 按行归一化
```

---

## 本章小结

| 核心概念 | 关键方法/属性 | 使用场景 |
|----------|--------------|----------|
| **Series** | `.values`, `.index`, `.dtype`, `.describe()` | 一维带标签数据 |
| **DataFrame** | `.shape`, `.dtypes`, `.info()`, `.head()` | 二维表格数据 |
| **CSV读写** | `pd.read_csv()`, `df.to_csv()` | 通用文本表格 |
| **Excel读写** | `pd.read_excel()`, `df.to_excel()` | 与 Office 交互 |
| **JSON读写** | `pd.read_json()`, `pd.json_normalize()` | API 数据解析 |
| **loc** | `df.loc[行标签, 列标签]` | 按名称精确定位 |
| **iloc** | `df.iloc[行位置, 列位置]` | 按位置切片 |
| **条件筛选** | `df[条件]`, `.query()`, `.isin()` | 数据过滤 |
| **缺失值** | `.isnull()`, `.fillna()`, `.dropna()` | 数据完整性 |
| **重复值** | `.duplicated()`, `.drop_duplicates()` | 数据去重 |
| **类型转换** | `.astype()`, `pd.to_datetime()`, `pd.to_numeric()` | 类型规范化 |
| **groupby** | `.groupby().agg()`, `.transform()`, `.filter()` | 分组统计 |
| **透视表** | `pd.pivot_table()`, `pd.crosstab()` | 多维交叉分析 |

**Pandas 数据处理流程总结**：

```
读取数据 → 基本探索(shape/dtypes/describe) → 数据清洗(缺失/重复/类型)
→ 数据选择过滤 → 特征工程 → 分组聚合分析 → 导出结果
```

---

## 深度学习应用：数据预处理流程

在深度学习项目中，原始数据通常存储在 CSV 或数据库中。下面展示一个完整的流程：从原始表格数据到可以输入神经网络的张量。

### 场景：泰坦尼克号乘客生存预测

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

# ─────────────────────────────────────────
# 步骤1：加载数据
# ─────────────────────────────────────────
# 实际项目中从文件或URL加载
# df = pd.read_csv('titanic.csv')

# 模拟数据（结构与泰坦尼克数据集一致）
np.random.seed(42)
n = 200
df = pd.DataFrame({
    'PassengerId': range(1, n + 1),
    'Survived': np.random.randint(0, 2, n),
    'Pclass': np.random.choice([1, 2, 3], n, p=[0.2, 0.3, 0.5]),
    'Name': [f'Passenger_{i}' for i in range(n)],
    'Sex': np.random.choice(['male', 'female'], n),
    'Age': np.where(np.random.random(n) < 0.2, np.nan,
                    np.random.normal(30, 14, n).clip(1, 80)),
    'SibSp': np.random.randint(0, 5, n),
    'Parch': np.random.randint(0, 4, n),
    'Fare': np.random.exponential(32, n),
    'Embarked': np.where(np.random.random(n) < 0.05, np.nan,
                         np.random.choice(['S', 'C', 'Q'], n, p=[0.7, 0.2, 0.1]))
})

print(f"原始数据形状: {df.shape}")
print(f"\n缺失值统计:")
print(df.isnull().sum())

# ─────────────────────────────────────────
# 步骤2：基本探索（EDA）
# ─────────────────────────────────────────
print(f"\n生存率: {df['Survived'].mean():.2%}")
print(f"\n各舱位人数:\n{df['Pclass'].value_counts()}")
print(f"\n年龄统计:\n{df['Age'].describe()}")

# ─────────────────────────────────────────
# 步骤3：特征工程与数据清洗
# ─────────────────────────────────────────

def preprocess_titanic(df: pd.DataFrame) -> pd.DataFrame:
    """泰坦尼克数据预处理流水线"""
    df = df.copy()

    # 3.1 删除无用列（ID和名字对预测无帮助）
    df = df.drop(columns=['PassengerId', 'Name'])

    # 3.2 缺失值处理
    # 年龄：按性别和舱位分组，用中位数填充（比全局中位数更准确）
    age_median = df.groupby(['Sex', 'Pclass'])['Age'].transform('median')
    df['Age'] = df['Age'].fillna(age_median)
    df['Age'] = df['Age'].fillna(df['Age'].median())  # 兜底

    # 登船港口：用众数填充
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # Fare：用中位数填充（极少情况）
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    # 3.3 特征创建
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1  # 家庭总人数
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)  # 是否单身旅行
    df['AgeBin'] = pd.cut(
        df['Age'],
        bins=[0, 12, 18, 35, 60, 100],
        labels=[0, 1, 2, 3, 4]
    ).astype(int)                                      # 年龄分段
    df['FareBin'] = pd.qcut(
        df['Fare'], q=4, labels=[0, 1, 2, 3]
    ).astype(int)                                      # 票价分位数分段

    # 3.4 类别变量编码
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    # One-Hot 编码（适合无序分类）
    embarked_dummies = pd.get_dummies(
        df['Embarked'], prefix='Embarked', drop_first=True
    )
    df = pd.concat([df, embarked_dummies], axis=1)
    df = df.drop(columns=['Embarked'])

    # 3.5 删除已被工程化替代的原始列
    df = df.drop(columns=['SibSp', 'Parch', 'Age', 'Fare'])

    return df

df_processed = preprocess_titanic(df)
print(f"\n预处理后数据形状: {df_processed.shape}")
print(f"\n特征列表: {list(df_processed.columns)}")
print(f"\n数据类型:\n{df_processed.dtypes}")

# ─────────────────────────────────────────
# 步骤4：数值标准化
# ─────────────────────────────────────────

# 分离特征和标签
X = df_processed.drop(columns=['Survived'])
y = df_processed['Survived']

# 需要标准化的连续特征
scale_cols = ['Pclass', 'FamilySize', 'AgeBin', 'FareBin']

# 划分训练集和测试集（在标准化之前划分，防止数据泄露）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 用训练集统计量标准化（测试集用同一个 scaler）
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_test_scaled[scale_cols] = scaler.transform(X_test[scale_cols])

print(f"\n训练集形状: {X_train_scaled.shape}")
print(f"测试集形状: {X_test_scaled.shape}")

# ─────────────────────────────────────────
# 步骤5：转换为 PyTorch Dataset
# ─────────────────────────────────────────

class TitanicDataset(Dataset):
    """自定义 PyTorch 数据集，封装 Pandas DataFrame"""

    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TitanicDataset(X_train_scaled, y_train)
test_dataset = TitanicDataset(X_test_scaled, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 验证数据加载
for X_batch, y_batch in train_loader:
    print(f"\n批次特征形状: {X_batch.shape}")   # [32, 特征数]
    print(f"批次标签形状: {y_batch.shape}")     # [32]
    print(f"特征值范围: [{X_batch.min():.2f}, {X_batch.max():.2f}]")
    break

print("\n✓ 数据预处理完成，已转换为可供模型训练的 DataLoader")
```

### 关键要点总结

| 步骤 | Pandas操作 | 深度学习意义 |
|------|-----------|-------------|
| 缺失值填充 | `fillna()`, `transform('median')` | 避免模型收到 NaN 输入报错 |
| 类别编码 | `map()`, `get_dummies()` | 将字符串转为模型可处理的数值 |
| 特征创建 | 列运算、`pd.cut()` | 帮助模型更容易学习非线性关系 |
| 标准化 | `StandardScaler.fit_transform()` | 统一量纲，加速梯度下降收敛 |
| 防数据泄露 | 先 `train_test_split`，后 `transform` | 确保测试集评估的真实性 |

---

## 练习题

### 基础题

**练习 14-1：学生成绩分析**

创建以下 DataFrame，完成指定分析：

```python
data = {
    '学生': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank'],
    '班级': ['A', 'A', 'B', 'B', 'A', 'B'],
    '数学': [85, 92, None, 95, 78, 88],
    '英语': [90, 85, 82, None, 95, 76],
    '物理': [75, 88, 90, 85, None, 92]
}
```

要求：
1. 用各科全体均值填充缺失值
2. 新增"总分"列和"平均分"列
3. 筛选出平均分大于 85 的学生
4. 按班级统计各科平均分

**练习 14-2：数据类型清洗**

有如下"脏数据"，请完成清洗：

```python
dirty_df = pd.DataFrame({
    '日期': ['2024/01/15', '2024-02-20', '20240315', '2024.04.10'],
    '金额': ['¥1,200.00', '$800', '1500元', '2,000.50'],
    '状态': [' 已完成 ', 'PENDING', 'completed', ' Failed ']
})
```

要求：
1. 将"日期"列统一解析为 datetime 类型
2. 将"金额"列清洗为 float 类型（去除货币符号、逗号等）
3. 将"状态"列标准化为：`completed`、`pending`、`failed`

---

### 进阶题

**练习 14-3：电商订单分析**

```python
import pandas as pd
import numpy as np

np.random.seed(0)
n = 500
orders = pd.DataFrame({
    '订单ID': range(1, n+1),
    '用户ID': np.random.randint(1, 100, n),
    '商品类别': np.random.choice(['电子', '服装', '食品', '图书', '家居'], n),
    '金额': np.random.exponential(200, n).round(2),
    '下单时间': pd.date_range('2024-01-01', periods=n, freq='12H'),
    '是否退款': np.random.choice([True, False], n, p=[0.1, 0.9])
})
```

完成以下分析：
1. 计算每个用户的：订单数量、总消费金额、平均订单金额、退款率
2. 找出消费金额前10%的"高价值用户"
3. 按月份和商品类别制作销售额透视表
4. 找出连续下单（相邻订单间隔不超过24小时）的用户

**练习 14-4：时序数据处理**

```python
# 生成每日股票价格数据
dates = pd.date_range('2023-01-01', '2024-12-31', freq='B')  # 工作日
np.random.seed(42)
price = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.015, len(dates)))
stock = pd.DataFrame({'日期': dates, '收盘价': price.round(2)})
stock = stock.set_index('日期')
```

完成以下任务：
1. 计算每日涨跌幅（百分比）
2. 计算5日、20日、60日移动均线
3. 找出单日涨幅最大的前5个交易日
4. 按季度统计最高价、最低价、平均价、波动率（标准差）
5. 计算每个月末最后一个交易日的收盘价

---

### 挑战题

**练习 14-5：完整的机器学习数据预处理流水线**

设计并实现一个通用的数据预处理类 `DataPreprocessor`，满足以下要求：

```python
class DataPreprocessor:
    """通用数据预处理流水线"""

    def __init__(self, target_col: str, drop_cols: list = None):
        """
        target_col: 目标列名
        drop_cols: 需要删除的列名列表
        """
        ...

    def fit(self, df: pd.DataFrame) -> 'DataPreprocessor':
        """
        在训练集上学习：
        - 识别数值列和类别列
        - 学习数值列的填充值（中位数）和标准化参数
        - 学习类别列的填充值（众数）和编码映射
        返回 self（支持链式调用）
        """
        ...

    def transform(self, df: pd.DataFrame) -> tuple:
        """
        应用已学习的变换（可用于训练集和测试集）：
        - 填充缺失值
        - 标准化数值特征
        - One-Hot 编码类别特征
        返回 (X_array: np.ndarray, y_array: np.ndarray)
        """
        ...

    def fit_transform(self, df: pd.DataFrame) -> tuple:
        """fit 和 transform 的组合"""
        return self.fit(df).transform(df)
```

**测试用例**：

```python
# 构造测试数据
train_data = pd.DataFrame({
    'age': [25, 30, np.nan, 35, 28],
    'income': [50000, 75000, 60000, np.nan, 45000],
    'city': ['北京', '上海', '北京', np.nan, '广州'],
    'education': ['本科', '硕士', '本科', '博士', np.nan],
    'label': [0, 1, 0, 1, 0]
})

test_data = pd.DataFrame({
    'age': [27, np.nan, 40],
    'income': [55000, 80000, np.nan],
    'city': ['北京', '深圳', '上海'],   # 注意：深圳在训练集中未出现
    'education': ['硕士', '本科', '博士'],
    'label': [0, 1, 1]
})

preprocessor = DataPreprocessor(target_col='label', drop_cols=None)
X_train, y_train = preprocessor.fit_transform(train_data)
X_test, y_test = preprocessor.transform(test_data)

print(f"训练集特征形状: {X_train.shape}")
print(f"测试集特征形状: {X_test.shape}")
# 要求：训练集和测试集特征数量必须相同
assert X_train.shape[1] == X_test.shape[1], "特征维度不一致！"
print("测试通过！")
```

---

## 练习答案

### 练习 14-1 答案

```python
import pandas as pd
import numpy as np

data = {
    '学生': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank'],
    '班级': ['A', 'A', 'B', 'B', 'A', 'B'],
    '数学': [85, 92, None, 95, 78, 88],
    '英语': [90, 85, 82, None, 95, 76],
    '物理': [75, 88, 90, 85, None, 92]
}
df = pd.DataFrame(data)

# 1. 用各科均值填充缺失值
for col in ['数学', '英语', '物理']:
    df[col] = df[col].fillna(df[col].mean())

# 2. 新增总分和平均分列
df['总分'] = df['数学'] + df['英语'] + df['物理']
df['平均分'] = df[['数学', '英语', '物理']].mean(axis=1).round(2)

# 3. 筛选平均分大于85的学生
high_score = df[df['平均分'] > 85][['学生', '班级', '平均分']]
print("平均分大于85的学生：")
print(high_score)

# 4. 按班级统计各科平均分
class_avg = df.groupby('班级')[['数学', '英语', '物理']].mean().round(2)
print("\n各班级科目平均分：")
print(class_avg)
```

### 练习 14-2 答案

```python
dirty_df = pd.DataFrame({
    '日期': ['2024/01/15', '2024-02-20', '20240315', '2024.04.10'],
    '金额': ['¥1,200.00', '$800', '1500元', '2,000.50'],
    '状态': [' 已完成 ', 'PENDING', 'completed', ' Failed ']
})

# 1. 统一解析日期（pd.to_datetime 支持多种格式）
dirty_df['日期'] = pd.to_datetime(
    dirty_df['日期'].str.replace(r'[./]', '-', regex=True)
)

# 2. 清洗金额：去除非数字字符（保留小数点）
dirty_df['金额'] = (
    dirty_df['金额']
    .str.replace(r'[¥$元,\s]', '', regex=True)
    .astype(float)
)

# 3. 标准化状态列
status_map = {
    ' 已完成 ': 'completed', 'PENDING': 'pending',
    'completed': 'completed', ' Failed ': 'failed'
}
dirty_df['状态'] = dirty_df['状态'].map(status_map)

# 更健壮的方式：先strip和lower，再映射
dirty_df['状态'] = (
    dirty_df['状态']
    .str.strip()
    .str.lower()
    .map({'已完成': 'completed', 'pending': 'pending',
          'completed': 'completed', 'failed': 'failed'})
)

print(dirty_df)
print(dirty_df.dtypes)
```

### 练习 14-3 答案

```python
import pandas as pd
import numpy as np

np.random.seed(0)
n = 500
orders = pd.DataFrame({
    '订单ID': range(1, n+1),
    '用户ID': np.random.randint(1, 100, n),
    '商品类别': np.random.choice(['电子', '服装', '食品', '图书', '家居'], n),
    '金额': np.random.exponential(200, n).round(2),
    '下单时间': pd.date_range('2024-01-01', periods=n, freq='12H'),
    '是否退款': np.random.choice([True, False], n, p=[0.1, 0.9])
})

# 1. 用户统计
user_stats = orders.groupby('用户ID').agg(
    订单数量=('订单ID', 'count'),
    总消费金额=('金额', 'sum'),
    平均订单金额=('金额', 'mean'),
    退款率=('是否退款', 'mean')
).round(2)
print("用户统计（前5行）：")
print(user_stats.head())

# 2. 高价值用户（消费总额前10%）
threshold = user_stats['总消费金额'].quantile(0.9)
high_value_users = user_stats[user_stats['总消费金额'] >= threshold]
print(f"\n高价值用户数量: {len(high_value_users)}")

# 3. 月份 x 类别 销售额透视表
orders['月份'] = orders['下单时间'].dt.month
pivot = pd.pivot_table(
    orders,
    values='金额',
    index='月份',
    columns='商品类别',
    aggfunc='sum',
    fill_value=0
).round(2)
print("\n月度销售额透视表：")
print(pivot)

# 4. 找出连续下单用户
orders_sorted = orders.sort_values(['用户ID', '下单时间'])
orders_sorted['上次下单'] = orders_sorted.groupby('用户ID')['下单时间'].shift(1)
orders_sorted['间隔小时'] = (
    orders_sorted['下单时间'] - orders_sorted['上次下单']
).dt.total_seconds() / 3600

consecutive_users = orders_sorted[
    orders_sorted['间隔小时'] <= 24
]['用户ID'].unique()
print(f"\n连续下单用户数量: {len(consecutive_users)}")
```

### 练习 14-4 答案

```python
import pandas as pd
import numpy as np

dates = pd.date_range('2023-01-01', '2024-12-31', freq='B')
np.random.seed(42)
price = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.015, len(dates)))
stock = pd.DataFrame({'收盘价': price.round(2)}, index=dates)

# 1. 每日涨跌幅
stock['涨跌幅'] = stock['收盘价'].pct_change() * 100

# 2. 移动均线
stock['MA5'] = stock['收盘价'].rolling(window=5).mean()
stock['MA20'] = stock['收盘价'].rolling(window=20).mean()
stock['MA60'] = stock['收盘价'].rolling(window=60).mean()

# 3. 单日涨幅最大的5个交易日
top5 = stock['涨跌幅'].nlargest(5)
print("涨幅最大的5个交易日：")
print(top5.round(2))

# 4. 按季度统计
quarterly = stock['收盘价'].resample('Q').agg(
    最高价='max',
    最低价='min',
    平均价='mean',
    波动率='std'
).round(2)
print("\n季度统计：")
print(quarterly)

# 5. 每月末最后一个交易日收盘价
month_end = stock['收盘价'].resample('ME').last()
print("\n月末收盘价（前6个月）：")
print(month_end.head(6).round(2))
```

### 练习 14-5 答案

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    """通用数据预处理流水线"""

    def __init__(self, target_col: str, drop_cols: list = None):
        self.target_col = target_col
        self.drop_cols = drop_cols or []
        self._num_cols = []
        self._cat_cols = []
        self._num_fill_values = {}
        self._cat_fill_values = {}
        self._scaler = StandardScaler()
        self._cat_columns_after_encode = []   # 编码后的列名（用于对齐）

    def fit(self, df: pd.DataFrame) -> 'DataPreprocessor':
        work = df.drop(columns=self.drop_cols + [self.target_col], errors='ignore')

        # 识别数值列和类别列
        self._num_cols = work.select_dtypes(include=[np.number]).columns.tolist()
        self._cat_cols = work.select_dtypes(include=['object', 'category']).columns.tolist()

        # 学习填充值
        for col in self._num_cols:
            self._num_fill_values[col] = work[col].median()
        for col in self._cat_cols:
            self._cat_fill_values[col] = work[col].mode()[0] if not work[col].mode().empty else 'unknown'

        # 用训练集的分布进行标准化拟合
        work_filled = work.copy()
        for col in self._num_cols:
            work_filled[col] = work_filled[col].fillna(self._num_fill_values[col])

        self._scaler.fit(work_filled[self._num_cols])

        # 记录 One-Hot 编码后的列名（用训练集确定）
        cat_filled = work_filled.copy()
        for col in self._cat_cols:
            cat_filled[col] = cat_filled[col].fillna(self._cat_fill_values[col])
        dummies = pd.get_dummies(cat_filled[self._cat_cols], drop_first=True)
        self._cat_columns_after_encode = dummies.columns.tolist()

        return self

    def transform(self, df: pd.DataFrame) -> tuple:
        work = df.drop(columns=self.drop_cols, errors='ignore').copy()

        # 提取目标列
        y = work.pop(self.target_col).values if self.target_col in work.columns else None

        # 填充缺失值
        for col in self._num_cols:
            if col in work.columns:
                work[col] = work[col].fillna(self._num_fill_values[col])
        for col in self._cat_cols:
            if col in work.columns:
                work[col] = work[col].fillna(self._cat_fill_values[col])

        # 标准化数值列
        existing_num_cols = [c for c in self._num_cols if c in work.columns]
        work[existing_num_cols] = self._scaler.transform(work[existing_num_cols])

        # One-Hot 编码类别列
        existing_cat_cols = [c for c in self._cat_cols if c in work.columns]
        dummies = pd.get_dummies(work[existing_cat_cols], drop_first=True)

        # 对齐列：确保与训练集相同的列结构（处理未见过的类别）
        dummies = dummies.reindex(columns=self._cat_columns_after_encode, fill_value=0)

        work = work.drop(columns=existing_cat_cols)
        work = pd.concat([work, dummies], axis=1)

        X = work.values.astype(np.float32)
        return X, y

    def fit_transform(self, df: pd.DataFrame) -> tuple:
        return self.fit(df).transform(df)


# ── 测试 ──
train_data = pd.DataFrame({
    'age': [25, 30, np.nan, 35, 28],
    'income': [50000, 75000, 60000, np.nan, 45000],
    'city': ['北京', '上海', '北京', np.nan, '广州'],
    'education': ['本科', '硕士', '本科', '博士', np.nan],
    'label': [0, 1, 0, 1, 0]
})

test_data = pd.DataFrame({
    'age': [27, np.nan, 40],
    'income': [55000, 80000, np.nan],
    'city': ['北京', '深圳', '上海'],
    'education': ['硕士', '本科', '博士'],
    'label': [0, 1, 1]
})

preprocessor = DataPreprocessor(target_col='label')
X_train, y_train = preprocessor.fit_transform(train_data)
X_test, y_test = preprocessor.transform(test_data)

print(f"训练集特征形状: {X_train.shape}")
print(f"测试集特征形状: {X_test.shape}")
assert X_train.shape[1] == X_test.shape[1], "特征维度不一致！"
print("所有测试通过！")

# 输出特征预览
print(f"\n训练集特征预览（前3行）：")
print(X_train[:3].round(3))
```

---

> **下一章预告**：第15章将介绍 Matplotlib 与 Seaborn 数据可视化，你将学会如何把本章处理好的数据用图表直观呈现，以及如何可视化模型训练过程中的损失曲线与评估指标。
