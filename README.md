# 机器学习项目

## 项目结构

本项目采用以下文件夹结构，以实现文件的有序分类与高效组织：

```
ml/
├── src/              # 源代码文件夹
│   ├── __pycache__/  # Python缓存文件
│   ├── binning_methods.py
│   ├── feature_binning_analyzer.py
│   └── pr_fea_sel.py
├── data/             # 数据文件夹
│   ├── data.csv
│   ├── binned_selected_features_data.csv
│   └── selected_features_data.csv
├── results/          # 结果文件夹
│   ├── enhanced_pr_feature_selection_results.png
│   └── feature_binning_analysis.png
├── docs/             # 文档文件夹
├── tests/            # 测试文件夹
└── README.md         # 项目说明文件
```

## 文件夹说明

- **src/**: 存放所有Python源代码文件，包括数据处理、特征选择和分析等脚本。
- **data/**: 存放项目使用的所有数据文件，包括原始数据和处理后的数据。
- **results/**: 存放项目运行生成的结果文件，如图表、模型输出等。
- **docs/**: 存放项目相关文档，如使用说明、API文档等。
- **tests/**: 存放测试脚本，用于验证代码功能。

## 使用说明

1. 源代码文件位于`src/`目录下，可以根据需要修改和扩展。
2. 数据文件位于`data/`目录下，运行代码前请确保数据文件路径正确。
3. 运行结果将保存在`results/`目录下。
4. 如需添加文档，请将其放在`docs/`目录下。
5. 如需添加测试用例，请将其放在`tests/`目录下。