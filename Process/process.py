import os
import pandas as pd
import numpy as np
import torch
from Bert import data_loader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
################################# load data ###################################


def loadData(fold_x_train, fold_x_test):
    def analyze_static_features(df, features_dict=None):
        """
        分析静态特征的重要性和影响力

        Parameters:
        -----------
        df : pandas.DataFrame
            包含原始特征和标签的数据框
        features_dict : dict
            特征名称的中文映射字典
        """
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        df = df.groupby('student_id').first().reset_index()
        df = df.drop_duplicates(subset=['student_id'])
        # 定义特征列
        one_hot_columns = [col for col in ['major', 'grade', 'sex（1男，2女）']
                           if col in df.columns]
        numeric_columns = ['gpa', 'gks（挂科数）', '是否低保（1是，0否）', '重残人家庭', '单亲家庭', '孤儿']

        # 特征名称映射
        if features_dict is None:
            features_dict = {
                'gpa': 'GPA',
                'gks（挂科数）': 'Number of failed courses',
                '是否低保（1是，0否）': 'Low-income family',
                '重残人家庭': 'Severely disabled family',
                '单亲家庭': 'Single-parent family',
                '孤儿': 'Orphan',
                'major': 'Major',
                'grade': 'Grade',
                'sex（1男，2女）': 'Gender'
            }

        # 准备数据
        X_numeric = df[numeric_columns].copy()
        X_categorical = pd.get_dummies(df[one_hot_columns])
        y = df['关注等级（1密切、2重点、3日常、4日常）']

        # 标准化数值特征
        scaler = StandardScaler()
        X_numeric_scaled = pd.DataFrame(
            scaler.fit_transform(X_numeric),
            columns=X_numeric.columns
        )

        plt.figure(figsize=(15, 15))

        # 1. 数值特征与标签的箱线图
        plt.subplot(221)
        feature_importance = {}
        for feature in numeric_columns:
            mi_score = mutual_info_classif(
                X_numeric_scaled[[feature]],
                y,
                discrete_features=False
            )[0]
            feature_importance[features_dict[feature]] = mi_score

        # 绘制特征重要性条形图
        importance_df = pd.DataFrame(
            list(feature_importance.items()),
            columns=['Feature', 'Importance']
        ).sort_values('Importance', ascending=True)

        sns.barplot(data=importance_df, y='Feature', x='Importance')
        plt.title('Numerical Feature Importance Analysis (Based on Mutual Information)', fontsize=14)
        plt.xlabel('Mutual Information Score', fontsize=12)
        plt.ylabel('Feature', fontsize=12)

        # 2. 数值特征相关性热力图
        plt.subplot(222)
        correlation_matrix = X_numeric_scaled.corr()
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap='coolwarm',
            center=0,
            xticklabels=[features_dict[col] for col in numeric_columns],
            yticklabels=[features_dict[col] for col in numeric_columns]
        )
        plt.title('Numerical Feature Correlation Analysis', fontsize=14)

        # 3. 类别特征的分布
        plt.subplot(223)
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        X_combined = pd.concat([X_numeric_scaled, X_categorical], axis=1)
        rf_model.fit(X_combined, y)

        feature_imp = pd.DataFrame({
            'Feature': [features_dict.get(col, col) for col in X_combined.columns],
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=True)

        sns.barplot(data=feature_imp.tail(10), y='Feature', x='Importance')
        plt.title('Random Forest Feature Importance (Top 10)', fontsize=14)
        plt.xlabel('Feature Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)

        plt.tight_layout()
        plt.show()

        # 打印详细的特征分析报告
        print("\n=== 特征重要性分析报告 ===")
        # 替换随机森林部分的代码
        # 3. 使用LASSO进行特征重要性分析
        plt.subplot(223)
        lasso = LogisticRegressionCV(
            cv=5,
            penalty='l1',
            solver='saga',
            multi_class='multinomial',
            max_iter=8000,
            random_state=42,
            tol=1e-4
        )
        X_combined = pd.concat([X_numeric_scaled, X_categorical], axis=1)
        lasso.fit(X_combined, y)

        feature_imp = pd.DataFrame({
            'Feature': [features_dict.get(col, col) for col in X_combined.columns],
            'Importance': np.sum(np.abs(lasso.coef_), axis=0)
        }).sort_values('Importance', ascending=True)

        sns.barplot(data=feature_imp.tail(10), y='Feature', x='Importance')
        plt.title('LASSO Feature Importance (Top 10)', fontsize=14)
        plt.xlabel('Feature Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        print("\n=== LASSO Feature Importance (Top 10) ===")
        for idx, row in feature_imp.tail(10).iterrows():
            print(f"{row['Feature']}: {row['Importance']:.4f}")

        print("\n2. 基于随机森林的整体特征重要性 (Top 10):")
        for idx, row in feature_imp.tail(10).iterrows():
            print(f"{row['Feature']}: {row['Importance']:.4f}")

        # 生成特征对预测的影响力报告
        results = permutation_importance(
            rf_model, X_combined, y,
            n_repeats=10,
            random_state=42
        )

        print("\n3. 特征排列重要性 (Top 10):")
        perm_imp_df = pd.DataFrame({
            'Feature': [features_dict.get(col, col) for col in X_combined.columns],
            'Importance_mean': results.importances_mean,
            'Importance_std': results.importances_std
        }).sort_values('Importance_mean', ascending=False)

        for idx, row in perm_imp_df.head(10).iterrows():
            print(f"{row['Feature']}: {row['Importance_mean']:.4f} ± {row['Importance_std']:.4f}")

        return feature_imp, importance_df, perm_imp_df
    def visualize_feature_distribution(train_selected, labels, method='pca', interactive=False, save_path=None):
        """
        使用多种降维方法可视化特征分布

        Parameters:
        -----------
        train_selected : numpy.ndarray
            原始特征数据
        labels : numpy.ndarray
            数据标签
        method : str
            降维方法，可选 'pca', 'tsne', 'umap'
        interactive : bool
            是否使用交互式图表
        save_path : str
            保存路径
        """
        # 设置参数
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        n_components = 3
        labels_dict = {0: 'Close Attention', 1: 'Key Attention', 2: 'Routine Attention', 3: 'Normal'}
        colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']

        # 降维
        if method == 'pca':
            reducer = PCA(
                n_components=n_components,  # 保持原有的维度设置
                svd_solver='full',  # 选择SVD求解器，可选'auto', 'full', 'arpack', 'randomized'
                whiten=True,  # 白化处理，使得每个特征的方差为1
                random_state=42  # 设置随机种子，确保结果可重现
            )
            title = 'PCA Dimensionality Reduction Visualization'
        elif method == 'tsne':
            reducer = TSNE(n_components=n_components, perplexity=30, n_iter=6000)
            title = 't-SNE Dimensionality Reduction Visualization'
        elif method == 'umap':
            reducer = UMAP(n_components=n_components, n_neighbors=30, min_dist=0.1)
            title = 'UMAP Dimensionality Reduction Visualization'

        reduced_features = reducer.fit_transform(train_selected)

        if interactive:
            # 创建用于Plotly的数据框
            df = pd.DataFrame(reduced_features, columns=['Dimension 1', 'Dimension 2', 'Dimension 3'])
            df['Label'] = [labels_dict[l] for l in labels]

            # 创建交互式3D散点图
            fig = px.scatter_3d(df, x='Dimension 1', y='Dimension 2', z='Dimension 3',
                                color='Label', title=title,
                                color_discrete_sequence=colors,
                                opacity=0.7)

            # 更新布局
            fig.update_layout(
                scene=dict(
                    xaxis_title='Dimension 1',
                    yaxis_title='Dimension 2',
                    zaxis_title='Dimension 3'
                ),
                width=1200,
                height=800,
                showlegend=True
            )

            # 保存或显示
            if save_path:
                fig.write_html(save_path)
            else:
                fig.show()

        else:
            # Create static charts
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False

            fig = plt.figure(figsize=(20, 15))

            # 3D scatter plot
            ax1 = fig.add_subplot(221, projection='3d')
            for i, label in enumerate(sorted(np.unique(labels))):
                mask = labels == label
                ax1.scatter(reduced_features[mask, 0],
                            reduced_features[mask, 1],
                            reduced_features[mask, 2],
                            c=colors[i],
                            label=labels_dict[label],
                            s=100,
                            alpha=0.6)

            ax1.set_xlabel('Dimension 1', fontsize=12)
            ax1.set_ylabel('Dimension 2', fontsize=12)
            ax1.set_zlabel('Dimension 3', fontsize=12)
            ax1.set_title(title, fontsize=14)
            ax1.legend(fontsize=10)

            # 2D projection
            ax2 = fig.add_subplot(222)
            for i, label in enumerate(sorted(np.unique(labels))):
                mask = labels == label
                ax2.scatter(reduced_features[mask, 0],
                            reduced_features[mask, 1],
                            c=colors[i],
                            label=labels_dict[label],
                            s=100,
                            alpha=0.6)
            ax2.set_xlabel('Dimension 1', fontsize=12)
            ax2.set_ylabel('Dimension 2', fontsize=12)
            ax2.set_title('2D Projection (Dimension 1 vs Dimension 2)', fontsize=14)
            ax2.legend(fontsize=10)

            # Density distribution
            ax3 = fig.add_subplot(223)
            for i, label in enumerate(sorted(np.unique(labels))):
                mask = labels == label
                sns.kdeplot(data=reduced_features[mask, 0],
                            label=labels_dict[label],
                            color=colors[i],
                            ax=ax3)
            ax3.set_xlabel('Dimension 1', fontsize=12)
            ax3.set_ylabel('Density', fontsize=12)
            ax3.set_title('Density Distribution of Dimension 1', fontsize=14)
            ax3.legend(fontsize=10)

            plt.tight_layout()

            # if save_path:
            #    plt.savefig(save_path, dpi=300, bbox_inches='tight')
            # else:
            plt.show()

    def compare_dimension_reduction_methods(train_selected, labels):
        """比较不同降维方法的效果"""

        # PCA
        visualize_feature_distribution(
            train_selected=train_selected,
            labels=labels,
            method='pca',
            interactive=False,
            save_path='pca_visualization.html'
        )

        # t-SNE
        visualize_feature_distribution(
            train_selected=train_selected,
            labels=labels,
            method='tsne',
            interactive=False,
            save_path='tsne_visualization.html'
        )

        # UMAP
        visualize_feature_distribution(
            train_selected=train_selected,
            labels=labels,
            method='umap',
            interactive=False,
            save_path='umap_visualization.html'
        )

    def fit_transform_features(train_df, test_df):
        train_df = train_df.drop_duplicates(subset=['student_id'])
        test_df = test_df.drop_duplicates(subset=['student_id'])
        train_ids = train_df['student_id']
        test_ids = test_df['student_id']

        # 特征列定义
        one_hot_columns = [col for col in ['grade', 'sex（1男，2女）']
                           if col in train_df.columns]
        numeric_columns = ['gpa', 'gks（挂科数）', '是否低保（1是，0否）', '重残人家庭', '单亲家庭', '孤儿']

        # One-hot编码处理
        one_hot_encoder = pd.get_dummies(train_df[one_hot_columns])
        train_encoded = pd.get_dummies(train_df[one_hot_columns])
        test_encoded = pd.get_dummies(test_df[one_hot_columns])

        # 确保测试集有相同的one-hot列
        missing_cols = set(train_encoded.columns) - set(test_encoded.columns)
        for col in missing_cols:
            test_encoded[col] = 0
        test_encoded = test_encoded[train_encoded.columns]

        # 合并特征
        train_features = pd.concat([train_df[numeric_columns], train_encoded], axis=1)
        test_features = pd.concat([test_df[numeric_columns], test_encoded], axis=1)

        # 标准化
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_features)
        test_scaled = scaler.transform(test_features)

        # 降维前的可视化
        visualize_feature_distribution(
            train_scaled,
            train_df['关注等级（1密切、2重点、3日常、4日常）'].values,
            method='pca',
            interactive=True,
            save_path='before_lasso_pca_visualization.html'
        )

        # Lasso特征选择
        lasso_model = LogisticRegressionCV(
            cv=5,
            penalty='l1',
            solver='saga',
            multi_class='multinomial',
            max_iter=30000,
            random_state=42,
            tol=1e-4
        )
        lasso_model.fit(train_scaled, train_df['关注等级（1密切、2重点、3日常、4日常）'])

        # 基于LASSO的特征重要性
        feature_importance = np.sum(np.abs(lasso_model.coef_), axis=0)
        feature_names = list(train_features.columns)

        # 创建特征重要性DataFrame
        lasso_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=True)

        # 显示LASSO特征重要性
        plt.figure(figsize=(10, 6))
        sns.barplot(data=lasso_importance_df.tail(10), y='Feature', x='Importance')
        plt.title('LASSO Feature Importance (Top 10)', fontsize=14)
        plt.xlabel('Feature Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        plt.show()

        print("\n=== LASSO Feature Importance (Top 10) ===")
        for idx, row in lasso_importance_df.tail(10).iterrows():
            print(f"{row['Feature']}: {row['Importance']:.4f}")

        # 特征选择
        selected_features_mask = feature_importance > (np.mean(feature_importance) * 1.0)
        train_selected = train_scaled[:, selected_features_mask]
        test_selected = test_scaled[:, selected_features_mask]

        # 降维后的可视化
        visualize_feature_distribution(
            train_selected,
            train_df['关注等级（1密切、2重点、3日常、4日常）'].values,
            method='pca',
            interactive=True,
            save_path='after_lasso_pca_visualization.html'
        )

        # PCA降维
        pca = PCA(n_components=0.95)
        train_pca = pca.fit_transform(train_selected)
        test_pca = pca.transform(test_selected)

        # 创建最终特征字典
        train_features_dict = pd.DataFrame(train_pca, index=train_ids)
        test_features_dict = pd.DataFrame(test_pca, index=test_ids)

        return train_features_dict, test_features_dict, train_pca.shape[1]
    # 获取转换后的特征
    train_features, test_features, num_columns = fit_transform_features(fold_x_train, fold_x_test)

    def process_data(df, features):
        result = {}
        for student_id, group in df.groupby('student_id'):
            result[student_id] = {
                'content': {
                    i: group[group['theme'].str.contains(str(i))]['content'].tolist()
                    for i in range(1, 6)
                },
                'tags': {
                    i: data_loader(group[group['theme'].str.contains(str(i))]['content'].tolist())
                    for i in range(1, 6)
                    if group[group['theme'].str.contains(str(i))]['content'].tolist()
                },
                'tag': group['关注等级（1密切、2重点、3日常、4日常）'].iloc[0],
                'feature': torch.tensor(features.loc[student_id].values, dtype=torch.float32)
            }
        return result

    train_content = process_data(fold_x_train, train_features)
    test_content = process_data(fold_x_test, test_features)

    return train_content, test_content, num_columns