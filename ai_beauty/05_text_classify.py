import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import ast


def test():
    # 读取数据
    training_data = pd.read_csv("data/fine_food_reviews_with_embeddings_1k.csv")

    # 将'embedding'列中的字符串转换为浮点数列表
    training_data['embedding'] = training_data['embedding'].apply(ast.literal_eval)

    # 打印第一条数据以验证转换是否成功
    first_row = training_data.iloc[0]
    print(first_row)

    # 准备训练和测试数据
    X = list(training_data.embedding.values)
    y = training_data.Score

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建并训练随机森林分类器
    clf = RandomForestClassifier(n_estimators=300)
    clf.fit(X_train, y_train)

    # 进行预测
    preds = clf.predict(X_test)

    # 打印分类报告
    report = classification_report(y_test, preds)
    print(report)


if __name__ == '__main__':
    test()

