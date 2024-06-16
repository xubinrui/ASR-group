'''
功能：
1.定义模型
2.训练模型
3.保存模型

最后更新时间：
2024.6.15

负责人：崔彣婧
'''
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from joblib import dump
from data_utils import x_train,x_test,y_train,y_test

# Initialize the Multi Layer Perceptron Classifier
model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)
# Train the model
model.fit(x_train,y_train)
# Predict for the test set
y_pred=model.predict(x_test)
# Calculate the accuracy of our model
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)

# Print the accuracy
print("Accuracy: {:.2f}%".format(accuracy*100))


# 保存模型到文件
dump(model, 'mlp_model.joblib')