{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "e9b1ad98",
   "metadata": {
    "_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19",
    "_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5",
    "execution": {
     "iopub.execute_input": "2023-03-28T00:58:06.759282Z",
     "iopub.status.busy": "2023-03-28T00:58:06.758893Z",
     "iopub.status.idle": "2023-03-28T00:58:06.772634Z",
     "shell.execute_reply": "2023-03-28T00:58:06.771590Z"
    },
    "papermill": {
     "duration": 0.02256,
     "end_time": "2023-03-28T00:58:06.775831",
     "exception": false,
     "start_time": "2023-03-28T00:58:06.753271",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "/kaggle/input/digit-recognizer/sample_submission.csv\n",
      "/kaggle/input/digit-recognizer/train.csv\n",
      "/kaggle/input/digit-recognizer/test.csv\n"
     ]
    }
   ],
   "source": [
    "# This Python 3 environment comes with many helpful analytics libraries installed\n",
    "# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python\n",
    "# For example, here's several helpful packages to load\n",
    "\n",
    "import numpy as np # linear algebra\n",
    "import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\n",
    "\n",
    "# Input data files are available in the read-only \"../input/\" directory\n",
    "# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory\n",
    "\n",
    "import os\n",
    "for dirname, _, filenames in os.walk('/kaggle/input'):\n",
    "    for filename in filenames:\n",
    "        print(os.path.join(dirname, filename))\n",
    "\n",
    "# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using \"Save & Run All\" \n",
    "# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "485482c7",
   "metadata": {
    "papermill": {
     "duration": 0.003027,
     "end_time": "2023-03-28T00:58:06.782943",
     "exception": false,
     "start_time": "2023-03-28T00:58:06.779916",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Loading data"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "04c3b44a",
   "metadata": {
    "papermill": {
     "duration": 0.002973,
     "end_time": "2023-03-28T00:58:06.789057",
     "exception": false,
     "start_time": "2023-03-28T00:58:06.786084",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "**Training data**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "f0380e7f",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-03-28T00:58:06.797117Z",
     "iopub.status.busy": "2023-03-28T00:58:06.796245Z",
     "iopub.status.idle": "2023-03-28T00:58:21.004857Z",
     "shell.execute_reply": "2023-03-28T00:58:21.003681Z"
    },
    "papermill": {
     "duration": 14.215632,
     "end_time": "2023-03-28T00:58:21.007805",
     "exception": false,
     "start_time": "2023-03-28T00:58:06.792173",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# import necessary libraries\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import accuracy_score, confusion_matrix\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from tensorflow.keras.datasets import mnist\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten\n",
    "from tensorflow.keras.utils import to_categorical\n",
    "\n",
    "# load the MNIST dataset\n",
    "train_data = pd.read_csv(\"/kaggle/input/digit-recognizer/train.csv\")\n",
    "test_data = pd.read_csv(\"/kaggle/input/digit-recognizer/test.csv\")\n",
    "X = train_data.iloc[:, train_data.columns != 'label'] \n",
    "y = train_data.iloc[:, train_data.columns == 'label']\n",
    "from sklearn.model_selection import train_test_split\n",
    "X_train, X_test, y_train, y_test = \\\n",
    "    train_test_split(X, y, stratify=y, \\\n",
    "    train_size=0.75, random_state = 0)\n",
    "X_train = pd.DataFrame(X_train)\n",
    "X_test = pd.DataFrame(X_test)\n",
    "y_test_ML = y_test\n",
    "y_train_ML = y_train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "62fca2d9",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-03-28T00:58:21.017288Z",
     "iopub.status.busy": "2023-03-28T00:58:21.015670Z",
     "iopub.status.idle": "2023-03-28T00:58:21.023776Z",
     "shell.execute_reply": "2023-03-28T00:58:21.022854Z"
    },
    "papermill": {
     "duration": 0.014657,
     "end_time": "2023-03-28T00:58:21.026004",
     "exception": false,
     "start_time": "2023-03-28T00:58:21.011347",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# preprocess the data\n",
    "X_train = X_train.values.reshape(X_train.shape[0], 28, 28, 1)\n",
    "X_test = X_test.values.reshape(X_test.shape[0], 28, 28, 1)\n",
    "y_train = to_categorical(y_train, num_classes=10)\n",
    "y_test = to_categorical(y_test, num_classes=10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "d0b9d6d2",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-03-28T00:58:21.033904Z",
     "iopub.status.busy": "2023-03-28T00:58:21.033435Z",
     "iopub.status.idle": "2023-03-28T00:58:24.979908Z",
     "shell.execute_reply": "2023-03-28T00:58:24.978426Z"
    },
    "papermill": {
     "duration": 3.954571,
     "end_time": "2023-03-28T00:58:24.983763",
     "exception": false,
     "start_time": "2023-03-28T00:58:21.029192",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# create a CNN model\n",
    "model = Sequential()\n",
    "model.add(Conv2D(32, kernel_size=(3, 3), activation=\"relu\", input_shape=(28, 28, 1)))\n",
    "model.add(MaxPooling2D(pool_size=(2, 2)))\n",
    "model.add(Conv2D(64, kernel_size=(3, 3), activation=\"relu\"))\n",
    "model.add(MaxPooling2D(pool_size=(2, 2)))\n",
    "model.add(Flatten())\n",
    "model.add(Dense(128, activation=\"relu\"))\n",
    "model.add(Dense(10, activation=\"softmax\"))\n",
    "model.compile(loss=\"categorical_crossentropy\", optimizer=\"adam\", metrics=[\"accuracy\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "13e3e983",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-03-28T00:58:24.999493Z",
     "iopub.status.busy": "2023-03-28T00:58:24.998445Z",
     "iopub.status.idle": "2023-03-28T00:58:47.533491Z",
     "shell.execute_reply": "2023-03-28T00:58:47.532398Z"
    },
    "papermill": {
     "duration": 22.544764,
     "end_time": "2023-03-28T00:58:47.535820",
     "exception": false,
     "start_time": "2023-03-28T00:58:24.991056",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/10\n",
      "197/197 [==============================] - 11s 8ms/step - loss: 2.0357 - accuracy: 0.8636 - val_loss: 0.1568 - val_accuracy: 0.9540\n",
      "Epoch 2/10\n",
      "197/197 [==============================] - 1s 7ms/step - loss: 0.1022 - accuracy: 0.9689 - val_loss: 0.1124 - val_accuracy: 0.9675\n",
      "Epoch 3/10\n",
      "197/197 [==============================] - 1s 6ms/step - loss: 0.0578 - accuracy: 0.9822 - val_loss: 0.1034 - val_accuracy: 0.9710\n",
      "Epoch 4/10\n",
      "197/197 [==============================] - 1s 6ms/step - loss: 0.0401 - accuracy: 0.9869 - val_loss: 0.0926 - val_accuracy: 0.9749\n",
      "Epoch 5/10\n",
      "197/197 [==============================] - 1s 6ms/step - loss: 0.0260 - accuracy: 0.9922 - val_loss: 0.0855 - val_accuracy: 0.9773\n",
      "Epoch 6/10\n",
      "197/197 [==============================] - 1s 6ms/step - loss: 0.0212 - accuracy: 0.9935 - val_loss: 0.0857 - val_accuracy: 0.9783\n",
      "Epoch 7/10\n",
      "197/197 [==============================] - 1s 7ms/step - loss: 0.0161 - accuracy: 0.9942 - val_loss: 0.1018 - val_accuracy: 0.9768\n",
      "Epoch 8/10\n",
      "197/197 [==============================] - 1s 6ms/step - loss: 0.0088 - accuracy: 0.9975 - val_loss: 0.0862 - val_accuracy: 0.9805\n",
      "Epoch 9/10\n",
      "197/197 [==============================] - 1s 6ms/step - loss: 0.0062 - accuracy: 0.9982 - val_loss: 0.0933 - val_accuracy: 0.9792\n",
      "Epoch 10/10\n",
      "197/197 [==============================] - 1s 7ms/step - loss: 0.0103 - accuracy: 0.9960 - val_loss: 0.1070 - val_accuracy: 0.9778\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.History at 0x7f3fd4fa43d0>"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# train the CNN model\n",
    "model.fit(X_train, y_train, batch_size=128, epochs=10, validation_split=0.2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "440ad6b7",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-03-28T00:58:47.567004Z",
     "iopub.status.busy": "2023-03-28T00:58:47.566122Z",
     "iopub.status.idle": "2023-03-28T00:58:49.092983Z",
     "shell.execute_reply": "2023-03-28T00:58:49.091842Z"
    },
    "papermill": {
     "duration": 1.54411,
     "end_time": "2023-03-28T00:58:49.095234",
     "exception": false,
     "start_time": "2023-03-28T00:58:47.551124",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "329/329 [==============================] - 1s 4ms/step - loss: 0.0900 - accuracy: 0.9824\n",
      "CNN accuracy: 0.9823809266090393\n"
     ]
    }
   ],
   "source": [
    "# evaluate the CNN model\n",
    "score = model.evaluate(X_test, y_test)\n",
    "print(\"CNN accuracy:\", score[1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "2081e308",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-03-28T00:58:49.127835Z",
     "iopub.status.busy": "2023-03-28T00:58:49.127538Z",
     "iopub.status.idle": "2023-03-28T00:58:50.290398Z",
     "shell.execute_reply": "2023-03-28T00:58:50.289310Z"
    },
    "papermill": {
     "duration": 1.181987,
     "end_time": "2023-03-28T00:58:50.292769",
     "exception": false,
     "start_time": "2023-03-28T00:58:49.110782",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "329/329 [==============================] - 1s 2ms/step\n"
     ]
    }
   ],
   "source": [
    "# predict the test set labels using CNN\n",
    "y_pred = model.predict(X_test)\n",
    "y_pred = np.argmax(y_pred, axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "9cfc567d",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-03-28T00:58:50.326922Z",
     "iopub.status.busy": "2023-03-28T00:58:50.326053Z",
     "iopub.status.idle": "2023-03-28T01:01:25.702359Z",
     "shell.execute_reply": "2023-03-28T01:01:25.701275Z"
    },
    "papermill": {
     "duration": 155.395948,
     "end_time": "2023-03-28T01:01:25.705320",
     "exception": false,
     "start_time": "2023-03-28T00:58:50.309372",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/opt/conda/lib/python3.7/site-packages/sklearn/neighbors/_classification.py:198: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().\n",
      "  return self._fit(X, y)\n",
      "/opt/conda/lib/python3.7/site-packages/sklearn/utils/validation.py:993: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
      "  y = column_or_1d(y, warn=True)\n"
     ]
    }
   ],
   "source": [
    "# compare with traditional ML methods\n",
    "X_train_ml = X_train.reshape(X_train.shape[0], 28*28)\n",
    "X_test_ml = X_test.reshape(X_test.shape[0], 28*28)\n",
    "knn = KNeighborsClassifier(n_neighbors=5)\n",
    "knn.fit(X_train_ml, y_train_ML)\n",
    "y_pred_knn = knn.predict(X_test_ml)\n",
    "svm = SVC()\n",
    "svm.fit(X_train_ml, y_train_ML)\n",
    "y_pred_svm = svm.predict(X_test_ml)\n",
    "dt = DecisionTreeClassifier()\n",
    "dt.fit(X_train_ml, y_train_ML)\n",
    "y_pred_dt = dt.predict(X_test_ml)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "085dcf77",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-03-28T01:01:25.740655Z",
     "iopub.status.busy": "2023-03-28T01:01:25.739057Z",
     "iopub.status.idle": "2023-03-28T01:01:25.750975Z",
     "shell.execute_reply": "2023-03-28T01:01:25.749316Z"
    },
    "papermill": {
     "duration": 0.031182,
     "end_time": "2023-03-28T01:01:25.752937",
     "exception": false,
     "start_time": "2023-03-28T01:01:25.721755",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "KNN accuracy: 0.9654285714285714\n",
      "SVM accuracy: 0.9736190476190476\n",
      "Decision Tree accuracy: 0.8551428571428571\n",
      "CNN accuracy: 0.9823809523809524\n"
     ]
    }
   ],
   "source": [
    "# calculate the accuracy scores\n",
    "acc_knn = accuracy_score(np.argmax(y_test, axis=1), y_pred_knn)\n",
    "acc_svm = accuracy_score(np.argmax(y_test, axis=1), y_pred_svm)\n",
    "acc_dt = accuracy_score(np.argmax(y_test, axis=1), y_pred_dt)\n",
    "acc_cnn = accuracy_score(np.argmax(y_test, axis=1), y_pred)\n",
    "print(\"KNN accuracy:\", acc_knn)\n",
    "print(\"SVM accuracy:\", acc_svm)\n",
    "print(\"Decision Tree accuracy:\", acc_dt)\n",
    "print(\"CNN accuracy:\", acc_cnn)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "af2d315f",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-03-28T01:01:25.786718Z",
     "iopub.status.busy": "2023-03-28T01:01:25.785934Z",
     "iopub.status.idle": "2023-03-28T01:01:26.564661Z",
     "shell.execute_reply": "2023-03-28T01:01:26.563346Z"
    },
    "papermill": {
     "duration": 0.798266,
     "end_time": "2023-03-28T01:01:26.567224",
     "exception": false,
     "start_time": "2023-03-28T01:01:25.768958",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "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vXtdxvTo0UMhISHatWuXmjdvrtDQUOtL8JtvvlHr1q0VEREhp9OpyMhItWrVSocPH870+bp8+bImTJigihUravjw4RmOKVasmBo2bGjd//PPP9WvXz+VKFFCAQEBKlu2rIYNG6akpCSX+RwOhx577DHNmzdPlSpVUr58+VSjRg19/PHHLtuTtuxOnTrJ4XDo7rvvlvT3eybt31c/B1fvxjt16lTVqFFDISEhCg0NVcWKFfXss89aj2d2GNqyZcsUExOjfPnyKTQ0VM2aNUv3S2Paruq7d+/WQw89pPDwcBUtWlRxcXE6e/Zshs9ZRuLi4vTll1/qxx9/tKatXbtWBw4cUM+ePdONP3HihPr166fKlSsrJCREERERatKkiTZt2mSN2b9/v9WMjR492vq76tGjh0vtO3bsUPv27VWgQAGrWbl6F/wvvvhC/v7+evLJJ13qSNuV/K233srytgL45zh16pQKFy6c4Y9LPj6uLXdmh9uWLl3a+tyS/u9zZ82aNdf9/rz77rtVtWpVbdq0SXfccYeCgoKs7/CUlBSXse5+f02bNk2VKlWS0+nU3Llzr/l5m5FevXrp0qVLWrBgQbrHZs+eLenv7wZJunTpkp5//nlVrFjR6qN69uypEydOpJt3wYIFiomJUUhIiEJCQlSzZk3rM/p6/dbNPgfXExcXpyVLlrgc1pR2moXOnTunG5/VXlWSzpw5oyeeeEJly5aV0+lURESEWrZsqR9++CHd2EmTJqlMmTIKCQlRTEyMtmzZ4vJ4RoehpZ1mICv9XVb7zYzmmzlzpu699950QVGaW2+9VdWrV7fuHzx4UN26dbP6u0qVKunll192OVTtykPwrrXtd999t7p16yZJuv32213ew1f/HV45z5X9WGpqqp5//nlVqFBBQUFByp8/v6pXr65XX33VGpPZYWju9Lo///yzWrZsqZCQEEVFRemJJ55I9z69lux4L27YsEG33XabJKlnz57W31fa59q1+vSr+9eFCxfK4XDo9ddfd6lj5MiR8vX1zfQUJsgBBvCg2bNnG0nmlVdeMeHh4aZq1armyJEjWZr3t99+M5LMiy++aHbu3GkcDocZPny49fjIkSONJHPixAlr2rx584zD4TDt2rUzS5YsMcuXLzetW7c2vr6+Zu3atda4jRs3mieeeMK8//77ZuPGjebDDz807dq1M0FBQeaHH36wxq1fv95IMiVKlDDt27c3y5YtMx9//LE5depUltd17733miJFipjp06ebDRs2mKVLl5oRI0aYhQsXGmOM2b17t2nQoIEpVqyY2bx5s3W7lkaNGpkqVaqY1NRUU6dOHVOuXDlz6dKldM9bmqNHj5qoqCgTHR1t3nzzTbN27VozduxY43Q6TY8ePVyWLcmMHDnSuv/tt9+awMBAU716dbNw4UKzbNky07JlS1O6dGkjyfz222/W2OjoaFOyZElTuXJl8/bbb5tVq1aZDh06GElm48aN1ri090VUVJSJi4szK1asMNOnTzcREREmKirKnD592ho7fvx4I8k89NBD5pNPPjFvv/22KVu2rAkPDzf79u2zxsXGxhp/f39TunRpEx8fbz777DOzatUq89dff5lChQqZunXrmsWLF5uNGzeaRYsWmT59+pg9e/Zk+hx/+eWXRpIZMmTINV+LNBcuXDDVq1c3wcHB5qWXXjKrV682w4cPN35+fqZly5bpnuPSpUubevXqmcWLF5tPP/3U3H333cbPz8/88ssvxhhjfv75Z/Pf//7XSDLjx483mzdvNrt37zbG/P36N2rUKF0NsbGxJjo62rr/7rvvGknm8ccfN6tXrzZr164106ZNMwMGDLDGpL3H169fb0175513jCTTvHlzs3TpUrNo0SJTp04dExAQYDZt2mSNS/sbrFChghkxYoRZs2aNmTRpknE6naZnz57Xfc5iY2NNcHCwSU1NNdHR0ebpp5+2HuvUqZO56667zIkTJ9K9J3/44QfTt29fs3DhQrNhwwbz8ccfm169ehkfHx9rOy5evGhWrlxpJJlevXpZf1c///yzS+3R0dFmyJAhZs2aNWbp0qUuj13phRdeMJLMRx99ZIwx5vvvvzf58uUz3bp1u+52Avhn6t27t/UZvGXLFut7OiNXf86liY6ONrGxsdZ9d74/GzVqZAoVKmQiIyPNa6+9ZlatWmUGDBhgJJn+/ftb49z9/ipRooSpXr26WbBggVm3bp3ZuXPnNT9vM5KSkmKio6NNzZo1XaYnJyeb4sWLmzvuuMMad99995ng4GAzevRos2bNGjNz5kxTokQJU7lyZXP+/Hlr3uHDhxtJ5sEHHzTvvfeeWb16tZk0aZLVO16r37rZ5+D777/PdFvTeraEhAQTHBxs3njjDeux22+/3Tz88MNm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2/X222+rfPny+uOPP/T888+rfv362r17twoVKmR3eW779ddfNXXqVA0ePFjPPvusvv76aw0YMEBOp1MPP/yw3eXdsKVLl+rMmTPq0aOH3aXcsCFDhujs2bOqWLGifH19lZKSonHjxumhhx6yuzS3hYaGKiYmRmPHjlWlSpVUtGhRvfvuu/rqq69066232l0ecgD9U97gDf0TvVPuRP+UN+T1/smbeicp9/ZPhEXXcXXSaozJk+mrt3rsscf03Xff5flfrCtUqKCdO3fqzJkz+uCDDxQbG6uNGzfmuabn0KFDGjhwoFavXp0uGc9rWrRoYf27WrVqiomJUbly5TR37lwNHjzYxspuTGpqqurWravx48dLkmrVqqXdu3dr6tSpebrZeeutt9SiRQtFRkbaXcoNW7RokebPn68FCxaoSpUq2rlzpwYNGqTIyEjFxsbaXZ7b5s2bp7i4OJUoUUK+vr6qXbu2unTpoh07dthdGnIQ/VPu5g39E71T7kT/lDfk9f7J23onKXf2T4RFmShcuLB8fX3T/Qp2/PjxdL+WwR6PP/64li1bps8//1wlS5a0u5ybEhAQoFtuuUWSVLduXW3dulWvvvqq3nzzTZsrc8/27dt1/Phx1alTx5qWkpKizz//XK+//rqSkpLk6+trY4U3Ljg4WNWqVdNPP/1kdyk3pHjx4uka6EqVKumDDz6wqaKbd+DAAa1du1ZLliyxu5Sb8tRTT+mZZ55R586dJf3dXB84cEDx8fF5suEpV66cNm7cqMTERCUkJKh48eLq1KmTypQpY3dpyAH0T7mft/RP9E55A/1T7uMN/ZO39U5S7uyfOGdRJgICAlSnTh3rLPFp1qxZo/r169tUFaS/f5187LHHtGTJEq1bt84r/wNijFFSUpLdZbitadOm2rVrl3bu3Gnd6tatq65du2rnzp15utlJSkrS3r17Vbx4cbtLuSENGjRId4nkffv2KTo62qaKbt7s2bMVERGhVq1a2V3KTTl//rx8fFy/jn19ffP05V+lv/+DULx4cZ0+fVqrVq1S27Zt7S4JOYD+Kffy9v6J3il3on/Kfbyhf/LW3knKXf0TexZdw+DBg9W9e3fVrVtXMTExmj59ug4ePKg+ffrYXZrb/vrrL/3888/W/d9++007d+5UwYIFVapUKRsrc1///v21YMECffTRRwoNDbV+vQwPD1dQUJDN1bnv2WefVYsWLRQVFaVz585p4cKF2rBhg1auXGl3aW4LDQ1Nd+6D4OBgFSpUKM+dE+HJJ59UmzZtVKpUKR0/flzPP/+8EhIS8uyvFf/5z39Uv359jR8/Xh07dtTXX3+t6dOna/r06XaXdkNSU1M1e/ZsxcbGys8vb3+VtWnTRuPGjVOpUqVUpUoVffPNN5o0aZLi4uLsLu2GrFq1SsYYVahQQT///LOeeuopVahQQT179rS7NOQQ+qfcyZv6J3qn3Iv+KXfzlv7J23onKZf2T3Zdhi2v+O9//2uio6NNQECAqV27dp69vOj69euNpHS32NhYu0tzW0bbIcnMnj3b7tJuSFxcnPUeK1KkiGnatKlZvXq13WV5TF69/GunTp1M8eLFjb+/v4mMjDQPPvig2b17t91l3ZTly5ebqlWrGqfTaSpWrGimT59ud0k3bNWqVUaS+fHHH+0u5aYlJCSYgQMHmlKlSpnAwEBTtmxZM2zYMJOUlGR3aTdk0aJFpmzZsiYgIMAUK1bM9O/f35w5c8buspDD6J9yH2/qn+idci/6p9zNW/onb+udjMmd/ZPDGGNyLpoCAAAAAABAbsY5iwAAAAAAAGAhLAIAAAAAAICFsAgAAAAAAAAWwiIAAAAAAABYCIsAAAAAAABgISwCAAAAAACAhbAIAAAAAAAAFsIiAAAAAAAAWAiLAGSrUaNGqWbNmtb9Hj16qF27djlex/79++VwOLRz585Mx5QuXVqTJ0/O8jLnzJmj/Pnz33RtDodDS5cuvenlAAAA70D/dH30T0D2IiwC/oF69Oghh8Mhh8Mhf39/lS1bVk8++aQSExOzfd2vvvqq5syZk6WxWWlQAAAAcgL9E4B/Ej+7CwBgj/vuu0+zZ8/W5cuXtWnTJvXu3VuJiYmaOnVqurGXL1+Wv7+/R9YbHh7ukeUAAADkNPonAP8U7FkE/EM5nU4VK1ZMUVFR6tKli7p27Wrtypu26/OsWbNUtmxZOZ1OGWN09uxZ/fvf/1ZERITCwsLUpEkTffvtty7LfeGFF1S0aFGFhoaqV69eunjxosvjV+9GnZqaqgkTJuiWW26R0+lUqVKlNG7cOElSmTJlJEm1atWSw+HQ3Xffbc03e/ZsVapUSYGBgapYsaLeeOMNl/V8/fXXqlWrlgIDA1W3bl198803bj9HkyZNUrVq1RQcHKyoqCj169dPf/31V7pxS5cuVfny5RUYGKhmzZrp0KFDLo8vX75cderUUWBgoMqWLavRo0crOTnZ7XoAAIC96J+uj/4J8A6ERQAkSUFBQbp8+bJ1/+eff9bixYv1wQcfWLsxt2rVSseOHdOnn36q7du3q3bt2mratKn+/PNPSdLixYs1cuRIjRs3Ttu2bVPx4sXTNSFXGzp0qCZMmKDhw4drz549WrBggYoWLSrp74ZFktauXaujR49qyZIlkqQZM2Zo2LBhGjdunPbu3avx48dr+PDhmjt3riQpMTFRrVu3VoUKFbR9+3aNGjVKTz75pNvPiY+Pj1577TV9//33mjt3rtatW6enn37aZcz58+c1btw4zZ07V//73/+UkJCgzp07W4+vWrVK3bp104ABA7Rnzx69+eabmjNnjtXQAQCAvIv+KT36J8BLGAD/OLGxsaZt27bW/a+++soUKlTIdOzY0RhjzMiRI42/v785fvy4Neazzz4zYWFh5uLFiy7LKleunHnzzTeNMcbExMSYPn36uDx+++23mxo1amS47oSEBON0Os2MGTMyrPO3334zksw333zjMj0qKsosWLDAZdrYsWNNTEyMMcaYN9980xQsWNAkJiZaj0+dOjXDZV0pOjravPLKK5k+vnjxYlOoUCHr/uzZs40ks2XLFmva3r17jSTz1VdfGWOMufPOO8348eNdljNv3jxTvHhx674k8+GHH2a6XgAAYD/6p4zRPwHeiXMWAf9QH3/8sUJCQpScnKzLly+rbdu2mjJlivV4dHS0ihQpYt3fvn27/vrrLxUqVMhlORcuXNAvv/wiSdq7d6/69Onj8nhMTIzWr1+fYQ179+5VUlKSmjZtmuW6T5w4oUOHDqlXr1565JFHrOnJycnW8fx79+5VjRo1lC9fPpc63LV+/XqNHz9ee/bsUUJCgpKTk3Xx4kUlJiYqODhYkuTn56e6deta81SsWFH58+fX3r17Va9ePW3fvl1bt251+SUsJSVFFy9e1Pnz511qBAAAuRv90/XRPwHegbAI+Idq3Lixpk6dKn9/f0VGRqY7AWPal3ma1NRUFS9eXBs2bEi3rBu9/GlQUJDb86Smpkr6e1fq22+/3eUxX19fSZIx5obqudKBAwfUsmVL9enTR2PHjlXBggX1xRdfqFevXi67m0t/X7r1amnTUlNTNXr0aD344IPpxgQGBt50nQAAIOfQP10b/RPgPQiLgH+o4OBg3XLLLVkeX7t2bR07dkx+fn4qXbp0hmMqVaqkLVu26OGHH7ambdmyJdNl3nrrrQoKCtJnn32m3r17p3s8ICBA0t+/JKUpWrSoSpQooV9//VVdu3bNcLmVK1fWvHnzdOHCBauhulYdGdm2bZuSk5P18ssvy8fn79O7LV68ON245ORkbdu2TfXq1ZMk/fjjjzpz5owqVqwo6e/n7ccff3TruQYAALkT/dO10T8B3oOwCECW3HPPPYqJiVG7du00YcIEVahQQUeOHNGnn36qdu3aqW7duho4cKBiY2NVt25dNWzYUO+88452796tsmXLZrjMwMBADRkyRE8//bQCAgLUoEEDnThxQrt371avXr0UERGhoKAgrVy5UiVLllRgYKDCw8M1atQoDRgwQGFhYWrRooWSkpK0bds2nT59WoMHD1aXLl00bNgw9erVS88995z279+vl156ya3tLVeunJKTkzVlyhS1adNG//vf/zRt2rR04/z9/fX444/rtddek7+/vx577DHdcccdVvMzYsQItW7dWlFRUerQoYN8fHz03XffadeuXXr++efdfyEAAECeQf9E/wTkVVwNDUCWOBwOffrpp7rrrrsUFxen8uXLq3Pnztq/f7919Y1OnTppxIgRGjJkiOrUqaMDBw6ob9++11zu8OHD9cQTT2jEiBGqVKmSOnXqpOPHj0v6+3j21157TW+++aYiIyPVtm1bSVLv3r01c+ZMzZkzR9WqVVOjRo00Z84c61KxISEhWr58ufbs2aNatWpp2LBhmjBhglvbW7NmTU2aNEkTJkxQ1apV9c477yg+Pj7duHz58mnIkCHq0qWLYmJiFBQUpIULF1qP33vvvfr444+1Zs0a3Xbbbbrjjjs0adIkRUdHu1UPAADIe+if6J+AvMphPHFwKgAAAAAAALwCexYBAAAAAADAQlgEAAAAAAAAC2ERAAAAAAAALIRFAAAAAAAAsBAWAQAAAAAAwEJYBAAAAAAAAAthEQAAAAAAACyERQAAAAAAALAQFgEAAAAAAMBCWAQAAAAAAAALYREAAAAAAAAshEUAAAAAAACwEBYBAAAAAADAQlgEAAAAAAAAC2ERAAAAAAAALIRFAAAAAAAAsBAWAQAAAAAAwEJYBK8wZ84cORwO6xYYGKhixYqp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dfr06UzX2b17dy1dulQjRozQ6tWrNXPmTN1zzz06depUpvMYY9SuXTu99NJL6t69uz755BMNHjxYc+fOVZMmTaywJs23336rJ554Qv/5z3/00UcfqXr16urVq5fLdt6MuLg4+fv7a968eXr//ffl7++v8ePH66GHHlLlypW1ePFizZs3T+fOndOdd96pPXv2WPNm9fnNyOrVqyUpy7sFb9y4UU2aNNHZs2f11ltv6d1331VoaKjatGmTYdPYu3dv+fv7a8GCBZo4caI2bNigbt26WY9v3rxZLVu2VFBQkDZv3qzNmzerVatWWaolTV55/Xv16qUtW7Zo7969kqQzZ85oyZIl6tWrV4bjf/nlF8XExGjq1KlavXq1RowYoa+++koNGzbU5cuXJf39/D7++OOSpCVLlljPYe3ata3l7NixQ0899ZQGDBiglStX6l//+leG65s8ebIqVaqkjh07Wg316NGjtWHDBs2fP5/dwAHAi6SkpGjdunWqU6eOoqKi3Jq3U6dOqlOnjp577jnr++hafH19FR8fr927d2vu3Llurevo0aP6/vvv1bx5c5cfrjLj6e/33r17a8mSJZKkxx9/PMPD8q5n8+bN6tSpk8qWLauFCxfqk08+0YgRI5ScnHzN+Tzdc13PrbfeqjvvvNMlLJs1a5ZKly6tpk2bphuflJSkP//8U08++aSWLl2qd999Vw0bNtSDDz6ot99+22X7g4KC1LJlS6tPeeONN1yWlVEfnFF9M2bM0Pvvv6/XXntNknTs2DF16dJFd955p8sPiUCeY/OeTYBHXOswtDRFixY1lSpVsu5XrFjR1KpVy1y+fNllXOvWrU3x4sWt3V3j4uKMv7+/2bNnT6bLTtvNefbs2da0kJAQM2jQoGvWHRsba6Kjo637K1euNJLMxIkTXcYtWrTISHI5XCg6OtoEBgaaAwcOWNMuXLhgChYsaB599NFrrvdKGR2GlvZ8Pvzwwy5jDx48aPz8/Mzjjz/uMv3cuXOmWLFipmPHjta0rD6/GUnbNTyj3c8zcscdd5iIiAhz7tw5a1pycrKpWrWqKVmypLXbcNp29evXz2X+iRMnGknm6NGj1rSMdkPP6HVOI8mMHDnSup/bX39Jpn///iY1NdWUKVPGPPnkk8YYY/773/+akJAQc+7cueseSpaammouX75sDhw4YCS5HBZwrXmj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53E6nWrUqJEmTJggSRle4VZSjvZcV4uJiVFcXJweeOABPfDAA5mOczgcCggIcAl6jh07lmGv5qm9tRITE9WhQweVLl1a69ev12OPPaZnnnlGX3311U0vG7ATexbBq3z//ffWuXGOHz+uTZs2afbs2fL19dWHH37o8mvOm2++qRYtWujee+9Vjx49VKJECf3555/au3evduzYoffee0/S35dGXbFihe666y49++yzqlatms6cOaOVK1dq8ODBqlixYro6zp49q8aNG6tLly6qWLGiQkNDtXXrVq1cuTLTPW0kqVmzZrr33ns1ZMgQJSQkqEGDBvruu+80cuRI1apVS927d/f8k+aG0qVLa8yYMRo2bJh+/fVX3XfffSpQoID++OMPff311woODrYas6w+vxlJe72aN2+umJgY9e3bV40bN1ZwcLAOHDig999/X8uXL7d2F4+Pj1ezZs3UuHFjPfnkkwoICNAbb7yh77//Xu+++67be7FkxuFwqFu3bpo1a5bKlSunGjVq6Ouvv9aCBQtcxuXF17958+Yul/jNSMWKFVWuXDk988wzMsaoYMGCWr58ucvlbNNUq1ZNkvTqq68qNjZW/v7+qlChwjVDwozs2rVLAwYMUGxsrBUQvfXWW2rfvr0mT56sQYMGubU8AEDulnbFzX79+qlOnTrq27evqlSposuXL+ubb77R9OnTVbVqVbVp0ybTZZQpU0Z9+/Z16zxEo0aN0jvvvKP169crODj4uuM7dOig4cOHa+zYsfrhhx/Uq1cvlStXTufPn9dXX32lN998U506dVLz5s1z9Pu9U6dOGjFihDp37qynnnpKFy9e1GuvvaaUlBSXcSNGjNDhw4fVtGlTlSxZUmfOnNGrr74qf39/NWrUKNPl51TPlZG33nrrumNat26tJUuWqF+/fmrfvr0OHTqksWPHqnjx4vrpp59cxlarVk0bNmzQ8uXLVbx4cYWGhl5zz/fM9OnTRwcPHrR64ZdfflmbN29W586d9c033yh//vxuLxPIFWw7tTbgQWlXJki7BQQEmIiICNOoUSMzfvx4c/z48Qzn+/bbb03Hjh1NRESE8ff3N8WKFTNNmjQx06ZNcxl36NAhExcXZ4oVK2b8/f1NZGSk6dixo/njjz+MMemvFnHx4kXTp08fU716dRMWFmaCgoJMhQoVzMiRI01iYqK13Iyu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lClT4rDDDotDDjnkHa/b0tISzc3NHU4AAOVOfwIAKslmvWbRW912221x4403xgc/+MEoFArp8t133z3+/Oc/F3VbN9xwQzz88MOxaNGizbp+Y2NjnHfeeUXdBwBA3vQnAKCSFP3IohdffDEGDBiw0eXr16/vUH7eyerVq2Pq1Klx7bXXdvgN278yY8aMaGpqSqfVq1dv9v0BAORFfwIAKknRY9F+++0X/+f//J90/s2Cc9VVV8W4ceM2+3aWLFkSa9asiTFjxkRtbW3U1tbGggUL4pJLLona2tpoa2vb6HPq6uqivr6+wwkAoNzpTwBAJSn6aWiNjY3xb//2b/Hoo49Ga2trfPe7340VK1bE73//+1iwYMFm387BBx8cy5cv73DZ8ccfHyNHjowzzzwzampqio0GAFCW9CcAoJIUPRYdcMAB8dvf/ja+/e1vx/Dhw+OXv/xl7LvvvvH73/8+9txzz82+nT59+sQee+zR4bLevXtH//79N7ocAKCS6U8AQCUpeiyKiNhzzz1j3rx5pc4CAFC19CcAoFJs0VjU1tYWt956a6xcuTIKhULstttuMXHixKit3aKbS+67776t+nwAgHKlPwEAlaLodvLHP/4xJk6cGM8//3yMGDEiIiIef/zx2H777eP2228v6qHUAADvBvoTAFBJin43tBNPPDFGjRoVzzzzTDz88MPx8MMPx+rVq2OvvfaKf//3f++MjAAAFU1/AgAqSdGPLHrkkUdi8eLFsd1226XLtttuu5g5c2bst99+JQ0HAFAN9CcAoJIU/ciiESNGxAsvvLDR5WvWrImdd965JKEAAKqJ/gQAVJLNGouam5vT6cILL4zTTjstfvrTn8YzzzwTzzzzTPz0pz+NadOmxaxZszo7LwBARdCfAIBKtVlPQ9t2222jUCik81mWxVFHHZUuy7IsIiIOP/zwaGtr64SYAACVRX8CACrVZo1Fv/71rzs7BwBAVdGfAIBKtVlj0Uc+8pHOzgEAUFX0JwCgUhX9bmhvevXVV2PVqlXx2muvdbh8r7322upQAADVSH8CACpB0WPRiy++GMcff3zceeedm/y459wDAHSkPwEAlWSz3g3traZNmxZr166NhQsXRq9eveKuu+6KefPmxS677BK33357Z2QEAKho+hMAUEmKfmTRr371q/jZz34W++23X3Tr1i2GDBkSn/jE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lStXxn/+53/G//7f/zsuvfTSvKNtkRNPPDHmz58f11xzTSxfvjzGjx8fhxxySDz77LN5R6OL6E/lqZr6k+5U3vSn8lRN/anaulNEmfanjLf1gQ98IDvppJM6XDZy5Mjsa1/7Wk6JSiMisltvvTXvGCWzZs2aLCKyBQsW5B2lZLbbbrvsBz/4Qd4xtti6deuyXXbZJZs/f372kY98JJs6dWrekYp2zjnnZHvvvXfeMUrmzDPPzA466KC8Y3SKqVOnZsOHD8/a29vzjrJFDjvssGzy5MkdLvv0pz+dfe5zn8sp0ZZ79dVXs5qamuwXv/hFh8v33nvv7KyzzsopFV1Nf6oM1dafdKfyoD9VjkruT9XUnbKsfPuTRxa9jddeey2WLFkS48eP73D5+PHj43e/+11OqdiUpqamiIjo169fzkm2XltbW9xwww2xfv36GDduXN5xttiUKVPisMMOi0MOOSTvKFvlT3/6UwwePDiGDh0axxxzTDz55JN5R9pit99+e4wdOzaOPPLIGDBgQIwePTquuuqqvGNttddeey2uvfbamDx5chQKhbzjbJGDDjoo7r333nj88ccjIuKRRx6JBx54IA499NCckxWvtbU12traomfPnh0u79WrVzzwwAM5paIr6U+Vo1r6k+5UfvSn8lfp/amaulNE+fan2tzuucy99NJL0dbWFgMHDuxw+cCBA+P555/PKRX/LMuymD59ehx00EGxxx575B1niy1fvjzGjRsXGzZsiPe85z1x6623xu677553rC1yww03xMMPP5z7c2y31v777x8/+tGPYtddd40XXnghLrjggjjggANixYoV0b9//7zjFe3JJ5+MOXPmxPTp0+PrX/96PPTQQ3HaaadFXV1dfOELX8g73ha77bbb4uWXX44vfvGLeUfZYmeeeWY0NTXFyJEjo6amJtra2mLmzJnx2c9+Nu9oRevTp0+MGzcuzj///Nhtt91i4MCBcf3118eDDz4Yu+yyS97x6AL6U2Wohv6kO5Un/akyVHp/qqbuFFG+/clY9A7+eWnNsqwi19dqdcopp8Qf/vCHiv+N9YgRI2LZsmXx8ssvx8033xyTJk2KBQsWVFzpWb16dUydOjV++ctfbrSMV5oJEyak/7/nnnvGuHHjYvjw4TFv3ryYPn16jsm2THt7e4wdOzYuvPDCiIgYPXp0rFixIubMmVPRZeeHP/xhTJgwIQYPHpx3lC124403xrXXXhvXXXddjBo1KpYtWxbTpk2LwYMHx6RJk/KOV7RrrrkmJk+eHO9///ujpqYm9t133zj22GPj4YcfzjsaXUh/Km/V0J90p/KkP1WGSu9P1dadIsqzPxmL3sZ73/veqKmp2ei3YGvWrNnot2Xk49RTT43bb7897r///thhhx3yjrNVevToETvvvHNERIwdOzYWLVoU3/3ud+OKK67IOVlxlixZEmvWrIkxY8aky9ra2uL++++Pyy67LFpaWqKmpibHhFuud+/eseeee8af/vSnvKNskUGDBm1UoHfbbbe4+eabc0q09Z5++um455574pZbbsk7ylY544wz4mtf+1occ8wxEfFGuX766aejsbGxIgvP8OHDY8GCBbF+/fpobm6OQYMGxdFHHx1Dhw7NOxpdQH8qf9XSn3SnyqA/lZ9q6E/V1p0iyrM/ec2it9GjR48YM2ZMepX4N82fPz8OOOCAnFIR8cZvJ0855ZS45ZZb4le/+lVV/gMky7JoaWnJO0bRDj744Fi+fHksW7YsncaOHRvHHXdcLFu2rKLLTktLS6xcuTIGDRqUd5QtcuCBB270FsmPP/54DBkyJKdEW2/u3LkxYMCAOOyww/KOslVeffXV6Nat41/HNTU1Ff32rxFv/ANh0KBBsXbt2rj77rtj4sSJeUeiC+hP5ava+5PuVJ70p/JTDf2pWrtTRHn1J48s+hemT58en//852Ps2LExbty4uPLKK2PVqlVx0kkn5R2taK+88ko88cQT6fxTTz0Vy5Yti379+sWOO+6YY7LiTZkyJa677rr42c9+Fn369Em/vezbt2/06tUr53TF+/rXvx4TJkyIhoaGWLduXdxwww1x3333xV133ZV3tKL16dNno9c+6N27d/Tv37/iXhPh9NNPj8MPPzx23HHHWLNmTVxwwQXR3Nxcsb+t+I//+I844IAD4sILL4yjjjoqHnroobjyyivjyiuvzDvaFmlvb4+5c+fGpEmTora2sv8qO/zww2PmzJmx4447xqhRo2Lp0qVx8cUXx+TJk/OOtkXuvvvuyLIsRowYEU888UScccYZMWLEiDj++OPzjkYX0Z/KUzX1J92pfOlP5a1a+lO1daeIMu1Peb0NW6X43ve+lw0ZMiTr0aNHtu+++1bs24v++te/ziJio9OkSZPyjla0TR1HRGRz587NO9oWmTx5cvoe23777bODDz44++Uvf5l3rJKp1Ld/Pfroo7NBgwZl3bt3zwYPHpx9+tOfzlasWJF3rK3y85//PNtjjz2yurq6bOTIkdmVV16Zd6Qtdvfdd2cRkT322GN5R9lqzc3N2dSpU7Mdd9wx69mzZzZs2LDsrLPOylpaWvKOtkVuvPHGbNiwYVmPHj2y973vfdmUKVOyl19+Oe9YdDH9qfxUU3/SncqX/lTeqqU/VVt3yrLy7E+FLMuyrpumAAAAAChnXrMIAAAAgMRYBAAAAEBiLAIAAAAgMRYBAAAAkBiLAAAAAEiMRQAAAAAkxiIAAAAAEmMRAAAAAImxCOhU5557buyzzz7p/Be/+MU44ogjujzHX/7ylygUCrFs2bK3vc5OO+0Us2fP3uzbvPrqq2Pbbbfd6myFQiFuu+22rb4dAKA66E/vTH+CzmUsgnehL37xi1EoFKJQKET37t1j2LBhcfrpp8f69es7/b6/+93vxtVXX71Z192cggIA0BX0J+DdpDbvAEA+/u3f/i3mzp0br7/+evzmN7+JE088MdavXx9z5szZ6Lqvv/56dO/evST327dv35LcDgBAV9OfgHcLjyyCd6m6urp43/veFw0NDXHsscfGcccdlx7K++ZDn//7v/87hg0bFnV1dZFlWTQ1NcW///u/x4ABA6K+vj4+/vGPxyOPPNLhdi+66KIYOHBg9OnTJ0444YTYsGFDh4//88Oo29vbY9asWbHzzjtHXV1d7LjjjjFz5syIiBg6dGhERIwePToKhUJ89KMfTZ83d+7c2G233aJnz54xcuTIuPzyyzvcz0MPPRSjR4+Onj17xtixY2Pp0qVF/xldfPHFseeee0bv3r2joaEhTj755HjllVc2ut5tt90Wu+66a/Ts2TM+8YlPxOrVqzt8/Oc//3mMGTMmevbsGcOGDYvzzjsvWltbi84DAORLf3pn+hNUB2MREBERvXr1itdffz2df+KJJ+Kmm26Km2++OT2M+bDDDovnn38+7rjjjliyZEnsu+++cfDBB8ff//73iIi46aab4pxzzomZM2fG4sWLY9CgQRuVkH82Y8aMmDVrVpx99tnx6KOPxnXXXRcDBw6MiDcKS0TEPffcE88991zccsstERFx1VVXxVlnnRUzZ86MlStXxoUXXhhnn312zJs3LyIi1q9fH5/61KdixIgRsWTJkjj33HPj9NNPL/rPpFu3bnHJJZfEH//4x5g3b1786le/iq9+9asdrvPqq6/GzJkzY968efHb3/42mpub45hjjkkfv/vuu+Nzn/tcnHbaafHoo4/GFVdcEVdffXUqdABA5dKfNqY/QZXIgHedSZMmZRMnTkznH3zwwax///7ZUUcdlWVZlp1zzjlZ9+7dszVr1qTr3HvvvVl9fX22YcOGDrc1fPjw7IorrsiyLMvGjRuXnXTSSR0+vv/++2d77733Ju+7ubk5q6ury6666qpN5nzqqaeyiMiWLl3a4fKGhobsuuuu63DZ+eefn40bNy7Lsiy74oorsn79+mXr169PH58zZ84mb+uthgwZkv3Xf/3X2378pptuyvr375/Oz507N4uIbOHChemylStXZhGRPfjgg1mWZdmHPvSh7MILL+xwO9dcc002aNCgdD4isltvvfVt7xcAyJ/+tGn6E1Qnr1kE71K/+MUv4j3veU+0trbG66+/HhMnToxLL700fXzIkCGx/fbbp/NLliyJV155Jfr379/hdv7xj3/En//854iIWLlyZZx00kkdPj5u3Lj49a9/vckMK1eujJaWljj44IM3O/eLL74Yq1evjhNOOCG+9KUvpctbW1vT8/lXrlwZe++9d2yzzTYdchTr17/+dVx44YXx6KOPRnNzc7S2tsaGDRti/fr10bt374iIqK2tjbFjx6bPGTlyZGy77baxcuXK+MAHPhBLliyJRYsWdfhNWFtbW2zYsCFeffXVDhkBgPKmP70z/Qmqg7EI3qU+9rGPxZw5c6J79+4xePDgjV6A8c2/zN/U3t4egwYNivvuu2+j29rStz/t1atX0Z/T3t4eEW88lHr//ffv8LGampqIiMiybIvyvNXTTz8dhx56aJx00klx/vnnR79+/eKBBx6IE044ocPDzSPeeOvWf/bmZe3t7XHeeefFpz/96Y2u07Nnz63OCQB0Hf3pX9OfoHoYi+Bdqnfv3rHzzjtv9vX33XffeP7556O2tjZ22mmnTV5nt912i4ULF8YXvvCFdNnChQvf9jZ32WWX6NWrV9x7771x4oknbvTxHj16RMQbv0l608CBA+P9739/PPnkk3Hcccdt8nZ33333uOaaa+If//hHKlT/KsemLF68OFpbW+M73/lOdOv2xsu73XTTTRtdr7W1NRYvXhwf+MAHIiLisccei5dffjlGjhwZEW/8uT322GNF/VkDAOVJf/rX9CeoHsYiYLMccsghMW7cuDjiiCNi1qxZMWLEiPjrX/8ad9xxRxxxxBExduzYmDp1akyaNCnGjh0bBx10UPz4xz+OFStWxLBhwzZ5mz179owzzzwzvvrVr0aPHj3iwAMPjBdffDFWrFgRJ5xwQgwYMCB69eoVd911V+ywww7Rs2fP6Nu3b5x77rlx2mmnRX19fUyYMCFaWlpi8eLFsXbt2pg+fXoce+yxcdZZZ8UJJ5wQ3/jGN+Ivf/lLfPvb3y7qeIcPHx6tra1x6aWXxuGHHx6//e1v4/vf//5G1+vevXuceuqpcckll0T37t3jlFNOiQ9+8IOp/Hzzm9+MT33qU9HQ0BBHHnlkdOvWLf7whz/E8uXL44ILLij+CwEAVAz9SX+CSuXd0IDNUigU4o477ogPf/jDMXny5Nh1113jmGOOib/85S/p3TeOPvro+OY3vxlnnnlmjBkzJp5++un48pe//C9v9+yzz46vfOUr8c1vfjN22223OProo2PNmjUR8cbz2S+55JK44oorYvDgwTFx4sSIiDjxxBPjBz/4QVx99dWx5557xkc+8pG4+uqr01vFvuc974mf//zn8eijj8bo0aPjrLPOilmzZhV1vPvss09cfPHFMWvWrNhjjz3ixz/+cTQ2Nm50vW222SbOPPPMOPbYY2PcuHHRq1evuOGGG9LHP/nJT8YvfvGLmD9/fuy3337xwQ9+MC6++OIYMmRIUXkAgMqjP+lPUKkKWSmenAoAAABAVfDIIgAAAAASYxEAAAAAibEIAAAAgMRYBAAAAEBiLAIAAAAgMRYBAAAAkBiLAAAAAEiMRQAAAAAkxiIAAAAAEmMRAAAAAImxCAAAAIDEWAQAAABA8n8BBKMT3GLOLOkAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 1200x1200 with 4 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# plot the confusion matrices\n",
    "fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))\n",
    "axes[0, 0].set_title(\"K-Nearest Neighbors Confusion Matrix\")\n",
    "axes[0, 1].set_title(\"Support Vector Machine Confusion Matrix\")\n",
    "axes[1, 0].set_title(\"Decision Tree Confusion Matrix\")\n",
    "axes[1, 1].set_title(\"CNN Confusion Matrix\")\n",
    "confmat_knn = confusion_matrix(np.argmax(y_test, axis=1), y_pred_knn)\n",
    "confmat_svm = confusion_matrix(np.argmax(y_test, axis=1), y_pred_svm)\n",
    "confmat_dt = confusion_matrix(np.argmax(y_test, axis=1), y_pred_dt)\n",
    "confmat_cnn = confusion_matrix(np.argmax(y_test, axis=1), y_pred)\n",
    "axes[0, 0].imshow(confmat_knn, cmap=plt.cm.Blues)\n",
    "axes[0, 1].imshow(confmat_svm, cmap=plt.cm.Blues)\n",
    "axes[1, 0].imshow(confmat_dt, cmap=plt.cm.Blues)\n",
    "axes[1, 1].imshow(confmat_cnn, cmap=plt.cm.Blues)\n",
    "for i in range(2):\n",
    "    for j in range(2):\n",
    "        axes[i, j].set_xticks(range(10))\n",
    "        axes[i, j].set_yticks(range(10))\n",
    "        axes[i, j].set_xlabel(\"Predicted label\")\n",
    "        axes[i, j].set_ylabel(\"True label\")\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "2f6b9548",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-03-28T01:01:26.606171Z",
     "iopub.status.busy": "2023-03-28T01:01:26.605362Z",
     "iopub.status.idle": "2023-03-28T01:01:29.643106Z",
     "shell.execute_reply": "2023-03-28T01:01:29.642056Z"
    },
    "papermill": {
     "duration": 3.059087,
     "end_time": "2023-03-28T01:01:29.645574",
     "exception": false,
     "start_time": "2023-03-28T01:01:26.586487",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "875/875 [==============================] - 2s 2ms/step\n"
     ]
    }
   ],
   "source": [
    "X_test_sub = test_data.iloc[:, :]\n",
    "X_test_sub = X_test_sub.values.reshape(X_test_sub.shape[0], 28, 28, 1)\n",
    "y_pred_sub = model.predict(X_test_sub)\n",
    "y_pred_sub = np.argmax(y_pred_sub, axis=1)\n",
    "y_pred_sub = pd.Series(y_pred_sub,name=\"Label\")\n",
    "submission = pd.concat([pd.Series(range(1,28001),name = \"ImageId\"),y_pred_sub],axis = 1)\n",
    "submission.to_csv('submission.csv', index=False)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.12"
  },
  "papermill": {
   "default_parameters": {},
   "duration": 216.052895,
   "end_time": "2023-03-28T01:01:32.932550",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2023-03-28T00:57:56.879655",
   "version": "2.4.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
