{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Untitled0.ipynb",
      "provenance": [],
      "authorship_tag": "ABX9TyPtbk78SX319ht3mG0QVrpW",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/levanquoc/Face_Recognition/blob/master/facenet.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "SDq48yEZnOKp",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "21d5f4a7-71cb-4099-f8ea-5fd2b9ca3897"
      },
      "source": [
        "from google.colab import drive\n",
        "import os\n",
        "\n",
        "drive.mount(\"/content/gdrive\")\n",
        "path = \"/content/gdrive/My Drive/AI_COLAB\"\n",
        "os.chdir(path)"
      ],
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Drive already mounted at /content/gdrive; to attempt to forcibly remount, call drive.mount(\"/content/gdrive\", force_remount=True).\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "2SG4CChEnnmZ",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 119
        },
        "outputId": "5af9c4c4-3f8d-4895-b23b-947049a8eb8e"
      },
      "source": [
        "!pip install face_recognition\n"
      ],
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Requirement already satisfied: face_recognition in /usr/local/lib/python3.6/dist-packages (1.3.0)\n",
            "Requirement already satisfied: Click>=6.0 in /usr/local/lib/python3.6/dist-packages (from face_recognition) (7.1.1)\n",
            "Requirement already satisfied: numpy in /usr/local/lib/python3.6/dist-packages (from face_recognition) (1.18.2)\n",
            "Requirement already satisfied: Pillow in /usr/local/lib/python3.6/dist-packages (from face_recognition) (7.0.0)\n",
            "Requirement already satisfied: dlib>=19.7 in /usr/local/lib/python3.6/dist-packages (from face_recognition) (19.18.0)\n",
            "Requirement already satisfied: face-recognition-models>=0.3.0 in /usr/local/lib/python3.6/dist-packages (from face_recognition) (0.3.0)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "_Iek9U36nrxp",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "from face_recognition import face_locations\n",
        "import matplotlib.pyplot as plt\n",
        "import cv2\n",
        "IMAGE_TEST = os.path.join(path, \"Dataset/khanh/001.jpg\")\n",
        "\n",
        "def _image_read(image_path):\n",
        "  \"\"\"\n",
        "  input:\n",
        "    image_path: link file ảnh\n",
        "  return:\n",
        "    image: numpy array của ảnh\n",
        "  \"\"\"\n",
        "  image = cv2.imread(image_path)\n",
        "  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)\n",
        "  return image\n",
        "\n",
        "\n",
        "image = _image_read(IMAGE_TEST)\n",
        "\n",
        "def _extract_bbox(image, single = True):\n",
        "  \"\"\"\n",
        "  Trích xuất ra tọa độ của face từ ảnh input\n",
        "  input:\n",
        "    image: ảnh input theo kênh RGB. \n",
        "    single: Lấy ra 1 face trên 1 bức ảnh nếu True hoặc nhiều faces nếu False. Mặc định True.\n",
        "  return:\n",
        "    bbox: Tọa độ của bbox: <start_Y>, <start_X>, <end_Y>, <end_X>\n",
        "  \"\"\"\n",
        "  bboxs = face_locations(image)\n",
        "  if len(bboxs)==0:\n",
        "    return None\n",
        "  if single:\n",
        "    bbox = bboxs[0]\n",
        "    return bbox\n",
        "  else:\n",
        "    return bboxs"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "U3qvGYexood-",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 265
        },
        "outputId": "4fb0624b-d7a4-4f72-cc51-5ac66b7ca941"
      },
      "source": [
        "def _extract_face(image, bbox, face_scale_thres = (20, 20)):\n",
        "  \"\"\"\n",
        "  input:\n",
        "    image: ma trận RGB ảnh đầu vào\n",
        "    bbox: tọa độ của ảnh input\n",
        "    face_scale_thres: ngưỡng kích thước (h, w) của face. Nếu nhỏ hơn ngưỡng này thì loại bỏ face\n",
        "  return:\n",
        "    face: ma trận RGB ảnh khuôn mặt được trích xuất từ image input.\n",
        "  \"\"\"\n",
        "  h, w = image.shape[:2]\n",
        "  try:\n",
        "    (startY, startX, endY, endX) = bbox\n",
        "  except:\n",
        "    return None\n",
        "  minX, maxX = min(startX, endX), max(startX, endX)\n",
        "  minY, maxY = min(startY, endY), max(startY, endY)\n",
        "  face = image[minY:maxY, minX:maxX].copy()\n",
        "  # extract the face ROI and grab the ROI dimensions\n",
        "  (fH, fW) = face.shape[:2]\n",
        "\n",
        "  # ensure the face width and height are sufficiently large\n",
        "  if fW < face_scale_thres[0] or fH < face_scale_thres[1]:\n",
        "    return None\n",
        "  else:\n",
        "    return face\n",
        "\n",
        "bbox = _extract_bbox(image)\n",
        "face = _extract_face(image, bbox)\n",
        "plt.axis(\"off\")\n",
        "plt.imshow(face)"
      ],
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.image.AxesImage at 0x7f9ffb07af28>"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 5
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "AD6H6m5woyns",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "98e78162-44db-4b2f-c2c6-24c03fb59564"
      },
      "source": [
        "from imutils import paths\n",
        "DATASET_PATH = \"./Dataset\"\n",
        "\n",
        "def _model_processing(face_scale_thres = (20, 20)):\n",
        "  \"\"\"\n",
        "  face_scale_thres: Ngưỡng (W, H) để chấp nhận một khuôn mặt.\n",
        "  \"\"\"\n",
        "  image_links = list(paths.list_images(DATASET_PATH))\n",
        "  images_file = [] \n",
        "  y_labels = []\n",
        "  faces = []\n",
        "  total = 0\n",
        "  for image_link in image_links:\n",
        "    split_img_links = image_link.split(\"/\")\n",
        "    # Lấy nhãn của ảnh\n",
        "    name = split_img_links[-2] \n",
        "    # Đọc ảnh\n",
        "    image = _image_read(image_link)\n",
        "    (h, w) = image.shape[:2]\n",
        "    # Detect vị trí các khuôn mặt trên ảnh. Gỉa định rằng mỗi bức ảnh chỉ có duy nhất 1 khuôn mặt của chủ nhân classes.\n",
        "    bbox =_extract_bbox(image, single=True)\n",
        "    # print(bbox_ratio)\n",
        "    if bbox is not None:\n",
        "      # Lấy ra face\n",
        "      face = _extract_face(image, bbox, face_scale_thres = (20, 20))\n",
        "      if face is not None:\n",
        "        faces.append(face)\n",
        "        y_labels.append(name)\n",
        "        images_file.append(image_links)\n",
        "        total += 1\n",
        "      else:\n",
        "        next\n",
        "  print(\"Total bbox face extracted: {}\".format(total))\n",
        "  return faces, y_labels, images_file\n",
        "\n",
        "faces, y_labels, images_file = _model_processing()"
      ],
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Total bbox face extracted: 15\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "3YmDTaGQo8Z9",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "import pickle\n",
        "\n",
        "def _save_pickle(obj, file_path):\n",
        "  with open(file_path, 'wb') as f:\n",
        "    pickle.dump(obj, f)\n",
        "\n",
        "def _load_pickle(file_path):\n",
        "  with open(file_path, 'rb') as f:\n",
        "    obj = pickle.load(f)\n",
        "  return obj\n",
        "\n",
        "_save_pickle(faces, \"./faces.pkl\")\n",
        "_save_pickle(y_labels, \"./y_labels.pkl\")\n",
        "_save_pickle(images_file, \"./images_file.pkl\")\n"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "rjeDF5E1pFoM",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "import tensorflow as tf\n",
        "model = tf.keras.Sequential([\n",
        "    tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28,28,1)),\n",
        "    tf.keras.layers.MaxPooling2D(pool_size=2),\n",
        "    tf.keras.layers.Dropout(0.3),\n",
        "    tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'),\n",
        "    tf.keras.layers.MaxPooling2D(pool_size=2),\n",
        "    tf.keras.layers.Dropout(0.3),\n",
        "    tf.keras.layers.Flatten(),\n",
        "    tf.keras.layers.Dense(256, activation=None), # No activation on final dense layer\n",
        "    tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)) # L2 normalize embeddings\n",
        "    ])"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "fKXgxixHpOE0",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "bc2ffdec-a679-4ec0-c5d1-7801747ac4d7"
      },
      "source": [
        "faces = _load_pickle(\"./faces.pkl\")\n",
        "import cv2\n",
        "import numpy as np\n",
        "faceResizes = []\n",
        "for face in faces:\n",
        "  face_rz = cv2.resize(face, (28, 28))\n",
        "  face_rz=cv2.cvtColor(face_rz,cv2.COLOR_BGR2GRAY)\n",
        "  faceResizes.append(face_rz)\n",
        "\n",
        "X_train = np.stack(faceResizes)\n",
        "X_train=X_train.reshape(X_train.shape[0],28,28,1)\n",
        "X_train.shape\n"
      ],
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(15, 28, 28, 1)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 9
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "nSJWJKr263By",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "X_train[0]"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "BRjgRBeVpdeT",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "import tensorflow_addons as tfa\n",
        "\n",
        "model.compile(\n",
        "    optimizer=tf.keras.optimizers.Adam(0.001),\n",
        "    loss=tfa.losses.TripletSemiHardLoss())"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "2CvIw3NOprPE",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "y_train=_load_pickle(\"y_labels.pkl\")"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "xRuB_3V-pe48",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "7c75a717-d985-4d54-e64c-17675d76d9e1"
      },
      "source": [
        "print(X_train.shape, len(y_train))"
      ],
      "execution_count": 11,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "(15, 28, 28, 1) 15\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "J2Duu9AWp354",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "780efad6-c877-4c67-c09d-7a48bd6e4d1d"
      },
      "source": [
        "gen_train = tf.data.Dataset.from_tensor_slices((X_train, y_train)).repeat().shuffle(1024).batch(2)\n",
        "gen_train"
      ],
      "execution_count": 19,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<BatchDataset shapes: ((None, 28, 28, 1), (None,)), types: (tf.uint8, tf.string)>"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 19
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ZyM3PmRIqFRT",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "0c5a0737-aec6-4fa7-eafd-c6dea82cca14"
      },
      "source": [
        "history = model.fit(\n",
        "    gen_train,\n",
        "    steps_per_epoch = 100,\n",
        "    epochs=5)"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Epoch 1/5\n"
          ],
          "name": "stdout"
        }
      ]
    }
  ]
}