import sys

from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

form_window = uic.loadUiType('./cat_and_dog.ui')[0]

class Exam(QWidget, form_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.model = load_model('./cat_and_dog_0.832.h5')
        self.path = ('../datasets/cat_dog/train/cat.4.jpg', '')

        pixmap = QPixmap(self.path[0])
        self.label.setPixmap(pixmap)

        self.pushButton.clicked.connect(self.image_open_slot)
    def image_open_slot(self):
        old_path = self.path
        self.path = QFileDialog.getOpenFileName(self,
                    'Open file', '../datasets/cat_dog/train',
                    'Image Files(*.jpg;*.png);;All Files(*.*)')

        if self.path[0] == '':
            self.path = old_path
        else:
            pixmap = QPixmap(self.path[0])
            self.label.setPixmap(pixmap)
            try:
                img = Image.open(self.path[0])
                img = img.convert('RGB')
                img = img.resize((64, 64))  # w*h 사이즈 새로, tuple로 묶어서 줘야함
                data = np.asarray(img)
                data = data / 255
                data = data.reshape(1, 64, 64, 3) #1개 뿐이 없는데 64의 64에 3가지 색상
                pred = self.model.predict(data)
                if pred < 0.5:
                    #예측 값이 0에 가까우면 고양이니
                    self.lbl_result.setText('cat {}%'.format
                                            ((1 - pred[0][0]) * 100))
                else:
                    self.lbl_result.setText('dog {}%'.format
                                            ((pred[0][0]) * 100))
            except:
                print('error')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = Exam()
    mainWindow.show()
    sys.exit(app.exec_())