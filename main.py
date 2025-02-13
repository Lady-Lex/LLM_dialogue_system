import os
import sys

import qdarkstyle
from PyQt5 import QtWidgets, QtCore, uic
from PyQt5.QtCore import QThread, pyqtSignal
from qdarkstyle import LightPalette

from chatmodules.repeater import repeater_get_response
from chatmodules.gpt4all_chatbot import GPT4AllChatbot

import warnings
warnings.filterwarnings("ignore")

models_dir = "models/"

class WorkThread(QThread):
    trigger = pyqtSignal(str)

    def __init__(self, dialogue_list, func):
        super(WorkThread, self).__init__()
        self.dialogue_list = dialogue_list
        self.func = func

    def run(self):
        response = self.func(self.dialogue_list)
        self.trigger.emit(response)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ui_path = os.path.dirname(os.path.abspath(__file__))
        uic.loadUi(os.path.join(ui_path, "gui/main.ui"), self)

        self.sendPushButton.clicked.connect(self.send_message)
        self.clearPushButton.clicked.connect(self.clear_all)

        self.inputPlainTextEdit.installEventFilter(self)

        # Add models to modelComboBox
        models_list = os.listdir(models_dir)

        self.repeater_model_name = f"{len(models_list)}: Repeater"
        for i, model_file in enumerate(models_list):
            self.modelComboBox.addItem(f"{i}: " + model_file)
        self.modelComboBox.addItem(self.repeater_model_name)


        self.current_model_name = self.repeater_model_name
        self.current_model = GPT4AllChatbot(0)

        self.dialogue_list = []
        self.work = None

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.KeyPress and obj is self.inputPlainTextEdit:
            if event.key() == QtCore.Qt.Key_Return and self.inputPlainTextEdit.hasFocus():
                self.send_message()
        return super().eventFilter(obj, event)

    def send_message(self):
        # Check current model
        chosen_model = self.modelComboBox.currentText()
        old_text = self.dialogueTextEdit.toPlainText()

        if self.current_model_name != chosen_model:
            self.dialogueTextEdit.setPlainText(old_text + "Chosen model changed to: " + chosen_model + ". Loading..." + "\n\n")

            self.current_model_name = chosen_model
            if chosen_model != self.repeater_model_name:
                self.current_model.exit()
                self.current_model = GPT4AllChatbot(int(chosen_model.split(":")[0]))

        message = self.inputPlainTextEdit.toPlainText()
        self.dialogue_list.append(message)

        # Get current text from dialogueTextEdit
        self.dialogueTextEdit.setPlainText(old_text + "User: " + message + "\n\nThinking...")
        
        if chosen_model == self.repeater_model_name:
            work_func = repeater_get_response
        else:
            work_func = self.current_model.get_response

        self.inputPlainTextEdit.clear()
        self.inputPlainTextEdit.setReadOnly(True)
        self.sendPushButton.setEnabled(False)
        self.clearPushButton.setEnabled(False)
        
        print(self.dialogue_list)
        self.work = WorkThread(self.dialogue_list, work_func)
        self.work.trigger.connect(self.handle_response)
        self.work.start()

    def handle_response(self, response: str):
        self.dialogue_list.append(response)
        last_response = response.split("- ")[-1]

        old_text = self.dialogueTextEdit.toPlainText()
        self.dialogueTextEdit.setPlainText(old_text[:-11] + "Bot: " + response + "\n\n")
        self.dialogueTextEdit.setPlainText(old_text[:-11] + "Bot: " + last_response + "\n\n")

        self.inputPlainTextEdit.setReadOnly(False)
        self.sendPushButton.setEnabled(True)
        self.clearPushButton.setEnabled(True)

    def clear_all(self):
        self.dialogue_list = []
        self.dialogueTextEdit.clear()
        self.inputPlainTextEdit.clear()

        chosen_model = self.modelComboBox.currentText()
        if chosen_model == self.repeater_model_name:
            pass
        else:
            self.current_model.reset_memory()


if __name__ == '__main__':
    # Fix DPI issues on Windows
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet(palette=LightPalette))
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
