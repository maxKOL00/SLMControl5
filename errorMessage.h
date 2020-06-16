#pragma once

#include <qmessagebox.h>
#include <sstream>



class ErrDialog
{
	public:
		ErrDialog::ErrDialog(std::string text) {
			msgBox = new QMessageBox;
			text = "ERROR: " + text;
			msgBox->setInformativeText(text.c_str());
			msgBox->setWindowModality(Qt::NonModal);
			msgBox->setStandardButtons(QMessageBox::Cancel);
			msgBox->setDefaultButton(QMessageBox::Cancel);
			msgBox->exec();
		}
		ErrDialog::~ErrDialog() {
			delete msgBox;
		}
	private:
		QMessageBox* msgBox;
	
};



template <typename T> void errBox(T msg, const std::string& file, int line)
{
	std::stringstream stream;
	std::string loc = std::string(file) + "; line " + std::to_string(line);
	stream << msg << "\n@ Location:" << loc;
	ErrDialog dlg(stream.str());
}