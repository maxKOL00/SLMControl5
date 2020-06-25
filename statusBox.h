#pragma once
#include <qplaintextedit.h>

class statusBox : public QObject {
	public:
		statusBox::statusBox() {}
		void initialize(QWidget* parent);
		void appendMessage(const QString& text);
		void appendColorMessage(const QString& text, std::string color);
		statusBox::~statusBox();
		void resize(QWidget* parent);
		void clear();
		void messageLogged(const QString& message);
	private:
		QPlainTextEdit *edit;
};