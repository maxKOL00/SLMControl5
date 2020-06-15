#pragma once
#include <qplaintextedit.h>

class statusBox {
	public:
		statusBox::statusBox(QWidget* parent);
		void appendMessage(const QString& text);
		statusBox::~statusBox();
		void resize(QWidget* parent);
		void clear();
	private:
		QPlainTextEdit *edit;
};