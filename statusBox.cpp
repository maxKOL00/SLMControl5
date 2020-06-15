#include "statusBox.h"
#include "qscrollbar.h"

statusBox::statusBox(QWidget* parent) {
    // set size and location of the button
    edit = new QPlainTextEdit(parent);

    //resize(parent);
    resize(parent);
    edit->setReadOnly(true);
    edit->setLineWrapMode(QPlainTextEdit::WidgetWidth);
}

void statusBox::resize(QWidget* parent) {
    QSize size = parent->size();
    int X = size.width();
    int Y = size.height();
    int startX = X / 3;
    int endY = Y - 20; 
    edit->setGeometry(QRect(QPoint(startX, 10),
                      QSize(startX, endY)));
}

statusBox::~statusBox() {
    delete edit;
}

void statusBox::appendMessage(const QString& text)
{
    edit->appendPlainText(text); // Adds the message to the widget
    edit->verticalScrollBar()->setValue(edit->verticalScrollBar()->maximum()); // Scrolls to the bottom
    
}

void statusBox::clear() {
    edit->clear();
}