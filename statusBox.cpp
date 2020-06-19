#include "statusBox.h"
#include "qscrollbar.h"
#include "qapplication.h"

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
    int endY = Y - 50; 
    edit->setGeometry(QRect(QPoint(startX, 60),
                      QSize(startX, endY)));
}

statusBox::~statusBox() {
    delete edit;
}

void statusBox::appendMessage(const QString& text)
{
    edit->appendPlainText(text); // Adds the message to the widget
    edit->verticalScrollBar()->setValue(edit->verticalScrollBar()->maximum()); // Scrolls to the bottom
    qApp->processEvents();//need this to display it in mainthreadloop otherwise waits till end
}
constexpr unsigned int str2int(const char* str, int h = 0)
{
    return !str[h] ? 5381 : (str2int(str, h + 1) * 33) ^ str[h];
}

const QColor getColor(std::string color) {
    switch (str2int(color.c_str())) {
        case str2int("red"):
            return Qt::red;
        case str2int("green"):
            return Qt::green;
    }
    return Qt::black;
}

void statusBox::appendColorMessage(const QString& text, std::string color) {

    /*QPalette p = edit->palette();
    const QColor Color = getColor(color);
    p.setColor(QPalette::Base, Color);
    edit->setPalette(p);*/
    QTextCharFormat tf;
    tf = edit->currentCharFormat();
    tf.setForeground(QBrush(getColor(color)));
    edit->setCurrentCharFormat(tf);
    // append
    edit->appendPlainText(text); // Adds the message to the widget
    edit->verticalScrollBar()->setValue(edit->verticalScrollBar()->maximum()); // Scrolls to the bottom
    // restore
    tf = edit->currentCharFormat();
    tf.setForeground(QBrush(Qt::black));
    edit->setCurrentCharFormat(tf);
    qApp->processEvents();
}

void statusBox::clear() {
    edit->clear();
    qApp->processEvents();
}

void statusBox::messageLogged(const QString& message)
{
    bool doScroll = (edit->verticalScrollBar()->isVisible()) || edit->verticalScrollBar()->sliderPosition() == edit->verticalScrollBar()->maximum();

    //QTextCursor c = edit->textCursor();
    //c.beginEditBlock();
    //c.movePosition(QTextCursor::End);
    //c.insertText(message);
    //c.endEditBlock();

    if (doScroll)
    {
        edit->verticalScrollBar()->setSliderPosition(edit->verticalScrollBar()->sliderPosition());
    }

}