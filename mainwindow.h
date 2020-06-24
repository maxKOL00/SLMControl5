#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "statusBox.h"
//#include "testClass.h"
#include <qpushbutton.h>
#include <qlabel.h>
#include <qcombobox.h>
#include "mainThread.h"
#include "configSystem.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    void resizeEvent(QResizeEvent*);
    void resize();
    ~MainWindow();
    void handleClearButton();
    void setImages();
    void SAVE();
    void SAVEAS();
    void NEW();

private slots:
    void handleButton();
    
private:
    Ui::MainWindow *ui;
    statusBox edit;
    configSystem form;
    //test test1;
    mainThread thread;
    QPushButton* m_button;
    QImage phase, theory;
    QLabel *phase_label, *theory_label, *phase_text, *theory_text;
    QLabel *statusText;
    QPushButton* clearStatus;
};
#endif // MAINWINDOW_H
