#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    ,thread(&edit)
{
   // edit.appendMessage("Initialized Window");
    ui->setupUi(this);
    form.initialize(this);
    edit.initialize(this);
    // Create the button, make "this" the parent

    clearStatus = new QPushButton("Clear", this);


    theory_label = new QLabel("",this);//just to color background
    theory_text = new QLabel("Theory Output:", this);
    phase_text = new QLabel("SLM Profile:", this); 
    statusText = new QLabel("Status Window", this);
    theory_label->setStyleSheet("QLabel { background-color : #d3d3d3; }");
    //phase_label->setStyleSheet("border: 1px solid black;");
    phase_label = new QLabel("",this);
    theory_label->setAlignment(Qt::AlignCenter);
    phase_label->setAlignment(Qt::AlignCenter);
    resize();
    
                          // Connect button signal to appropriate slot
    connect(form.getGTA(), SIGNAL(released()), this, SLOT(handleButton()));
    connect(clearStatus, &QPushButton::clicked, this, &MainWindow::handleClearButton);
    //use this syntax in future ^^
}


void MainWindow::resizeEvent(QResizeEvent*)
{
    edit.resize(this);
    resize();
    form.resize(this);
}

MainWindow::~MainWindow()
{
    delete ui;
    delete phase_label;
    delete theory_label;
    delete theory_text;
    delete phase_text;
    delete statusText;
    delete clearStatus;
}

void MainWindow::resize() {
    QSize size = this->size();
    int X = size.width() / 3;
    int Y = size.height();
    phase_label->setGeometry(QRect(QPoint((2*X) + 10, 30),
                            QSize(X - 20, (Y / 3) - 15)));
    theory_label->setGeometry(QRect(QPoint((2*X) + 10, (Y / 3) + 45),
                            QSize(X - 20, (Y / 3) - 5)));
    phase_text->setGeometry(QRect(QPoint((2 * X) + 10, 0), QSize(250,30)));
    theory_text->setGeometry(QRect(QPoint((2 * X) + 10, (Y / 3) + 10), QSize(250, 30)));
    clearStatus->setGeometry(QRect(QPoint((2 * X) - 105, 10), QSize(100, 40)));
    statusText->setGeometry(QRect(QPoint((X) + 5, 10), QSize(200, 40)));
    //
    theory = theory.scaled(theory_label->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
    phase = phase.scaled(phase_label->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
    phase_label->setPixmap(QPixmap::fromImage(phase));
    theory_label->setPixmap(QPixmap::fromImage(theory));
}

void MainWindow::setImages() {
    if (!phase.load("data/phase_map.bmp")) {
        errBox("Could not load image for diplay", __FILE__, __LINE__);
    }
    if (!theory.load("data/theory_output.bmp")) {
        errBox("Could not load image for diplay", __FILE__, __LINE__);
    }
    theory = theory.mirrored(false, true);//flip vetrically
    resize();
    theory = theory.scaled(theory_label->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
    phase = phase.scaled(phase_label->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
    phase_label->setPixmap(QPixmap::fromImage(phase));
    theory_label->setPixmap(QPixmap::fromImage(theory));
}
void MainWindow::handleButton()
{
    if (thread.run_thread()) {
        edit.appendColorMessage("Main Thread Exited with Error", "red");
    }
    else {
        setImages();
    }
}

void MainWindow::handleClearButton() {
    edit.clear();
}

