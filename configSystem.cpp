#include "configSystem.h"

configSystem::configSystem(QWidget* parent)
{

    configSelection = new QComboBox(parent);
    save = new QPushButton("Save", parent);
    New = new QPushButton("New", parent);
    saveAs = new QPushButton("Save As", parent);
    Calibrate = new QPushButton("Calibrate", parent);
    GTA = new QPushButton("Generate Tweezers", parent);

    saveInfo = new QLabel("*SLM and Camera Information must be entered manually.", parent);
    OpticalText = new QLabel("Optical System", parent);
    CalibrationText = new QLabel("Calibration", parent);
    TweezerText = new QLabel("Tweezer Array", parent);
    AxialText = new QLabel("Axial Scan", parent);
    O1 = new QLabel("Wavelength UM: ", parent); OT1 = new QTextEdit(parent);
    O2 = new QLabel("Focal Length MM: ", parent); OT2 = new QTextEdit(parent);
    O3 = new QLabel("Waist UM: ", parent); OT3 = new QTextEdit(parent);
    O4 = new QLabel("Beam Waist X MM: ", parent); OT4 = new QTextEdit(parent);
    O5 = new QLabel("Beam Waist Y MM: ", parent); OT5 = new QTextEdit(parent);

    C1 = new QLabel("LUT Patch Size X Px: ", parent); CT1 = new QTextEdit(parent);
    C2 = new QLabel("LUT Patch Size Y Px: ", parent); CT2 = new QTextEdit(parent);
    C3 = new QLabel("Horizontal Offset: ", parent); CT3 = new QTextEdit(parent);
    C4 = new QLabel("Frame Rate: ", parent); CT4 = new QTextEdit(parent);
    C5 = new QLabel("Grating Max: ", parent); CT5 = new QTextEdit(parent);
    C6 = new QLabel("Grating Period Px: ", parent); CT6 = new QTextEdit(parent);
    C7 = new QLabel("Patch Size X Px: ", parent); CT7 = new QTextEdit(parent);
    C8 = new QLabel("Patch Size Y Px: ", parent); CT8 = new QTextEdit(parent);
    C9 = new QLabel("Grating Diff: ", parent); CT9 = new QTextEdit(parent);

    T1 = new QLabel("Num Traps X: ", parent); TT1 = new QTextEdit(parent);
    T2 = new QLabel("Num Traps Y: ", parent); TT2 = new QTextEdit(parent);
    T3 = new QLabel("Spacing X UM: ", parent); TT3 = new QTextEdit(parent);
    T4 = new QLabel("Spacing Y UM: ", parent); TT4 = new QTextEdit(parent);
    T5 = new QLabel("Radial Shift X UM: ", parent); TT5 = new QTextEdit(parent);
    T6 = new QLabel("Radial Shift Y UM: ", parent); TT6 = new QTextEdit(parent);
    T7 = new QLabel("Axial Shift UM: ", parent); TT7 = new QTextEdit(parent);
    T8 = new QLabel("Num Pixels Padded: ", parent); TT8 = new QTextEdit(parent);
    T9 = new QLabel("Max Iterations: ", parent); TT9 = new QTextEdit(parent);
    T10 = new QLabel("Max Camera Iter. : ", parent); TT10 = new QTextEdit(parent);
    T11 = new QLabel("Fixed Phase Itter. : ", parent); TT11 = new QTextEdit(parent);
    T12 = new QLabel("Fixed Phase Nonuni. : ", parent); TT12 = new QTextEdit(parent);
    T13 = new QLabel("Max Nonuniformity %: ", parent); TT13 = new QTextEdit(parent);
    T14 = new QLabel("Cam. Nonuniformity %: ", parent); TT14 = new QTextEdit(parent);
    T15 = new QLabel("Weighting Parameter: ", parent); TT15 = new QTextEdit(parent);
    T16 = new QLabel("Camera Feedback", parent); TT16 = new QCheckBox(parent);

    A1 = new QLabel("Range Lower UM: ", parent); AT1 = new QTextEdit(parent);
    A2 = new QLabel("Range Upper UM: ", parent); AT2 = new QTextEdit(parent);
    A3 = new QLabel("Scan Stepsize UM: ", parent); AT3 = new QTextEdit(parent);
}

void configSystem::resize(QWidget* parent) {
    QSize size = parent->size();
    int mainMargin = 50;
    int buttonHeight = size.height() / 40;// 50 full size
    int sectionBreak = size.height() / 57; // 35 full size
    int newLine = (size.height() / 100) + buttonHeight; //70 full size
    int left = mainMargin;
    int top = mainMargin;
    int right = (size.width() / 3) - mainMargin;
    //int bottom = (size.height() / 3) - mainMargin;
    int Y = size.height();
    int pointX = left;
    int pointY = top;
    int center = (right - left) / 2;
    //Top BAR
    int BSpace = ((double(right) - double(left)) / 23.6);
    int BWidth = (right - left - (BSpace * 3)) / 4;
    configSelection->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX += (BSpace + BWidth);
    save->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX += (BSpace + BWidth);
    saveAs->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX += (BSpace + BWidth);
    New->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX = left;
    //discalimer text
    pointY += newLine;
    saveInfo->setGeometry(QRect(QPoint(pointX, pointY), QSize(right - left, buttonHeight)));
    pointY += newLine;
    pointY += sectionBreak;
    //Optical section
    OpticalText->setGeometry(QRect(QPoint(pointX, pointY), QSize(right - left, buttonHeight)));
    pointY += newLine;
    O1->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX += (BSpace + BWidth);
    OT1->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX += (BSpace + BWidth);
    O2->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX += (BSpace + BWidth);
    OT2->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX = left;
    pointY += newLine;
    O3->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX += (BSpace + BWidth);
    OT3->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX += (BSpace + BWidth);
    O4->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX += (BSpace + BWidth);
    OT4->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX = left;
    pointY += newLine;
    O5->setGeometry(QRect(QPoint(center - BSpace - (BWidth / 2), pointY), QSize(BWidth, buttonHeight)));
    OT5->setGeometry(QRect(QPoint(center + (BWidth / 2), pointY), QSize(BWidth, buttonHeight)));
    pointY += newLine;
    //Calibration
    pointY += sectionBreak;
    CalibrationText->setGeometry(QRect(QPoint(pointX, pointY), QSize(right - left, buttonHeight)));
    pointY += newLine;
    C1->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX += (BSpace + BWidth);
    CT1->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX += (BSpace + BWidth);
    C2->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX += (BSpace + BWidth);
    CT2->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX = left;
    pointY += newLine;
    C3->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX += (BSpace + BWidth);
    CT3->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX += (BSpace + BWidth);
    C4->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX += (BSpace + BWidth);
    CT4->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX = left;
    pointY += newLine;
    C5->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX += (BSpace + BWidth);
    CT5->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX += (BSpace + BWidth);
    C6->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX += (BSpace + BWidth);
    CT6->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX = left;
    pointY += newLine;
    C7->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX += (BSpace + BWidth);
    CT7->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX += (BSpace + BWidth);
    C8->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX += (BSpace + BWidth);
    CT8->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX = left;
    pointY += newLine;
    C9->setGeometry(QRect(QPoint(center - BSpace - (BWidth / 2), pointY), QSize(BWidth, buttonHeight)));
    CT9->setGeometry(QRect(QPoint(center + (BWidth / 2), pointY), QSize(BWidth, buttonHeight)));
    pointY += newLine;
    //Tweezer
    pointY += sectionBreak;
    TweezerText->setGeometry(QRect(QPoint(pointX, pointY), QSize(right - left, buttonHeight)));
    pointY += newLine;
    T1->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX += (BSpace + BWidth);
    TT1->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX += (BSpace + BWidth);
    T2->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX += (BSpace + BWidth);
    TT2->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX = left;
    pointY += newLine;
    T3->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX += (BSpace + BWidth);
    TT3->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX += (BSpace + BWidth);
    T4->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX += (BSpace + BWidth);
    TT4->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX = left;
    pointY += newLine;
    T5->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX += (BSpace + BWidth);
    TT5->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX += (BSpace + BWidth);
    T6->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX += (BSpace + BWidth);
    TT6->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX = left;
    pointY += newLine;
    T7->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX += (BSpace + BWidth);
    TT7->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX += (BSpace + BWidth);
    T8->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX += (BSpace + BWidth);
    TT8->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX = left;
    pointY += newLine;
    T9->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX += (BSpace + BWidth);
    TT9->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX += (BSpace + BWidth);
    T10->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX += (BSpace + BWidth);
    TT10->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX = left;
    pointY += newLine;
    T11->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX += (BSpace + BWidth);
    TT11->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX += (BSpace + BWidth);
    T12->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX += (BSpace + BWidth);
    TT12->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX = left;
    pointY += newLine;
    T13->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX += (BSpace + BWidth);
    TT13->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX += (BSpace + BWidth);
    T14->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX += (BSpace + BWidth);
    TT14->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX = left;
    pointY += newLine;
    T15->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX += (BSpace + BWidth);
    TT15->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX += (BSpace + BWidth);
    T16->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX += (BSpace + BWidth);
    TT16->setGeometry(QRect(QPoint(pointX, pointY), QSize(30, buttonHeight)));
    pointX = left;
    pointY += newLine;
    //Axial Scan
    pointY += sectionBreak;
    AxialText->setGeometry(QRect(QPoint(pointX, pointY), QSize(right - left, buttonHeight)));
    pointY += newLine;
    A1->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX += (BSpace + BWidth);
    AT1->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX += (BSpace + BWidth);
    A2->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX += (BSpace + BWidth);
    AT2->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX = left;
    pointY += newLine;
    A3->setGeometry(QRect(QPoint(center - BSpace - (BWidth / 2), pointY), QSize(BWidth, buttonHeight)));
    AT3->setGeometry(QRect(QPoint(center + (BWidth / 2), pointY), QSize(BWidth, buttonHeight)));
    pointY += newLine;
    pointX = left;
    //Bottom Bar
    pointY += sectionBreak;
    Calibrate->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX += (BSpace + BWidth);
    GTA->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX += (BSpace + BWidth);
}

configSystem::~configSystem() {
    //delete ui;
    delete configSelection;
    delete save;
    delete saveAs;
    delete New;
    delete saveInfo; delete OpticalText; delete CalibrationText; delete TweezerText; 
    delete AxialText; delete GTA; delete Calibrate;
    delete O1; delete O2; delete O3; delete O4; delete O5;
    delete OT1; delete OT2; delete OT3; delete OT4; delete OT5;
    delete C1; delete C2; delete C3; delete C4; delete C5; delete C6; delete C7; delete C8; delete C9;
    delete CT1; delete CT2; delete CT3; delete CT4; delete CT5; delete CT6; delete CT7; delete CT8; delete CT9;
    delete T1; delete T2; delete T3; delete T4; delete T5; delete T6; delete T7; delete T8;
    delete T9; delete T10; delete T11; delete T12; delete T13; delete T14; delete T15; delete T16;
    delete TT1; delete TT2; delete TT3; delete TT4; delete TT5; delete TT6; delete TT7; delete TT8;
    delete TT9; delete TT10; delete TT11; delete TT12; delete TT13; delete TT14; delete TT15; delete TT16;
}

