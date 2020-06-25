#include "configSystem.h"
#include <qfiledialog.h>
#include <fstream>



configSystem::configSystem(statusBox *box)
{
    editF = box;
}

void configSystem::updateConfigSelector() {
    QDir path("configFiles");
    QStringList files = path.entryList(QDir::Files);
    configSelection->addItems(files);
    configSelection->show();//get the files in the config directory.
}
void configSystem::initialize(QWidget* parent) {
    configSelection = new QComboBox(parent);
    QDir path("configFiles");
    QStringList files = path.entryList(QDir::Files);
    configSelection->addItems(files);
    configSelection->show();//get the files in the config directory.

    arrayType = new QComboBox(parent);
    arrayType->addItem("RECTANGULAR");
    arrayType->addItem("TRIANGULAR");
    arrayType->addItem("HONEYCOMB");
    arrayType->addItem("KAGOME");
    arrayType->addItem("TEST");
    arrayType->show();

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


    QObject::connect(save, &QPushButton::released, [this]() {SAVE(); });
    QObject::connect(saveAs, &QPushButton::released, [this]() {SAVEAS(); });
    QObject::connect(New, &QPushButton::released, [this]() {NEW(); });
    Parent = parent;
}

double configSystem::getWavelengthUM() {
    bool ok;
    double val = OT1->toPlainText().toDouble(&ok);
    if (val < 0.1 || val > 1 || !ok) {
        editF->appendColorMessage(" wavelength not physical. Using default 0.850", "red");
        return 0.850;
    }
    else {
        return val;
    }
}
int configSystem::getFocalLenghtMM() {
    bool ok;
    int val = OT2->toPlainText().toInt(&ok);
    if (val < 1|| val > 2000 || !ok) {
        editF->appendColorMessage("Focal length not physical. Using default 180", "red");
        return 180;
    }
    else {
        return val;
    }
}
double configSystem::getWaistUM() {
    bool ok;
    double val = OT3->toPlainText().toDouble(&ok);
    if (val < 0.0 || !ok) {
        editF->appendColorMessage("Waist not physical. Using default 16.5", "red");
        return 16.5;
    }
    else {
        return val;
    }
}
double configSystem::getBeamWaistX() {
    bool ok;
    double val = OT4->toPlainText().toDouble(&ok);
    if (val < 0.0 || !ok) {
        editF->appendColorMessage("Beam Waist not physical. Using default 15", "red");
        return 15;
    }
    else {
        return val;
    }
}
double configSystem::getBeamWaistY() {
    bool ok;
    double val = OT5->toPlainText().toDouble(&ok);
    if (val < 0.0 || !ok) {
        editF->appendColorMessage("Beam Waist not physical. Using default 15", "red");
        return 15;
    }
    else {
        return val;
    }
}
int configSystem::getLUTPatchSizeX() {
    bool ok;
    int val = CT1->toPlainText().toInt(&ok);
    if (val < 1 || val > 3000 || !ok) {
        editF->appendColorMessage("Invalid LUT patch size px #. Using default 480", "red");
        return 480;
    }
    else {
        return val;
    }
}
int configSystem::getLUTPatchSizeY() {
    bool ok;
    int val = CT2->toPlainText().toInt(&ok);
    if (val < 1 || val > 3000 || !ok) {
        editF->appendColorMessage("Invalid LUT patch size Y px #. Using default 288", "red");
        return 288;
    }
    else {
        return val;
    }
}
int configSystem::getHorizontalOffset() {
    bool ok;
    int val = CT3->toPlainText().toInt(&ok);
    if (!ok) {
        editF->appendColorMessage("Invalid Horizontal Offset. Using default 384", "red");
        return 384;
    }
    else {
        return val;
    }
}
int configSystem::getFrameRate() {
    bool ok;
    int val = CT4->toPlainText().toInt(&ok);
    if (val < 0 || !ok) {
        editF->appendColorMessage("Invalid Frame Rate. Using default 5", "red");
        return 5;
    }
    else {
        return val;
    }
}
int configSystem::getGratingMax() {
    bool ok;
    int val = CT5->toPlainText().toInt(&ok);
    if (val < 1 || !ok) {
        editF->appendColorMessage("Invalid Max Grating. Using default 255", "red");
        return 255;
    }
    else {
        return val;
    }
}
int configSystem::getGratingPeriod() {
    bool ok;
    int val = CT6->toPlainText().toInt(&ok);
    if (val < 1 || !ok) {
        editF->appendColorMessage("Invalid Grating Period. Using default 4", "red");
        return 4;
    }
    else {
        return val;
    }
}
int configSystem::getPatchSizeX() {
    bool ok;
    int val = CT7->toPlainText().toInt(&ok);
    if (val < 0 || val > 3000 || !ok) {
        editF->appendColorMessage("Invalid patch size px #. Using default 64", "red");
        return 64;
    }
    else {
        return val;
    }
}
int configSystem::getPatchSizeY() {
    bool ok;
    int val = CT8->toPlainText().toInt(&ok);
    if (val < 0 || val > 3000 || !ok) {
        editF->appendColorMessage("Invalid patch size px #. Using default 64", "red");
        return 64;
    }
    else {
        return val;
    }
}
int configSystem::getGratingDiff() {
    bool ok;
    int val = CT9->toPlainText().toInt(&ok);
    if (val < 1 || val > 3000 || !ok) {
        editF->appendColorMessage("Invalid Grating Diff. Using default 40", "red");
        return 40;
    }
    else {
        return val;
    }
}
int configSystem::getNumTrapsX(){
    bool ok;
    int val = TT1->toPlainText().toInt(&ok);
    if (val < 1 || !ok) {
        editF->appendColorMessage("Invalid Num Traps X. Using default 5", "red");
        return 5;
    }
    else {
        return val;
    }
}
int configSystem::getNumTrapsY(){
    bool ok;
    int val = TT2->toPlainText().toInt(&ok);
    if (val < 1 || !ok) {
        editF->appendColorMessage("Invalid Num Traps Y. Using default 5", "red");
        return 5;
    }
    else {
        return val;
    }
}
double configSystem::getSpacingX(){
    bool ok;
    double val = TT3->toPlainText().toDouble(&ok);
    if (val < 0.0 || !ok) {
        editF->appendColorMessage("trap spacing. Using default 250", "red");
        return 250;
    }
    else {
        return val;
    }
}
double configSystem::getSpacingY(){
    bool ok;
    double val = TT4->toPlainText().toDouble(&ok);
    if (val < 0.0 || !ok) {
        editF->appendColorMessage("trap spacing. Using default 250", "red");
        return 250;
    }
    else {
        return val;
    }
}
double configSystem::getRadialShiftX(){
    bool ok;
    double val = TT5->toPlainText().toDouble(&ok);
    if (val < 0.0 || !ok) {
        editF->appendColorMessage("radial shift invalid. Using default 0", "red");
        return 0;
    }
    else {
        return val;
    }
}
double configSystem::getRadialShiftY(){
    bool ok;
    double val = TT6->toPlainText().toDouble(&ok);
    if (val < 0.0 || !ok) {
        editF->appendColorMessage("radial shift invalid. Using default 0", "red");
        return 0;
    }
    else {
        return val;
    }
}
double configSystem::getAxialShift() {
    bool ok;
    double val = TT7->toPlainText().toDouble(&ok);
    if (val < 0.0 || !ok) {
        editF->appendColorMessage("axial shift invalid. Using default 0", "red");
        return 0;
    }
    else {
        return val;
    }
}
int configSystem::getPaddedPixels(){ 
    bool ok;
    int val = TT8->toPlainText().toInt(&ok);
    if (val < 1 || !ok) {
        editF->appendColorMessage("Invalid padded size. Using default 8192", "red");
        return 8192;
    }
    else {
        return val;
    }
}
int configSystem::getMaxItterations() {
    bool ok;
    int val = TT9->toPlainText().toInt(&ok);
    if (val < 1 || !ok) {
        editF->appendColorMessage("Invalid Itterations. Using default 80", "red");
        return 80;
    }
    else {
        return val;
    }
}
int configSystem::getMaxCameraItterations() {
    bool ok;
    int val = TT10->toPlainText().toInt(&ok);
    if (val < 1 || !ok) {
        editF->appendColorMessage("Invalid camera Itterations. Using default 25", "red");
        return 25;
    }
    else {
        return val;
    }
}
int configSystem::getFixedPhaseItterations() {
    bool ok;
    int val = TT11->toPlainText().toInt(&ok);
    if (val < 1 || !ok) {
        editF->appendColorMessage("Invalid fixed phase Itterations. Using default 80", "red");
        return 80;
    }
    else {
        return val;
    }
}
double configSystem::getFixedPhaseNonuniformity() {
    bool ok;
    double val = TT12->toPlainText().toDouble(&ok);
    if (val < 0.00|| !ok) {
        editF->appendColorMessage("Nonuniformity fixed phase percent invalid. Using default 1.5", "red");
        return 1.5;
    }
    else {
        return val;
    }
}
double configSystem::getMaxNonuniformityPercent() {
    bool ok;
    double val = TT13->toPlainText().toDouble(&ok);
    if (val < 0.00 || !ok) {
        editF->appendColorMessage("Nonuniformity maximum percent invalid. Using default 0.05", "red");
        return 0.05;
    }
    else {
        return val;
    }
}
double configSystem::getMaxCameraNonuniformityPercent() {
    bool ok;
    double val = TT14->toPlainText().toDouble(&ok);
    if (val < 0.00 || !ok) {
        editF->appendColorMessage("Nonuniformity maximum percent invalid. Using default 0.7", "red");
        return 0.7;
    }
    else {
        return val;
    }
}
double configSystem::getWeight() {
    bool ok;
    double val = TT15->toPlainText().toDouble(&ok);
    if (val < 0.01 || val > 1.0 || !ok) {
        editF->appendColorMessage("Nonuniformity maximum percent invalid. Using default 0.7", "red");
        return 0.7;
    }
    else {
        return val;
    }
}
bool configSystem::getCameraFeedback() {
    if (TT16->isChecked()) {
        return true;
    }
    else { return false; }
}
int configSystem::getAxialScanLower(){
    bool ok;
    int val = AT1->toPlainText().toInt(&ok);
    if (!ok) {
        editF->appendColorMessage("Invalid lower axial scan range. Using default 1500", "red");
        return 1500;
    }
    else {
        return val;
    }
}
int configSystem::getAxialScanUpper() {
    bool ok;
    int val = AT2->toPlainText().toInt(&ok);
    if (!ok) {
        editF->appendColorMessage("Invalid upper axial scan range. Using default -1500", "red");
        return -1500;
    }
    else {
        return val;
    }
}
int configSystem::getAxialScanStepSize() {
    bool ok;
    int val = AT3->toPlainText().toInt(&ok);
    if (!ok) {
        editF->appendColorMessage("Invalid axial scan stepsize. Using default 50", "red");
        return 50;
    }
    else {
        return val;
    }
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
    TweezerText->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
    pointX += (BSpace + BWidth);
    arrayType->setGeometry(QRect(QPoint(pointX, pointY), QSize(BWidth, buttonHeight)));
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
    delete configSelection; delete arrayType;
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


void configSystem::SAVE() {
    if (!std::filesystem::exists("configFiles")) {
        std::filesystem::create_directories("configFiles");
    }
    QString current = configSelection->itemText(arrayType->currentIndex());
    std::string file = current.toLocal8Bit().constData();
        makeNewFile("configFiles/" + file);//make a whole new file
}

void configSystem::SAVEAS() {
    if (!std::filesystem::exists("configFiles")) {
        std::filesystem::create_directories("configFiles");
    }
    QString fileName = QFileDialog::getSaveFileName(Parent, "Save File",
                                                    QString("configFiles"), "Configuration (*.json)");
    if (fileName.toLocal8Bit().constData() == "") {
        return;
    }
    //QDir path("configFiles");
    //QStringList files = path.entryList(QDir::Files);
    //bool newFile = true;
    //for (const auto& i : files)//check if need to make new file or not
    //{
    //    QFileInfo info(fileName);//remove absolute path
    //    if (i == info.fileName()) {
    //        saveFile(fileName.toLocal8Bit().constData());//just update the existing file
    //        newFile = false;
    //        break;
    //    }
    //    if (info.fileName() == "") { newFile = false; break; }
    //}
    //if (newFile) {
        makeNewFile(fileName.toLocal8Bit().constData());//make a whole new file
        updateConfigSelector();
    //}
}

void configSystem::makeNewFile(std::string filename) {
    std::fstream config_fstream(filename, std::fstream::out);
    if (!config_fstream.is_open()) {
        throw std::runtime_error("Could not open config.json");
    }
    nlohmann::json file;
    file["SLM"]["SLM_PX_X"] = 1920;
    file["SLM"]["SLM_PX_Y"] = 1152;
    file["SLM"]["SENSOR_SIZE_X_MM"] = 17.6;
    file["SLM"]["SENSOR_SIZE_Y_MM"] = 10.7;
    file["SLM"]["MAX_FRAME_RATE"] = 31;
    file["CAMERA"]["CAMERA_ID"] = "DEV_1AB2280011BE";
    file["CAMERA"]["CAMERA_MAX_FRAME_RATE"] = 14;
    file["CAMERA"]["CAMERA_PX_X"] = 2592;
    file["CAMERA"]["CAMERA_PX_Y"] = 1944;
    file["CAMERA"]["CAMERA_PX_SIZE_UM"] = 2.2;
    file["CAMERA"]["EXPOSURE_MODE"] = "manual";
    file["CAMERA"]["EXPOSURE_TIME_US"] = 400;
    file["SERIAL"]["PORT_NAME"] = "COM8";
    file["SERIAL"]["BAUD_RATE"] = 115200;
    file["OPTICAL_SYSTEM"]["WAVELENGTH_UM"] = getWavelengthUM();
    file["OPTICAL_SYSTEM"]["FOCAL_LENGTH_MM"] = getFocalLenghtMM();
    file["OPTICAL_SYSTEM"]["WAIST_UM"] = getWaistUM();
    file["OPTICAL_SYSTEM"]["BEAM_WAIST_X_MM"] = getBeamWaistX();
    file["OPTICAL_SYSTEM"]["BEAM_WAIST_Y_MM"] = getBeamWaistY();
    file["CALIBRATION"]["LUT_PATCH_SIZE_X_PX"] = getLUTPatchSizeX();
    file["CALIBRATION"]["LUT_PATCH_SIZE_Y_PX"] = getLUTPatchSizeY();
    file["CALIBRATION"]["HORIZONTAL_OFFSET"] = getHorizontalOffset();
    file["CALIBRATION"]["FRAME_RATE"] = getFrameRate();
    file["CALIBRATION"]["GRATING_PERIOD_PX"] = getGratingPeriod();
    file["CALIBRATION"]["GRATING_MAX"] = getGratingMax();
    file["CALIBRATION"]["PATCH_SIZE_X_PX"] = getPatchSizeX();
    file["CALIBRATION"]["PATCH_SIZE_Y_PX"] = getPatchSizeY();
    file["CALIBRATION"]["GRATING_DIFF"] = getGratingDiff();
    file["CALIBRATION"]["PD_READOUT_FOLDER"] = "pd_readout";
    file["CALIBRATION"]["IMAGE_FOLDER"] = "images";
    file["TWEEZER_ARRAY_GENERATION"]["ARRAY_GEOMETRY"] = arrayType->itemText(arrayType->currentIndex()).toLocal8Bit().constData();
    file["TWEEZER_ARRAY_GENERATION"]["NUM_TRAPS_X"] = getNumTrapsX();
    file["TWEEZER_ARRAY_GENERATION"]["NUM_TRAPS_Y"] = getNumTrapsY();
    file["TWEEZER_ARRAY_GENERATION"]["SPACING_X_UM"] = getSpacingX();
    file["TWEEZER_ARRAY_GENERATION"]["SPACING_Y_UM"] = getSpacingY();
    file["TWEEZER_ARRAY_GENERATION"]["RADIAL_SHIFT_X_UM"] = getRadialShiftX();
    file["TWEEZER_ARRAY_GENERATION"]["RADIAL_SHIFT_Y_UM"] = getRadialShiftY();
    file["TWEEZER_ARRAY_GENERATION"]["AXIAL_SHIFT_UM"] = getAxialShift();
    file["TWEEZER_ARRAY_GENERATION"]["NUMBER_OF_PIXELS_PADDED"] = getPaddedPixels();
    file["TWEEZER_ARRAY_GENERATION"]["MAX_ITERATIONS"] = getMaxItterations();
    file["TWEEZER_ARRAY_GENERATION"]["MAX_ITERATIONS_CAMERA_FEEDBACK"] = getMaxCameraItterations();
    file["TWEEZER_ARRAY_GENERATION"]["FIXED_PHASE_LIMIT_ITERATIONS"] = getFixedPhaseItterations();
    file["TWEEZER_ARRAY_GENERATION"]["FIXED_PHASE_LIMIT_NONUNIFORMITY_PERCENT"] = getFixedPhaseNonuniformity();
    file["TWEEZER_ARRAY_GENERATION"]["MAX_NONUNIFORMITY_PERCENT"] = getMaxNonuniformityPercent();
    file["TWEEZER_ARRAY_GENERATION"]["MAX_NONUNIFORMITY_CAMERA_FEEDBACK_PERCENT"] = getMaxCameraNonuniformityPercent();
    file["TWEEZER_ARRAY_GENERATION"]["WEIGHTING_PARAMETER"] = getWeight();
    file["TWEEZER_ARRAY_GENERATION"]["CAMERA_FEEDBACK_ENABLED"] = getCameraFeedback();
    file["TWEEZER_ARRAY_GENERATION"]["LAYER_SEPARATION_UM"] = 50000.0;
    file["TWEEZER_ARRAY_GENERATION"]["OUTPUT_FOLDER"] = "data";
    file["TWEEZER_ARRAY_GENERATION"]["SAVE_DATA"] = true;
    file["TWEEZER_ARRAY_GENERATION"]["RANDOM_SEED"] = 95843670349586;
    file["TWEEZER_ARRAY_GENERATION"]["AXIAL_SCAN"]["AXIAL_SCAN_RANGE_LOWER_UM"] = getAxialScanLower();
    file["TWEEZER_ARRAY_GENERATION"]["AXIAL_SCAN"]["AXIAL_SCAN_RANGE_UPPER_UM"] = getAxialScanUpper();
    file["TWEEZER_ARRAY_GENERATION"]["AXIAL_SCAN"]["AXIAL_SCAN_STEPSIZE_UM"] = getAxialScanStepSize();

    config_fstream << std::setw(4) << file << std::endl;
    config_fstream.close();
}

void configSystem::NEW() {

}