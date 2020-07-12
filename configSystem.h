#pragma once
#include <qobject.h>
#include <qlabel.h>
#include <qpushbutton.h>
#include <qcombobox.h>
#include <qtextedit.h>
#include <qcheckbox.h>
#include <filesystem>
#include <qbuttongroup.h>
#include <qradiobutton.h>
#include "nlohmann/json.hpp"
#include "statusBox.h"
#include "parameters.h"


class configSystem {

	public:
        configSystem::configSystem(statusBox* box);
        void initialize(QWidget* parent);
        configSystem::~configSystem();
        QPushButton* getGTA() { return GTA; }
        QPushButton* getAxial() { return AxialScan; }
        void showStar() { saveText->setVisible(true); }
        void SAVE();
        void SAVEAS();
        void NEW();
        //void SAVEAS();
        void resize();
        std::string getConfigFile();
        void setToolTips();
        void setTextBoxes();
        void changed();
        void displayBasic();
        void displayAdvanced();
        void toggled(int id);
        void connectSaveStar();
	private:
        statusBox* editF;
        void makeNewFile(std::string filename);
        void updateConfigSelector();
        double getWavelengthUM();
        double getFocalLenghtMM();
        double getWaistUM();
        double getBeamWaistX();
        double getBeamWaistY();
        int getLUTPatchSizeX();
        int getLUTPatchSizeY();
        int getHorizontalOffset();
        int getFrameRate();
        int getPatchSizeX();
        int getPatchSizeY();
        int getGratingMax();
        int getGratingPeriod();
        int getGratingDiff();
        int getNumTrapsX();
        int getNumTrapsY();
        double getSpacingX();
        double getSpacingY();
        double getRadialShiftX();
        double getRadialShiftY();
        double getAxialShift();
        int getPaddedPixels();
        int getMaxItterations();
        int getMaxCameraItterations();
        int getFixedPhaseItterations();
        double getFixedPhaseNonuniformity();
        double getMaxNonuniformityPercent();
        double getMaxCameraNonuniformityPercent();
        double getWeight();
        bool getCameraFeedback();
        double getAxialScanLower();
        double getAxialScanUpper();
        double getAxialScanStepSize();
        QPushButton* save;
        QPushButton* saveAs;
        QPushButton* New;
        QRadioButton* basic, *advanced;
        QButtonGroup* settings;
        QComboBox* configSelection;
        QComboBox* arrayType;
        QLabel* saveInfo, * OpticalText, * CalibrationText, * TweezerText, * AxialText, *saveText;
        QLabel* O1, * O2, * O4, * O5;
        QTextEdit* OT1, * OT2, * OT3, * OT4, * OT5; 
        QLabel* C1, * C2, * C3, * C4, * C5, * C6, * C7, * C8;
        QTextEdit* CT1, * CT2, * CT3, * CT4, * CT5, * CT6, * CT7, * CT8;
        QLabel *T1, *T2, *T3, *T4, *T5, *T6, *T7, *T8, *T9, *T10, *T11, *T12, *T13, *T14, *T15, *T16;
        QTextEdit *TT1, *TT2, *TT3, *TT4, *TT5, *TT6, *TT7, *TT8, *TT9, *TT10, *TT11, *TT12, *TT13, *TT14, *TT15;
        QCheckBox* TT16;
        QLabel* A1, * A2, * A3;
        QTextEdit* AT1, * AT2, * AT3;
        QPushButton *GTA;
        QPushButton *Calibrate;
        QPushButton* AxialScan;
        QWidget* Parent;
};
