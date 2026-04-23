#ifndef _AD5593R_H_
#define _AD5593R_H_
#include <Arduino.h>
void initADAC(int dacAd, bool gain, bool VREF);
void setDACpin(unsigned char pin);
void setDACGain(bool gain);
void setADCpin(unsigned char pin);
void setGPIpin(unsigned char pin);
void setGPOpin(unsigned char pin);
void setDACVal(uint8_t DACnum, uint16_t DACvalue);
uint16_t readADC(uint8_t DACnum);
#endif
