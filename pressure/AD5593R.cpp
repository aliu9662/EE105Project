#include "AD5593R.h"
#include <Arduino.h>
#include <Wire.h>

#define _ADAC_NULL           B00000000
#define _ADAC_ADC_SEQUENCE   B00000010 // ADC sequence register - Selects ADCs for conversion
#define _ADAC_GP_CONTROL     B00000011 // General-purpose control register - DAC and ADC control register
#define _ADAC_ADC_CONFIG     B00000100 // ADC pin configuration - Selects which pins are ADC inputs
#define _ADAC_DAC_CONFIG     B00000101 // DAC pin configuration - Selects which pins are DAC outputs
#define _ADAC_PULL_DOWN      B00000110 // Pull-down configuration - Selects which pins have an 85 kO pull-down resistor to GND
#define _ADAC_LDAC_MODE      B00000111 // LDAC mode - Selects the operation of the load DAC
#define _ADAC_GPIO_WR_CONFIG B00001000 // GPIO write configuration - Selects which pins are general-purpose outputs
#define _ADAC_GPIO_WR_DATA   B00001001 // GPIO write data - Writes data to general-purpose outputs
#define _ADAC_GPIO_RD_CONFIG B00001010 // GPIO read configuration - Selects which pins are general-purpose inputs
#define _ADAC_POWER_REF_CTRL B00001011 // Power-down/reference control - Powers down the DACs and enables/disables the reference
#define _ADAC_OPEN_DRAIN_CFG B00001100 // Open-drain configuration - Selects open-drain or push-pull for geeral-purpose outputs
#define _ADAC_THREE_STATE    B00001101 // Three-state pins - Selects which pins are three-stated
#define _ADAC_RESERVED       B00001110 // Reserved
#define _ADAC_SOFT_RESET     B00001111 // Software reset - Resets the AD5593R
 
 
#define _ADAC_VREF_ON        B00000010 // VREF on
#define _ADAC_SEQUENCE_ON    B00000010 // Sequence on
 
#define _ADAC_DAC_WRITE      B00010000 // DAC write
#define _ADAC_ADC_READ       B01000000 // ADC read
#define _ADAC_DAC_READ       B01010000 // DAC read
#define _ADAC_GPIO_READ      B01110000 // GPIO read
#define _ADAC_REG_READ       B01100000 // Register read

int dacAdress = 0x10;
                  //DAC0                 DAC7
bool DACenable[8] = {0, 0, 0, 0, 0, 0, 0, 0};
bool ADCenable[8] = {0, 0, 0, 0, 0, 0, 0, 0};
bool GPIenable[8] = {0, 0, 0, 0, 0, 0, 0, 0};
bool GPOenable[8] = {0, 0, 0, 0, 0, 0, 0, 0};


//*********************************************************************//
//Functions for ADAC
//Inputs
//*********************************************************************//
//Initalisation of the adac

//setting up the reffrence voltage
void setVref(bool enable){
  Wire.beginTransmission(dacAdress);
  Wire.write(_ADAC_POWER_REF_CTRL);  // [D0]
  uint8_t setup = enable << 1;
  Wire.write(setup);  // [D0]
  Wire.write(B00000000);  // [D0]
 
  Wire.endTransmission();
}
//setting up the DAC channels
void setupDAC(){
  Wire.beginTransmission(dacAdress);
  Wire.write(_ADAC_DAC_CONFIG);
  uint8_t setup = 0;
  for(int i = 0; i <= 7; i++){
    setup = setup | DACenable[i] << i;
  }
  Wire.write(0x00);  // [D0]
  Wire.write(setup);  // [D0]
 
  Wire.endTransmission();
}
//setting up the ADC channels
void setupADC(){
  Wire.beginTransmission(dacAdress);
  Wire.write(_ADAC_ADC_CONFIG);
  uint8_t setup = 0;
  for(int i = 0; i <= 7; i++){
    setup = setup | ADCenable[i] << i;
  }
  Wire.write(0x00);  // [D0]
  Wire.write(setup);  // [D0]
 
  Wire.endTransmission();
}
//setting up the GPO channels
void setupGPO(){
  Wire.beginTransmission(dacAdress);
  Wire.write(_ADAC_GPIO_WR_CONFIG);
  uint8_t setup = 0;
  for(int i = 0; i <= 7; i++){
    setup = setup | GPOenable[i] << i;
  }
  
  Wire.write(0x00);  // [D0]
  Wire.write(setup);  // [D0]
 
  Wire.endTransmission();
}
//setting up the GPI channels
void setupGPI(){
  Wire.beginTransmission(dacAdress);
  Wire.write(_ADAC_GPIO_RD_CONFIG);
  uint8_t setup = 0;
  for(int i = 0; i <= 7; i++){
    setup = setup | GPIenable[i] << i;
  }
  Wire.write(0x00);  // [D0]
  Wire.write(setup);  // [D0]
 
  Wire.endTransmission();
}
//setting the DAC value
void setDACVal(uint8_t DACnum, uint16_t DACvalue){
  DACnum = DACnum & 0x07;   // limiting DACnum to a range of 0 to 7
  DACvalue = DACvalue & 0x0FFF;   // limiting DACvalue to a range of 0 to 4095 (12- bits)
  Wire.beginTransmission(dacAdress);
  Wire.write(_ADAC_DAC_WRITE|DACnum);
  Wire.write((DACvalue >> 8) & 0x0F);
//  Wire.write(((DACnum & 0x07) << 4) | 0x80 | ((DACvalue >> 8) & 0x0F));
  uint8_t lower_8bits = DACvalue & 0xFF;
  Wire.write(lower_8bits);
  
//  Wire.write(DACvalue >> 8);  // 
//  Wire.write(DACvalue & 0x00FF);  // 
 
  Wire.endTransmission();
}
//reading the ADC value
uint16_t readADC(uint8_t DACnum){
  uint16_t ADCvalue = 0;
  DACnum = DACnum & 0x0F;

  Wire.beginTransmission(dacAdress);
  Wire.write(_ADAC_ADC_SEQUENCE);
  Wire.write(0x02);
  Wire.write(byte(1 << DACnum));
  Wire.endTransmission();

  Wire.beginTransmission(dacAdress);
  Wire.write(_ADAC_ADC_READ);
  Wire.endTransmission();
 
  Wire.requestFrom(int(dacAdress), int(2), int(1));
  if (Wire.available()) ADCvalue = (Wire.read() & 0x0f) << 8;
  if (Wire.available()) ADCvalue = ADCvalue | Wire.read();

  Wire.endTransmission();
  return ADCvalue;
}
//Setting the gain of the DAC
void setDACGain(bool gain){
  Wire.beginTransmission(dacAdress);
  Wire.write(_ADAC_GP_CONTROL);
  Wire.write(0x00);  // [D4]
  Wire.write(gain << 4);  // [D4]
  
  Wire.endTransmission();
}
//setting the LDAC channel
void setLDAC(uint8_t LDAC){
  Wire.beginTransmission(dacAdress);
  Wire.write(_ADAC_LDAC_MODE);
  Wire.write(0x00);  // [D0]
  Wire.write(LDAC & 0x03);  // [D0]
 
  Wire.endTransmission();
}
void initADAC(int dacAd, bool gain, bool VREF){
    dacAdress = dacAd;
    setVref(1);
    setupDAC();
    setupADC();
    setupGPI();
    setupGPO();
    setLDAC(0);
    setDACGain(1);
}
void setDACpin(unsigned char pin){
    if(pin > 7){
        return;
    }
    else{
        DACenable[pin] = 1;
        setupDAC();
        if(ADCenable[pin]){
            ADCenable[pin] = 0;
            setupADC();
        }
        if(GPIenable[pin]){
            GPIenable[pin] = 0;
            setupGPI();
        }
        if(GPOenable[pin]){
            GPOenable[pin] = 0;
            setupGPO();
        }
    }
}
void setADCpin(unsigned char pin){
    if(pin > 7){
        return;
    }
    else{
        ADCenable[pin] = 1;
        setupADC();
        if(DACenable[pin]){
            DACenable[pin] = 0;
            setupDAC();
        }
        if(GPIenable[pin]){
            GPIenable[pin] = 0;
            setupGPI();
        }
        if(GPOenable[pin]){
            GPOenable[pin] = 0;
            setupGPO();
        }
    }
}
void setGPIpin(unsigned char pin){
    if(pin > 7){
        return;
    }
    else{
        GPIenable[pin] = 1;
        setupGPI();
        if(DACenable[pin]){
            DACenable[pin] = 0;
            setupDAC();
        }
        if(ADCenable[pin]){
            ADCenable[pin] = 0;
            setupADC();
        }
        if(GPOenable[pin]){
            GPOenable[pin] = 0;
            setupGPO();
        }
    }
}
void setGPOpin(unsigned char pin){
    if(pin > 7){
        return;
    }
    else{
        GPOenable[pin] = 1;
        setupGPO();
        if(DACenable[pin]){
            DACenable[pin] = 0;
            setupDAC();
        }
        if(ADCenable[pin]){
            ADCenable[pin] = 0;
            setupADC();
        }
        if(GPIenable[pin]){
            GPIenable[pin] = 0;
            setupGPI();
        }
    }
}
