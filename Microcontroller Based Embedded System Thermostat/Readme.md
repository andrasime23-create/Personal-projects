# Room Thermostat Embedded System

A digital room thermostat designed to monitor and display temperature using an AT89C51 microcontroller.  

-**Sensor & Logic**: Integrates an LM35DZ linear sensor with an AD8629 differential amplifier (20x gain) for precision signal conditioning.  

-**Conversion**: Uses an ADC0808 8-bit converter to translate analog readings into digital data for processing.  

-**Firmware**: Features custom software written in C and Assembly to manage LCD communication, ADC timing, and temperature calculations.  

-**Interface*: Provides real-time visual feedback via an LM016L LCD display.