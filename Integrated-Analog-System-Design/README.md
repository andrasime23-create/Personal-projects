# Modular Analog Signal Processing Pipeline

This project focuses on the design, dimensioning, and simulation of a complex four-stage analog circuit using AD8030 operational amplifiers. Developed as part of the SCIA (Analog Integrated Circuit Systems) curriculum, the pipeline transforms differential input signals into a rectified output through a series of precision processing stages

**System Architecture**:

-**Stage 1**: Instrumentation Amplifier – A three-op-amp configuration providing a gain of 12V/V for low-amplitude signals.  

-**Stage 2**: Tow-Thomas Band-Pass Filter – A frequency-selective stage with a 5kHz bandwidth and a quality factor of 0.707.  

-**Stage 3**: Programmable Gain Amplifier (PGA) – An inverting amplifier with 2dB resolution and 4 gain steps, utilizing out-of-signal-path switching.  

-**Stage 4*: Full-Wave Rectifier – A non-linear stage with linear gain, optimized for final signal conversion.

**Technical Validation (LTspice)**:

-**AC Analysis**: Confirmed an amplifier bandwidth of 11.13MHz and high interference rejection with a CMRR of 115.71dB.  

-**Transient Analysis**: Verified high-speed performance with a Slew Rate of 62.3V/µs and total harmonic distortion (THD) below 1%.  

-**Static Operation**: Validated DC operating points and output compensation for all integrated stages.

The project serves as a comprehensive laboratory for analog integrated circuit design, demonstrating how individual stages are integrated into a functional, high-performance signal chain.