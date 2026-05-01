ORG 0000H
LJMP MAIN

; LCD Control Pins
RS      EQU P2.5
RW      EQU P2.6
EN      EQU P2.7

; ADC Control Pins
ALE     EQU P2.3
OE      EQU P2.4
START   EQU P2.1
EOC     EQU P2.0
ADC_CLK EQU P2.2

; Relay Control
RELAY   EQU P0.2      ; Output to transistor for relay
SET_TEMP EQU 22       ; Threshold temperature in Celsius

; Variables
ADC_VALUE EQU 30H
TEMP_INT  EQU 31H
TEMP_DEC  EQU 32H

; Interrupt: Timer0 for ADC Clock (toggle ADC_CLK)
ORG 000BH
    CPL ADC_CLK
    RETI

; Main program
ORG 0030H
MAIN:
    MOV SP, #70H     ; Initialize stack pointer

    ; LCD Initialization
    MOV A, #38H
    ACALL COMNWRT
    ACALL DELAY

    MOV A, #0EH
    ACALL COMNWRT
    ACALL DELAY

    MOV A, #01H
    ACALL COMNWRT
    ACALL DELAY

    MOV A, #06H
    ACALL COMNWRT
    ACALL DELAY

    MOV A, #80H
    ACALL COMNWRT
    ACALL DELAY

    ; Display "TEMP: "
    MOV A, #'T'
    ACALL DATAWRT
    MOV A, #'E'
    ACALL DATAWRT
    MOV A, #'M'
    ACALL DATAWRT
    MOV A, #'P'
    ACALL DATAWRT
    MOV A, #':'
    ACALL DATAWRT
    MOV A, #' '
    ACALL DATAWRT

    ; Timer0 for ADC Clock (Mode 2)
    MOV TMOD, #02H
    MOV TH0, #0F8H
    MOV TL0, #0F8H
    MOV IE, #82H
    SETB TR0

MAIN_LOOP:

    ; Start ADC conversion
    SETB ALE
    SETB START
    ACALL SHORT_DELAY
    CLR ALE
    CLR START

    ; Wait for EOC
WAIT_EOC_LOW:
    JB EOC, WAIT_EOC_LOW
WAIT_EOC_HIGH:
    JNB EOC, WAIT_EOC_HIGH

    ; Read ADC result
    SETB OE
    ACALL SHORT_DELAY
    MOV A, P1
    MOV ADC_VALUE, A
    CLR OE

    ; Calculate temperature = (ADC * 30) / 255
    MOV A, ADC_VALUE
    MOV B, #30
    MUL AB             ; Result in A (low) and B (high)
    MOV R0, A
    MOV R1, B

    ; Integer part = high byte
    MOV A, R1
    MOV TEMP_INT, A

    ; Decimal part = (low byte * 10) / 256
    MOV A, R0
    MOV B, #10
    MUL AB
    MOV A, B
    MOV TEMP_DEC, A

    ; Set LCD cursor to position 5
    MOV A, #85H
    ACALL COMNWRT

    ; Display temperature
    MOV A, TEMP_INT
    ACALL DISP_TEMP_WITH_DECIMAL

    ; Show degree symbol and 'C'
    MOV A, #0DFH
    ACALL DATAWRT
    MOV A, #'C'
    ACALL DATAWRT

    ; === Relay control ===
    MOV A, TEMP_INT
    CLR C
    SUBB A, #SET_TEMP
    JC TURN_RELAY_ON

TURN_RELAY_OFF:
    CLR RELAY
    SJMP RELAY_DONE

TURN_RELAY_ON:
    SETB RELAY

RELAY_DONE:

    ; Delay
    MOV R7, #100
    ACALL DELAY_MS

    SJMP MAIN_LOOP

; ===== Display Temperature with Decimal =====
DISP_TEMP_WITH_DECIMAL:
    MOV A, TEMP_INT
    CLR C
    SUBB A, #10
    JC DISP_UNITS

    MOV A, TEMP_INT
    MOV B, #10
    DIV AB
    ADD A, #30H
    ACALL DATAWRT
    MOV A, B
    SJMP DISP_UNITS_DIGIT

DISP_UNITS:
    MOV A, TEMP_INT

DISP_UNITS_DIGIT:
    ADD A, #30H
    ACALL DATAWRT

    MOV A, #'.'
    ACALL DATAWRT

    MOV A, TEMP_DEC
    ADD A, #30H
    ACALL DATAWRT
    RET

; ===== LCD Command Write =====
COMNWRT:
    MOV P3, A
    CLR RS
    CLR RW
    SETB EN
    ACALL SHORT_DELAY
    CLR EN
    RET

; ===== LCD Data Write =====
DATAWRT:
    MOV P3, A
    SETB RS
    CLR RW
    SETB EN
    ACALL SHORT_DELAY
    CLR EN
    RET

; ===== Short Delay =====
SHORT_DELAY:
    MOV R2, #30
D1: MOV R1, #255
D2: DJNZ R1, D2
    DJNZ R2, D1
    RET

; ===== Long Delay =====
DELAY:
    MOV R3, #100
HERE2:
    MOV R4, #255
HERE:
    DJNZ R4, HERE
    DJNZ R3, HERE2
    RET

; ===== Delay in ms using R7 =====
DELAY_MS:
    MOV R5, #7
DELAY_MS_LOOP:
    MOV R6, #255
DELAY_MS_INNER:
    DJNZ R6, DELAY_MS_INNER
    DJNZ R5, DELAY_MS_LOOP
    DJNZ R7, DELAY_MS
    RET

END
