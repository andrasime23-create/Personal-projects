# Power Sources

The project has the aim of designing a power source that has an input voltage of 13V and an output of 3.3V, so that it can be used for real-life small electronic devices/projects.
## Project steps
- **Designing the rectifier**: A full-wave rectifier with a capacitive filter was chosen to transform AC voltage into DC.
- **The regulator**: This time, I designed a linear regulator, which keeps the voltage in a constant range, no matter the input voltage variations, nor the variations of the current or temperature. For comparative purposes, I also used an integrated regulator, showing the applicability in both cases. 
- **Buck converter**: The final step is to reduce the input voltage to a smaller output. This is done by implementing a Buck converter.
- **PID controller**: The output voltage from the Buck converter can be easily influenced by output variations of the load current. That is why it is important to implement a PID controller, which enhances the stability of the converter and makes it react faster to different changes.

