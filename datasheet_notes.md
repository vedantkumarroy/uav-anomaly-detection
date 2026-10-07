\# Datasheet Notes



Two items to include in the paper's methods or data section.



\## 1. The extreme ratio flights



Approximately 111 flights in the corpus (6.8%) carry extreme values in the

`gps\_hpos` and `gps\_vpos` innovation test ratios. Each of these flights has

approximately 314 samples that exceed 10^10 in at least one of these ratios.



At the typical ratio topic rate of 2 Hz, 314 samples corresponds to

approximately 157 seconds of sustained large-ratio behavior. \*\*These are not

brief spikes.\*\* The position reference for these flights is unreliable for

several minutes of continuous flight.



The values are not sentinel values. They are the squared normalized innovation

computed by PX4 itself:



