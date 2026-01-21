                               ####### 2nd Paper Discussion Points 

## Introduction 
- MEG is a technique used to measure brain activity. It is very good at telling when something happens in the brain (high time resolution).But the difficult part is figuring out where exactly inside the brain those signals came from.
- To solve this difficult problem, the authors introduce a new method called Deep-MEG. 
- Deep-MEG uses deep learning to directly learn the relationship between: MEG signals (input) and Brain source locations and strengths (output)
- The proposed method can find out surface as well as deep brain regions as sources of MEG signals (Main advantage)

## Method and Procedure
- Deep-MEG is designed to understand when the brain signal changes and where it comes from at the same time.So it has a temporal block and spatial block in its overall architecture 
i) Temporal Block (Time Information)
- MEG signals are time-varying signals.
- The Temporal Block looks at a small time window of 21 samples, which is about 20 milliseconds.
- It uses 4 CNN layers to 
    - Detect patterns in how the signal changes over time 
    - Extract important time-domain features
