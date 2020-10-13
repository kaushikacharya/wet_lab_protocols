# Wet Lab Protocols

This repository contains the Conditional Random Fields (CRF) model for the shared task.

#### Shared Task:
http://noisy-text.github.io/2020/wlp-task.html

##### BRAT styled annotated protocols
http://kb1.cse.ohio-state.edu:8010/index.xhtml#/wnut_20_data/

##### Entity, Action and Relation extraction demo 
- http://bionlp.osu.edu:5000/protocols
- Demo page of the system developed as part of Kulkarni et al's 2018 paper.

#### Data
- https://github.com/jeniyat/WNUT_2020_NER
- https://github.com/jeniyat/WNUT_2020_RE


#### How to run?

Below are the example commands.

- Load protocol and entity annotations(if available)
    - Segments protocol into sentences and words.
    - Executes spaCy's NLP pipeline over the protocol.
    ```python
      python -m src.dataset --ann_format standoff --protocol_id 101
    ```
    
- Execute Conditional Random Fields (CRF) model
    - Train CRF model
        ```python
          python -m src.crf
        ```
        - Saves model in ./output/models directory
        
    - Validate development set
        ```python
          python -m src.crf --train_model ./output/models/model_standoff.pkl --evaluate_collection
        ```
    - Predict on test set
       ```python
          python -m src.crf --train_model ./output/models/model_standoff.pkl --predict_collection
        ``` 