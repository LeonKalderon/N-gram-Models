# Natural Language Processing Assignment
# N-gram Models
The objective of this assignment was to create a simple Python application that calculate bi-grams and tri-grams models
with Laplace and Kneser-ney smoothing.

## Project Desrciption
The main module of the project is main.py that evaluates the different language models.
* Laplace smoothing on bi-grams
* Laplace smoothing on tri-grams
* Linear interpolation with the above two models.
* Kneser-Ney smoothing on bi-grams.

For all the above models, we calculate cross-entropy and perplexity in order to tune, evaluate and compare them.

### Problems Encoutered
For the entire corpus (http://www.statmt.org/europarl/) we need a minimum of 16gb RAM to avoid memory error.

## Getting Started
### Requirements
* ntlk
* sklearn
* typing

## Authors
* **Georgia Sarri**
* **George Vafeidis**
* **Leon Kalderon**
