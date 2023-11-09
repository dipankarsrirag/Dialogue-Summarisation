

### Project Tasks Overview

1. **Text Preprocessing:**
   - **Dialogsum Data:** Preprocessing is complete and ready for model training.
   - **Email Thread Dataset (Kaggle):** Inconsistent and requires further cleaning before it can be used for training.

2. **Vectorization:**
   We will be employing different vectorization techniques using pretrained embeddings and fine-tuning where necessary:
   - **GloVe:** Both the use of pretrained vectors and further fine-tuning.
   - **BERT:** Starting with a pretrained model and fine-tuning on our specific datasets.
   - **BART:** Utilizing a pretrained model with additional fine-tuning.
   - **T5:** Potential exploration of T5 embeddings for vectorization.

3. **Model Implementation:**
   We have been instructed to implement models based on the following architectures:
   - **Transformer-Based Solution:** To leverage state-of-the-art capabilities.
   - **LSTM/GRU:** To compare performance with transformer-based models.
   - **R-CNN:** To explore the capabilities of convolutional approaches in sequence modeling.

4. **Evaluation Metrics:**
   - Use the ROUGE score to evaluate the models' performance initially.

### Team Allocation

- **Dipankar:**
  - **Embedding:** GloVe
  - **Models:** R-CNN, Transformer

<<<<<<< HEAD
- **Pinak:**
  - **Embedding:** Bart

=======
>>>>>>> 93926ad (Moved things around. Created a corpus. Check data/processed.)
### Next Steps for Team Members:
- Each member needs to choose an embedding technique and at least one model to implement.
- Coordinate who will explore LSTM/GRU and who will investigate T5 embeddings.
- Determine who will be responsible for fine-tuning BERT and BART.
- Create your own branches and work on them.
- Let everyone know of your choices on the group.
