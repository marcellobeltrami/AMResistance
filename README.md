# 🧬 Antibiotic Resistance Prediction

This project explores machine learning methods to predict **antibiotic resistance classes** from **protein sequences**.  
It uses protein-level encoding (amino acid one-hot representation) and deep learning models (LSTM/GRU/Transformer) to classify sequences into known antibiotic resistance categories.

---

## Project Structure

```
AntibioticResistance/
├── lib/
│   ├── encoding.py        # Protein one-hot encoder & data preparation
│   ├── model.py           # Deep learning model (LSTM/GRU/Transformer)
├── main.ipynb             # Jupyter notebook for experiments
├── requirements.txt       # Dependencies
└── README.md              # Project description
```

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/YOUR_USERNAME/AntibioticResistance.git
cd AntibioticResistance
pip install -r requirements.txt
```

---

## Data

The input dataset should contain protein sequences with metadata. Example format:

| Allele     | Gene family | Product name                                  | Class          | Sequence protein |
|------------|-------------|-----------------------------------------------|----------------|------------------|
| aac(3)-VIIIa | aac(3)-VIII | aminoglycoside N-acetyltransferase AAC(3)-VIIIa | AMINOGLYCOSIDE | MDEKELIERAGG...  |
| aac(6’)-32 | aac(6’)      | aminoglycoside N-acetyltransferase AAC(6’)-32  | AMINOGLYCOSIDE | MSPSKTPVTLR...  |
| qnrS1      | qnr          | quinolone resistance protein QnrS1             | QUINOLONE      | MTQDLMTLFNV...  |

---

## Pipeline

1. **Encoding**
   - Protein sequences → one-hot vectors (20 amino acids)
   - Padding/truncation to fixed length

2. **Model**
   - Sequence classification with LSTM/GRU/Transformer
   - Multi-class classification over antibiotic resistance **Class**

3. **Training**
   ```python
   from lib.encoding import prepare_data
   from lib.model import ResistanceModel

   X, y, label_encoder = prepare_data(df, max_len=500)
   model = ResistanceModel(max_len=500, num_classes=len(label_encoder.classes_))
   history = model.train(X, y, epochs=10, batch_size=32, validation_split=0.2)
   ```

4. **Prediction**
   ```python
   preds = model.predict(X[:5])
   label_encoder.inverse_transform(preds.argmax(axis=1))
   ```

---

## Saving & Loading Models

Save:
```python
model.model.save("./model/resistance_model.h5")
```

Load:
```python
from tensorflow.keras.models import load_model
loaded_model = load_model("./model/resistance_model.h5")
```

---

## 🧪Example Output

```
Predictions: ['AMINOGLYCOSIDE', 'QUINOLONE', 'LIPOPEPTIDE']
```

---

## TODO

- [ ] Improve dataset preprocessing  
- [ ] Try Transformer-based encoders  
- [ ] Evaluate with larger benchmark datasets  
- [ ] Deploy as an API for resistance prediction  

---

## License
MIT License
