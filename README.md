# 🍽️ FoodDetector

FoodDetector è un progetto di prova che utilizza modelli di deep learning per classificare immagini di cibo in diverse categorie. L'obiettivo è esplorare tecniche di training, distillazione e compressione dei modelli per ottimizzare le prestazioni e semplificare il deployment.

---

## 📂 Struttura del Progetto

### Componenti Principali
1. **`web.py`**: Applicazione web interattiva sviluppata con Streamlit per caricare immagini e ottenere predizioni in tempo reale.
2. **`run_v2.ipynb`**: Notebook per il training e la valutazione di modelli come Vision Transformer, AlexNet, ASAP e XNAS.
3. **`embed.ipynb`**: Notebook per la distillazione del modello insegnante (Vision Transformer) in uno studente (MobileNetV2) e la compressione in FP32, FP16 e INT8.
4. **`dataset/`**: Directory contenente immagini organizzate in sottocartelle, una per ogni classe.

---

## 🚀 Funzionalità

1. **Classificazione delle Immagini**
   - Utilizza un modello MobileNetV2 distillato per classificare immagini di cibo.
   - Supporta categorie definite nel dataset.

2. **Training e Valutazione**
   - Addestramento di modelli avanzati come Vision Transformer e architetture personalizzate.
   - Visualizzazione delle curve di training e confronto delle prestazioni.

3. **Distillazione e Compressione**
   - Distillazione del modello insegnante in uno studente più leggero.
   - Compressione del modello in diversi formati per ottimizzare il deployment.

4. **Interfaccia Web**
   - Caricamento di immagini tramite un'interfaccia user-friendly.
   - Predizione in tempo reale con visualizzazione delle categorie.

---

## 🛠️ Requisiti

### Librerie Python
- `torch`, `torchvision`, `streamlit`, `numpy`, `matplotlib`, `tqdm`

### Hardware
- **GPU**: Consigliata per il training.
- **CPU**: Sufficiente per l'inferenza.

---

## 🧪 Utilizzo

### **Interfaccia Web**
1. Avvia l'applicazione web:
   ```bash
   streamlit run web.py
   ```
2. Carica un'immagine di cibo e visualizza la categoria predetta.

### **Training**
1. Apri il notebook `run_v2.ipynb`.
2. Esegui le celle per addestrare e valutare i modelli.

### **Distillazione e Compressione**
1. Apri il notebook `embed.ipynb`.
2. Esegui le celle per distillare il modello insegnante e comprimere il modello studente.

---

## 📊 Risultati

### Modelli
- **Insegnante**: Vision Transformer
- **Studente**: MobileNetV2

### Accuratezza
- **FP32**: ~88%
- **FP16**: ~88%
- **INT8**: ~91%

---
