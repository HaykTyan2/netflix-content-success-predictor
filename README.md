## 📂 Netflix442

- **/colab_notebook/**
  - **/Netflix442/**
    - `netflix_titles.csv` – Netflix dataset  
    - `imdb_top_1000.csv` – IMDb dataset  
    - `netflix_merged.csv` – merged dataset of both  
  - **/COLAB_SRC/**
    - `merging_data.ipynb` – merges the two datasets  
    - `new_netflix_project.ipynb` – trains the model on the merged dataset  

- **/static/** – CSS files for the frontend  
- **/templates/** – HTML files for the frontend  
- **app.py** – Flask backend connecting the model and frontend  
- **pipeline.joblib** – trained ML pipeline for use with the backend  

## !!!! How to Run !!!!

1. **Clone the repository**  
   ```bash
   git clone https://github.com/HaykTyan2/netflix-content-success-predictor.git
   cd netflix-content-success-predictor
2. Install dependencies
   ```bash
   pip install -r requirements.txt
3. Run the Flask App
   ```bash
   python app.py
4. Open your browser and navigate to:
   ```bash
   http://127.0.0.1:5000/
   
📘 Notes

The Colab notebooks (.ipynb) are included for dataset merging and model training.

A pre-trained pipeline (pipeline.joblib) is already provided, so you can run the application immediately without retraining the model.
