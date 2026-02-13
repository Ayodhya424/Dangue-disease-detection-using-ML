# Dengue Disease Detection using Machine Learning

This project leverages supervised machine learning to predict the likelihood of dengue infection based on patient health data.  
By analyzing clinical and laboratory features, the model assists in early detection and risk assessment, supporting healthcare providers in decision-making.

---

## Features
- Predict dengue infection using **Random Forest Classifier**
- Handles both numerical and categorical patient data
- Provides feature importance for medical insights
- Easy-to-run Python scripts with clear workflow

---

## Approach
- **Algorithm Used**: Random Forest Classifier  
- **Why Random Forest?**  
  - Combines multiple decision trees to reduce overfitting  
  - Works well with diverse medical datasets  
  - Offers interpretability through feature importance  

---

## Workflow
1. **Data Collection**: Patient records with dengue-related features (e.g., fever duration, platelet count, WBC count).  
2. **Preprocessing**: Handle missing values, normalize data, encode categorical variables.  
3. **Model Training**: Train Random Forest classifier on labeled data (dengue positive vs. negative).  
4. **Evaluation**: Use Accuracy, Precision, Recall, F1-score, and ROC-AUC metrics.  
5. **Prediction**: Input new patient data → model outputs probability of dengue infection.  

---

## Results
- Achieved strong accuracy and balanced performance across metrics.  
- Platelet count and fever duration identified as key predictors.
  
# Project Structure
<img width="408" height="207" alt="image" src="https://github.com/user-attachments/assets/d080a82a-0611-4ee3-ba91-0344f810c755" />

# Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/Ayodhya424/Dangue-disease-detection-using-ML.git
cd Dangue-disease-detection-using-ML
```


Install dependencies with:
```bash
pip install -r requirements.txt
```

# Usage
Run the Flask Application
```bash
python app.py
```
Once the server is running, open your browser and go to:
```bash
http://127.0.0.1:5000
```

<img width="1120" height="336" alt="image" src="https://github.com/user-attachments/assets/78c42a81-186f-4bcc-925d-7f55f0da3df3" />


<img width="1677" height="987" alt="image" src="https://github.com/user-attachments/assets/140fdb2a-797a-414d-9f98-f5b95ab11278" />


## Author
Developed by **Ayodhay**  
GitHub: [Ayodhya424](https://github.com/Ayodhya424)  

This project was created as part of my machine learning portfolio, focusing on healthcare applications.  
If you use this project or find it helpful, please give it a ⭐ on GitHub!


