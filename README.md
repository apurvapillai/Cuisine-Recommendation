# Cuisine Recommendation System

A machine learning-powered web application that predicts the cuisine of a recipe based on selected ingredients.  
Built with **Python, Flask, and scikit-learn**, this project leverages a logistic regression classifier to identify cuisine types from ingredient combinations.

---

## 🚀 Features
- Predicts the cuisine of a dish based on user-selected ingredients.
- Handles ingredient normalization to account for variations like "red onions" → "onion".
- Provides fallback recommendations if insufficient ingredient data is provided.
- Interactive **Flask web app** with:
  - Home page
  - Ingredient selection page
  - Prediction results page

---

## 📂 Tech Stack
- **Languages & Frameworks:** Python, Flask
- **ML Libraries:** scikit-learn, Pandas, NumPy
- **Model:** One-vs-Rest Logistic Regression Classifier
- **Data Source:** Kaggle Recipe Ingredients Dataset (39,000+ recipes)
- **Frontend:** HTML, CSS (Flask templates)

---

## 📊 Methodology
1. **Data Preprocessing**
   - Ingredient normalization and filtering for common vegetables
   - MultiLabelBinarizer to convert ingredients into binary vectors

2. **Model Training**
   - One-vs-Rest Logistic Regression Classifier
   - Evaluated using Accuracy, Precision, Recall, and F1 Score

3. **Web Application**
   - Integrated trained model (`model.pkl`) into a Flask backend
   - User interface allows ingredient selection and cuisine prediction

---

## 🖼 Sample Workflow
1. Select ingredients →  
2. Model predicts cuisine →  
3. Returns the most likely cuisine or fallback suggestion if data is insufficient.

---

## 🔮 Future Enhancements
- Add support for spices, meats, and grains to expand cuisine coverage.
- Multi-cuisine recommendations for fusion recipes.
- User profiles and personalized recommendations.
- Improved accuracy using ensemble models or neural networks.

---

## 📸 Screenshots
<img width="742" height="404" alt="image" src="https://github.com/user-attachments/assets/46c177bf-9766-4fb3-9b2d-11fb215056b8" />
<img width="758" height="392" alt="image" src="https://github.com/user-attachments/assets/4abc2d80-36b3-4bb1-a7a2-3ebe50ff05ff" />
<img width="753" height="394" alt="image" src="https://github.com/user-attachments/assets/6a4aa398-33d6-44d8-8753-434c08bf50ba" />


---

## ⚡ How to Run
```bash
# Clone the repository
git clone https://github.com/apurvapillai/Cuisine-Recommendation.git

# Navigate to the project folder
cd Cuisine-Recommendation

# Install dependencies
pip install -r requirements.txt

# Run the Flask app
python app.py
