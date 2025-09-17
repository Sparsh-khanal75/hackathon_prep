from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib

app = FastAPI()

# Try to load model with error checking
try:
    model = joblib.load('iris_model.pkl')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

class Flower(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def home():
    return {"message": "API working", "model_loaded": model is not None}

@app.post("/predict")
def predict_flower(flower: Flower):
    if model is None:
        return {"error": "Model not loaded"}
    
    # Make prediction
    data = [[flower.sepal_length, flower.sepal_width, 
             flower.petal_length, flower.petal_width]]
    
    prediction = model.predict(data)[0]
    flowers = ["setosa", "versicolor", "virginica"]
    
    return {"predicted_flower": flowers[prediction]}

@app.get("/ui", response_class=HTMLResponse)
def get_ui():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Iris Flower Predictor</title>
        <style>
            body { font-family: Arial; max-width: 600px; margin: 50px auto; padding: 20px; }
            input { padding: 10px; margin: 5px; width: 200px; }
            button { padding: 10px 20px; background: #007cba; color: white; border: none; border-radius: 5px; }
            button:hover { background: #005a87; }
            .result { margin-top: 20px; padding: 15px; background: #f0f8ff; border-radius: 5px; }
            .flower { font-size: 24px; font-weight: bold; color: #2d5a27; }
        </style>
    </head>
    <body>
        <h1>🌸 Iris Flower Predictor</h1>
        <p>Enter flower measurements to predict the iris species:</p>
        
        <form id="flowerForm">
            <div>
                <label>Sepal Length (cm):</label><br>
                <input type="number" id="sepal_length" step="0.1" placeholder="5.1" required>
            </div>
            <div>
                <label>Sepal Width (cm):</label><br>
                <input type="number" id="sepal_width" step="0.1" placeholder="3.5" required>
            </div>
            <div>
                <label>Petal Length (cm):</label><br>
                <input type="number" id="petal_length" step="0.1" placeholder="1.4" required>
            </div>
            <div>
                <label>Petal Width (cm):</label><br>
                <input type="number" id="petal_width" step="0.1" placeholder="0.2" required>
            </div>
            <br>
            <button type="submit">🔮 Predict Flower</button>
        </form>
        
        <div id="result" class="result" style="display:none;">
            <h3>Prediction Result:</h3>
            <div class="flower" id="prediction"></div>
        </div>

        <script>
            document.getElementById('flowerForm').onsubmit = async function(e) {
                e.preventDefault();
                
                const data = {
                    sepal_length: parseFloat(document.getElementById('sepal_length').value),
                    sepal_width: parseFloat(document.getElementById('sepal_width').value),
                    petal_length: parseFloat(document.getElementById('petal_length').value),
                    petal_width: parseFloat(document.getElementById('petal_width').value)
                };
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(data)
                    });
                    
                    const result = await response.json();
                    document.getElementById('prediction').textContent = 
                        '🌺 ' + result.predicted_flower.toUpperCase();
                    document.getElementById('result').style.display = 'block';
                    
                } catch (error) {
                    alert('Error: ' + error.message);
                }
            };
        </script>
    </body>
    </html>
    """