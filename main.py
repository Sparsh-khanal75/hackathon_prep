from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import time

app = FastAPI()

# Simple global counters for reliable metrics
prediction_counts = {"setosa": 0, "versicolor": 0, "virginica": 0}
total_api_calls = 0
total_prediction_time = 0.0

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
    global total_api_calls, prediction_counts, total_prediction_time
    
    if model is None:
        return {"error": "Model not loaded"}
    
    # Start timing and increment call counter
    start_time = time.time()
    total_api_calls += 1
    
    # Make prediction
    data = [[flower.sepal_length, flower.sepal_width, 
             flower.petal_length, flower.petal_width]]
    
    prediction = model.predict(data)[0]
    flowers = ["setosa", "versicolor", "virginica"]
    predicted_flower = flowers[prediction]
    
    # Update counters
    prediction_counts[predicted_flower] += 1
    total_prediction_time += (time.time() - start_time)
    
    return {"predicted_flower": predicted_flower}

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
            .stats { margin-top: 30px; padding: 15px; background: #f9f9f9; border-radius: 5px; }
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

        <div class="stats">
            <h3>📊 API Links:</h3>
            <p><a href="/stats" target="_blank">📈 Live Statistics</a></p>
            <p><a href="/metrics" target="_blank">🔍 Raw Metrics</a></p>
            <p><a href="/docs" target="_blank">📚 API Documentation</a></p>
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

@app.get("/stats")
def get_stats():
    """Human-readable statistics for demo"""
    total_predictions = sum(prediction_counts.values())
    avg_time = total_prediction_time / max(total_predictions, 1) if total_predictions > 0 else 0
    
    return {
        "🌸 ML API Statistics": {
            "total_predictions": total_predictions,
            "total_api_calls": total_api_calls,
            "average_prediction_time": f"{avg_time:.4f} seconds",
            "flower_breakdown": prediction_counts,
            "most_predicted": max(prediction_counts, key=prediction_counts.get) if total_predictions > 0 else "None"
        },
        "📊 Demo Instructions": {
            "make_predictions": "Visit /ui to make predictions",
            "view_raw_metrics": "Visit /metrics for Prometheus format",
            "api_docs": "Visit /docs for API documentation",
            "watch_live": "Refresh this page to see updated stats!"
        }
    }

@app.get("/metrics")
def get_metrics():
    """Simple Prometheus-style metrics for monitoring"""
    total_preds = sum(prediction_counts.values())
    avg_time = total_prediction_time / max(total_preds, 1) if total_preds > 0 else 0
    
    metrics = f"""# HELP flower_predictions_total Number of predictions by flower type
# TYPE flower_predictions_total counter
flower_predictions_total{{flower_type="setosa"}} {prediction_counts["setosa"]}
flower_predictions_total{{flower_type="versicolor"}} {prediction_counts["versicolor"]}
flower_predictions_total{{flower_type="virginica"}} {prediction_counts["virginica"]}

# HELP api_calls_total Total API calls made
# TYPE api_calls_total counter
api_calls_total {total_api_calls}

# HELP prediction_time_average_seconds Average prediction time in seconds
# TYPE prediction_time_average_seconds gauge
prediction_time_average_seconds {avg_time:.6f}

# HELP total_predictions_made Total number of ML predictions
# TYPE total_predictions_made counter
total_predictions_made {total_preds}
"""
    return Response(metrics, media_type="text/plain")

# Health check endpoint
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "total_predictions": sum(prediction_counts.values()),
        "uptime": "API is running"
    }