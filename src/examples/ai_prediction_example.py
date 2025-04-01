from src.analyzers.ai_price_predictor import AIPricePredictor
import pandas as pd

def main():
    # Initialize the predictor with a stock symbol (e.g., VNM for Vinamilk)
    symbol = "VNM"
    print(f"\nInitializing AI Price Predictor for {symbol}...")
    predictor = AIPricePredictor(symbol)
    
    # Train the model
    print("\nTraining the model...")
    metrics = predictor.train()
    print(f"Training R² score: {metrics['train_score']:.4f}")
    print(f"Testing R² score: {metrics['test_score']:.4f}")
    
    # Make a prediction
    print("\nMaking prediction...")
    prediction = predictor.predict()
    print("\nPrediction Results:")
    print(f"Trend: {prediction['trend']}")
    print(f"Predicted Return: {prediction['predicted_return']:.4f}")
    print(f"Confidence: {prediction['confidence']:.4f}")
    print(f"Strength: {prediction['strength']}")
    
    # Get feature importance
    print("\nTop 5 Most Important Features:")
    importance = predictor.get_feature_importance()
    for feature, score in list(importance.items())[:5]:
        print(f"{feature}: {score:.4f}")

if __name__ == "__main__":
    main() 