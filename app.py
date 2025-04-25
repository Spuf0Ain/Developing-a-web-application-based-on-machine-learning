import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression
import os
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import logging

# Setting up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initializing FastAPI
app = FastAPI()

# Adding CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://localhost:*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mounting the 'data' directory as a static file directory
app.mount("/data", StaticFiles(directory="data"), name="data")

# Defining the input model for predictions
class ApartmentInput(BaseModel):
    rooms: int
    area: float
    floor: int
    total_floors: int
    residential_complex: int
    district: str

# Loading and preprocessing data
def preprocess_data():
    # File names and corresponding district names
    file_names = ['alatau.csv', 'almaly.csv', 'auezov.csv', 'bostandik.csv', 'zhetisu.csv', 'medeu.csv']
    district_names = ['Alatau', 'Almaly', 'Auezov', 'Bostandyk', 'Zhetysu', 'Medeu']
    dfs = []
    
    for file, district in zip(file_names, district_names):
        file_path = os.path.join('data', file)
        if os.path.exists(file_path):
            logger.info(f"Loading file: {file_path}")
            df = pd.read_csv(file_path)
            df['district'] = district
            dfs.append(df)
        else:
            logger.warning(f"File not found: {file_path}")
    
    # Combining datasets
    if not dfs:
        raise FileNotFoundError("No CSV files found in the data directory.")
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Cleaning data
    try:
        # Replacing non-breaking spaces (\xa0) and other unwanted characters
        combined_df['price'] = combined_df['price'].str.replace('〒', '', regex=False)\
                                                  .str.replace('\xa0', '', regex=False)\
                                                  .str.replace(' ', '', regex=False)\
                                                  .str.strip()
        # Converting to float and handling errors
        combined_df['price'] = pd.to_numeric(combined_df['price'], errors='coerce')
        
        combined_df['area'] = combined_df['area'].str.replace(' м²', '', regex=False).astype(float)
        combined_df['rooms'] = combined_df['rooms'].str.extract(r'(\d)')[0].astype(int)
        combined_df['residential_complex'] = combined_df['residential_complex'].map({'Да': 1, 'Нет': 0})
        
        # Logging rows with invalid prices
        invalid_prices = combined_df[combined_df['price'].isna()]
        if not invalid_prices.empty:
            logger.warning(f"Found {len(invalid_prices)} rows with invalid prices:\n{invalid_prices}")
        
        # Filtering invalid rows
        combined_df = combined_df.dropna()
        combined_df = combined_df[(combined_df['price'] > 0) & (combined_df['area'] > 0) & 
                                 (combined_df['rooms'] > 0) & (combined_df['floor'] > 0) & 
                                 (combined_df['total_floors'] > 0)]
        
        if combined_df.empty:
            raise ValueError("No valid data remains after preprocessing.")
        
        return combined_df
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        raise

# Training and saving the model
def train_model():
    df = preprocess_data()
    features = ['rooms', 'area', 'floor', 'total_floors', 'residential_complex', 'district']
    X = pd.get_dummies(df[features], columns=['district'], drop_first=True)
    y = df['price']
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Saving the model
    joblib.dump(model, 'model.joblib')
    logger.info("Model trained and saved as model.joblib")
    return model, X.columns

# Loading the model and feature columns
model = None
feature_cols = None

# Always retrain the model to ensure consistency with the current dataset
logger.info("Training new model...")
model, feature_cols = train_model()

# Defining the prediction endpoint
@app.post("/predict")
async def predict(input: ApartmentInput):
    try:
        # Preparing input data
        input_dict = {
            'rooms': input.rooms,
            'area': input.area,
            'floor': input.floor,
            'total_floors': input.total_floors,
            'residential_complex': input.residential_complex,
            'district': input.district
        }
        input_df = pd.DataFrame([input_dict])
        input_encoded = pd.get_dummies(input_df, columns=['district'], drop_first=True)
        
        # Aligning columns with training data
        for col in feature_cols:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[feature_cols]
        
        # Making prediction
        prediction = model.predict(input_encoded)[0]
        
        # Ensuring positive prediction
        if prediction < 0:
            prediction = 0
            
        return {"prediction": round(prediction, 2)}
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Running the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)