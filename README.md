# Apartment Price Prediction

## Overview
This project is a web application that predicts apartment prices in Almaty, Kazakhstan, based on features such as the number of rooms, area, floor, total floors, residential complex status, and district. The application uses a Linear Regression model trained on real estate data and provides interactive visualizations to explore price trends across districts, areas, and floors.

The backend is built with FastAPI (Python), and the frontend is a single-page React application using Recharts for visualizations. The dataset consists of CSV files for six districts in Almaty: Alatau, Almaly, Auezov, Bostandyk, Zhetysu, and Medeu.

## Features
- **Price Prediction**: Enter apartment details (rooms, area, floor, total floors, residential complex, district) to get a predicted price.
- **Data Visualizations**:
  - Scatter plot: Price vs. Area by number of rooms, filterable by district.
  - Bar chart: Average price by district.
  - Line chart: Average price by floor, filterable by district.
- **Dataset Summary**: Displays total apartments, average price, average area, and an interesting fact about price variations across districts.

## Technologies Used
- **Backend**: FastAPI, Python, pandas, scikit-learn (LinearRegression), joblib
- **Frontend**: React, Recharts, Tailwind CSS, Papa Parse (for CSV parsing)
- **Data**: CSV files containing apartment data for six districts

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Node.js (optional, for development)
- A modern web browser (e.g., Chrome, Firefox)

### Installation
1. **Clone the Repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Set Up a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python Dependencies**:
   Install the required packages listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
   If you don’t have a `requirements.txt`, install the following:
   ```bash
   pip install fastapi uvicorn pandas scikit-learn joblib
   ```

4. **Prepare the Data**:
   - Ensure the `data` directory contains the following CSV files:
     - `alatau.csv`
     - `almaly.csv`
     - `auezov.csv`
     - `bostandik.csv`
     - `zhetisu.csv`
     - `medeu.csv`
   - Each CSV file should have the following columns: `price`, `rooms`, `area`, `floor`, `total_floors`, `residential_complex`. Example format:
     ```
     price,rooms,area,floor,total_floors,residential_complex
     34000000〒,2-комнатная,60 м²,5,12,Да
     ```

5. **Run the FastAPI Backend**:
   - Start the FastAPI server:
     ```bash
     uvicorn app:app --reload --port 8000
     ```
   - The server will be available at `http://localhost:8000`.

6. **Serve the Frontend**:
   - Use Python’s built-in HTTP server to serve the frontend:
     ```bash
     python -m http.server 8080
     ```
   - Open your browser and navigate to `http://localhost:8080/apartment_price_prediction.html`.

## Usage
1. **Make a Prediction**:
   - On the webpage, fill in the apartment details (district, rooms, area, floor, total floors, residential complex).
   - Click "Predict Price" to see the predicted price in millions of tenge (〒).

2. **Explore Visualizations**:
   - Use the "Filter Visualizations by District" dropdown to filter the scatter plot (Price vs. Area by Rooms) and line chart (Average Price by Floor) by district.
   - The bar chart (Average Price by District) shows data for all districts.

3. **View Dataset Summary**:
   - The "Dataset Summary" section displays the total number of apartments, average price, average area, and an interesting fact about price variations across districts.

## File Structure
- `app.py`: FastAPI backend script that handles data preprocessing, model training, and prediction API.
- `apartment_price_prediction.html`: Frontend HTML file containing React code for the UI and visualizations.
- `data/`: Directory containing CSV files for each district:
  - `alatau.csv`
  - `almaly.csv`
  - `auezov.csv`
  - `bostandik.csv`
  - `zhetisu.csv`
  - `medeu.csv`
- `model.joblib`: Trained Linear Regression model (automatically generated when the backend starts).

## Troubleshooting
- **Prediction Fails with Feature Mismatch Error**:
  - If you see an error like "Feature names unseen at fit time" (e.g., `district_Zhetysu` not found), it means the model was trained with a different set of districts.
  - Delete `model.joblib` and restart the FastAPI server to retrain the model with the current dataset:
    ```bash
    rm model.joblib
    uvicorn app:app --reload --port 8000
    ```

- **Scatter Plot X-Axis Labels Overlap**:
  - The scatter plot’s x-axis (Area) is configured to show ticks every 10 m², with labels rotated 45 degrees for readability. If labels still overlap, ensure your browser window is wide enough (at least 1000px), or adjust the `tickInterval` in the `ScatterPlot` component in `apartment_price_prediction.html`.

- **No Data in Visualizations**:
  - Check the browser console (F12 → Console) for logs like `ScatterPlot - Selected District: X, Filtered Rows: 0`. If the filtered rows are 0, ensure the district names match between the CSV files and the code.
  - Verify that the CSV files are correctly formatted and located in the `data` directory.

- **Data Points Look Clustered**:
  - The scatter plot uses transparency (`fillOpacity={0.6}`) to handle overlapping points. If the data is still too dense, consider adding stricter filters in `fetchCSV` (e.g., `area` between 10 and 100 m²) to remove outliers.

## Notes
- The application assumes the CSV files follow a specific format. If the format changes (e.g., different column names or data types), you may need to update the preprocessing logic in `app.py` and `apartment_price_prediction.html`.
- The Linear Regression model is retrained every time the FastAPI server starts to ensure consistency with the current dataset.
- The scatter plot’s x-axis is dynamically scaled to the min and max area values in the data, with a minimum chart width of 1000px to ensure readability.

## Future Improvements
- Add more robust data validation and outlier removal to improve prediction accuracy.
- Implement a hexbin plot or jittering for the scatter plot to better handle dense data.
- Add user authentication and a database to store prediction history.
- Allow users to upload their own CSV files for custom predictions.

## License
This project is for educational purposes and does not include a specific license. Please contact the author for usage permissions.