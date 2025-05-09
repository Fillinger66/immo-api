# 🏠 Property Price Prediction API for Belgium market

This project provides an example of web API built with **FastAPI** to serve predictions from a trained **XGBoost** model. It also includes a simple web form where users can input property details and receive an estimated price. The application is containerized with **Docker** for consistent deployment and easy scalability.

🌐 **Live Demo**: [https://immo-predict.onrender.com](https://immo-predict.onrender.com)

#### Note

It will take some times for the application to start.

---

## 🚀 Features

- **FastAPI Backend**: Python web framework for serving machine learning models.
- **XGBoost Model**: Predicts property prices based on structured features for Belgium country.
- **Web Form Interface**: HTML page served by FastAPI to collect user input.
- **Dockerized**: Entire app is deployable in a consistent container environment.
- **Deployed to Render**: Simple cloud deployment using [Render](https://render.com).

---

## 🗂️ Project Structure

├── api.py # Main FastAPI application\
├── Dockerfile # Docker build instructions\
├── docker-compose.yaml # Docker Compose configuration\
├── requirements.txt # Python dependencies\
├── render.yaml # Render deployment configuration\
├── static/ # Static files (HTML, CSS)\
│ ├── index.html # Web form\
│ └── styles.css # Styling for form\
├── model/ # Trained XGBoost model files\
├── pipeline/ # Preprocessing pipelines\
├── lib/ # Utilities\
├── test/ # Unit and integration tests\
└── .gitignore # Git ignore rules\

---

## ⚙️ Setup and Installation

### Prerequisites

- Python 3.7+
- Docker
- A trained XGBoost model
- Preprocessing pipeline files or encoders

### Local Setup

```bash
git clone https://github.com/Fillinger66/immo-api.git
cd immo-api

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

A rained model and a preprocessing pipeline are located in folders (model/, pipeline/).

---
## 🧪 Running Locally

### Option 1: Uvicorn (Development)

```
uvicorn api:app --reload
```

#### Web Interface: http://127.0.0.1:8000/

#### Swagger UI: http://127.0.0.1:8000/docs

### Option 2: Docker (Production-like)
```bash
docker build -t property-prediction-api .
docker run -p 8000:8000 property-prediction-api
```

## ☁️ Deployment to Render
1. Push code to GitHub or push a Docker image to Docker Hub.

2. Sign in to Render.

3. Create a new Web Service.

4. Deploy from Git or Docker image.

5. Make sure it listens on port 8000.

## 📡 API Endpoints

### GET /
#### Serves the HTML web form for property input.

### GET /docs
#### Swagger page for testing the API

### POST /predict/
#### Returns the predicted price based on property features.

## 🔎 Example JSON Input (with value types):
```json
{
  "type": "string",                          // string
  "subtype": "string",                       // string
  "bedroomCount": 0,                         // integer
  "bathroomCount": 0,                        // integer
  "province": "string",                      // string
  "locality": "string",                      // string
  "postCode": 0,                             // integer
  "habitableSurface": 0.0,                   // float
  "buildingCondition": "string",             // string (enum)
  "buildingConstructionYear": 1900,          // integer
  "facedeCount": 0,                          // integer
  "hasLift": false,                          // boolean
  "floodZoneType": "string",                 // string (enum)
  "heatingType": "string",                   // string (enum)
  "hasHeatPump": false,                      // boolean
  "hasPhotovoltaicPanels": false,            // boolean
  "hasThermicPanels": false,                 // boolean
  "kitchenType": "string",                   // string (enum)
  "landSurface": 0.0,                        // float
  "hasGarden": true,                         // boolean
  "gardenSurface": 0.0,                      // float
  "parkingCountIndoor": 0,                   // integer
  "parkingCountOutdoor": 0,                  // integer
  "hasAirConditioning": false,               // boolean
  "hasArmoredDoor": false,                   // boolean
  "hasVisiophone": true,                     // boolean
  "hasOffice": false,                        // boolean
  "toiletCount": 0,                          // integer
  "hasSwimmingPool": false,                  // boolean
  "hasFireplace": false,                     // boolean
  "hasTerrace": true,                        // boolean
  "terraceSurface": 0.0,                     // float
  "terraceOrientation": "string",            // string or 
  "epcScore": "D",                           // string (A-G)
  "cadastralIncome": 0.0,                    // float
  "primaryEnergyConsumptionPerSqm": 0.0,     // float
  "latitude": 0.0 ,                          // float
  "longitude": 0.0,                          // float
  "address": "string"                        // string
}
```

## 📄 License
#### This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).

#### You are free to:

#### ✅ Share — copy and redistribute the material

#### ✅ Adapt — remix, transform, and build upon it

#### Under these conditions:

#### 🔗 Attribution — You must give appropriate credit.

#### 🚫 NonCommercial — You may not use the material for commercial purposes.

### 🔗 [License Details](https://creativecommons.org/licenses/by-nc/4.0/)