import uvicorn
import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import html

from pydantic import BaseModel
from typing import Optional
import pandas as pd
from lib.model.XGBoostModel import XGBoostModel
from lib.DataPipeline import DataPipeline
from lib.DataManager import DataManager
from geopy.geocoders import Nominatim

# XGBoost model file path
MODEL_PATH = 'model/xgb_model.model'

# --- Load the trained XGBoost model ---
try:
    xgbmodel = XGBoostModel(verbose=1)
    xgbmodel.load_model(path_to_model=MODEL_PATH)
    print(f"XGBoost model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading XGBoost model: {e}")
    xgbmodel = None # Set model to None if loading fails




class PropertyFeatures(BaseModel):
    """
    Represents the features of a property for prediction.
    Matches the structure of the incoming JSON request.
    """
    type: str
    subtype: str
    bedroomCount: Optional[int] = 1
    bathroomCount: Optional[int] = 1
    province: str
    locality: str
    postCode: int
    habitableSurface: Optional[float] = None 
    buildingCondition: Optional[str] = None
    buildingConstructionYear: Optional[int] = None
    facedeCount: Optional[int] = 1
    hasLift: Optional[bool] = False
    floodZoneType: Optional[str] = "NO_FLOOD_ZONE"
    heatingType: Optional[str] = None
    hasHeatPump: Optional[bool] = False
    hasPhotovoltaicPanels: Optional[bool] = False
    hasThermicPanels: Optional[bool] = False
    kitchenType: Optional[str] = None
    landSurface: Optional[float] = -1 # Using float
    hasGarden: Optional[bool] = False
    gardenSurface: Optional[float] = 0 # Using float
    parkingCountIndoor: Optional[int] = 0
    parkingCountOutdoor: Optional[int] = 0
    hasAirConditioning: Optional[bool] = False
    hasArmoredDoor: Optional[bool] = False
    hasVisiophone: Optional[bool] = False
    hasOffice: Optional[bool] = False
    toiletCount: Optional[int] = 1
    hasSwimmingPool: Optional[bool] = False
    hasFireplace: Optional[bool] = False
    hasTerrace: Optional[bool] = False
    terraceSurface: Optional[float] = 0 
    terraceOrientation: Optional[str] = None 
    epcScore: Optional[str] = None
    cadastralIncome: Optional[int] = None
    primaryEnergyConsumptionPerSqm: Optional[int] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    address: Optional[str] = None

# --- Create a FastAPI application instance ---
app = FastAPI(
    title="XGBoost Model API",
    description="API for making predictions using a trained XGBoost model",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to your frontend URL
    allow_methods=["*"],
    allow_headers=["*"],
)
# The first argument "/static" is the URL path where the static files will be served.
# The second argument "directory='static'" tells FastAPI to look inside the local 'static' folder.
# The third argument "name='static'" is an internal name for this mounted directory.
app.mount("/static", StaticFiles(directory="static"), name="static")

#-----------------------------------------#
#-------------- FUNCTIONS ----------------#
#-----------------------------------------#
def create_dataframe(data:dict):
    """
    Creates a pandas DataFrame from a dictionary of property data.

    This function takes a dictionary containing property details, unescapes
    HTML entities in the 'province' and 'locality' fields, prints the
    cleaned data (for debugging/logging purposes), defines the expected
    column order for the DataFrame, and then creates a single-row DataFrame
    from the processed data dictionary.

    Args:
        data (dict): A dictionary where keys are column names and values
                     are the corresponding data for a single property.

    Returns:
        pd.DataFrame: A pandas DataFrame with a single row containing the
                      processed property data, with columns in the specified
                      order.
    """
    data["province"] = html.unescape(data["province"])
    data["locality"] = html.unescape(data["locality"])

    print("predict::create_dataframe:: -> data:")
    print(data)

    columns = ['type', 'subtype', 'bedroomCount', 'bathroomCount', 'province',
       'locality', 'postCode', 'habitableSurface', 'buildingCondition',
       'buildingConstructionYear', 'facedeCount', 'hasLift', 'floodZoneType',
       'heatingType', 'hasHeatPump', 'hasPhotovoltaicPanels',
       'hasThermicPanels', 'kitchenType', 'landSurface', 'hasGarden',
       'gardenSurface', 'parkingCountIndoor', 'parkingCountOutdoor',
       'hasAirConditioning', 'hasArmoredDoor', 'hasVisiophone', 'hasOffice',
       'toiletCount', 'hasSwimmingPool', 'hasFireplace', 'hasTerrace',
       'terraceSurface', 'terraceOrientation', 'epcScore', 'cadastralIncome',
       'primaryEnergyConsumptionPerSqm', 'latitude', 'longitude']
    
    df = pd.DataFrame([data],columns=columns)
    return df

def load_html_page(page_name:str):

    html_content = None
    # Construct the full path to the index.html file within the 'static' directory
    html_file_path = os.path.join("static", page_name)

    # Check if the file exists
    if not os.path.exists(html_file_path):
        # Return a 404 error if the file is not found
        return HTMLResponse(content=f"<h1>Error: {load_html_page} not found in the 'static' directory!</h1>", status_code=404)

    # Read the content of the HTML file and return it as an HTMLResponse
    with open(html_file_path, "r") as f:
        html_content = f.read()


    return HTMLResponse(content=html_content)

def get_coordinates_belgium(address):
    """
    Retrieves the latitude and longitude for a given address in Belgium using Nominatim.

    Args:
        address (str): The address to geocode.

    Returns:
        tuple or None: A tuple containing (latitude, longitude) if found, otherwise None.
    """
    geolocator = Nominatim(user_agent="my_belgium_geocoder")
    try:
        location = geolocator.geocode(address)
        if location:
            return location.latitude, location.longitude
        else:
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def get_lat_lng_for_prediction(df:pd.DataFrame, input_dict:dict):
    """
    Retrieves the latitude and longitude for a given address in Belgium using Nominatim.
    It will apply lat/lng from the full address or will put zipCode lat/lng instead if error
    or no address

    Args:
        df (DataFrame): The dataframe where lat/lng will be put in
        input_dict (dict) : the dictinnary of the values send by the user

    Returns:
       None
    """
    try:
        # Get the address to geocode it
        address = input_dict["address"]
        # Address is not useful for the model
        input_dict.pop("address")
        # Manage to get latitude/longitude from address if it'ss mentionned
        # else we will get latitude/longitude from zipCode
        if address != None and input_dict["locality"] !=None and input_dict["postCode"] !=None:
            print("predict::data -> Address : "+address+"")
            # address format : Grand Place 1, 1000 Brussels, Belgium
            complete_address = address+","+str(input_dict["postCode"])+" "+input_dict["locality"]+" Belgium"
            print("Address : "+complete_address+"")
            coordinates = get_coordinates_belgium(complete_address)
            if coordinates:
                lat, lng = coordinates
                df["latitude"] = lat
                df["longitude"] = lng
                print(f"predict::data -> Coordinates for '{complete_address}': Latitude = {lat}, Longitude = {lng}")
            else:
                df["latitude"] = df["latitude"].fillna(df["zipcode_Latitude"])
                df["longitude"] = df["longitude"].fillna(df["zipcode_Longitude"])
            
        else:
            # fill missing lat/lng with zipcode lat/lng
            df["latitude"] = df["latitude"].fillna(df["zipcode_Latitude"])
            df["longitude"] = df["longitude"].fillna(df["zipcode_Longitude"])
    except Exception as ex:
        print(f"predict::data:: Exception geocoder -> {ex}")
        df["latitude"] = df["latitude"].fillna(df["zipcode_Latitude"])
        df["longitude"] = df["longitude"].fillna(df["zipcode_Longitude"])


#-----------------------------------------#
#----------------- ROUTES ----------------#
#-----------------------------------------#

# --- Root Endpoint to Serve the HTML Form ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """
    Serves the main HTML page with the prediction form.
    """
    return load_html_page("index.html")
   

@app.get("/about.html", response_class=HTMLResponse)
async def read_about():
    """
    Serves the about HTML page with the project description and author informations.
    """
    return load_html_page("about.html")
   

# --- Define the prediction endpoint ---
@app.post("/predict/")
async def predict(data: PropertyFeatures):
    """
    Receives input data, makes a prediction using the pretrained XGBoost model,
    and returns the prediction as a list.
    """
    if xgbmodel is None:
        return {"error": "Model not loaded. Cannot make predictions."}

    try:
        # --- Prepare the input data for the model ---
        print("predict::data:")
        print(data)
        # convert Pydantic basemodel to a python dict
        input_dict = data.model_dump()

        #Create the dataframe to be process by pipeline
        df = create_dataframe(input_dict)

        # Get lat/lng from the zip code with Nominatim
        df = DataManager.get_lat_lng_for_zipcode(df=df,verbose=1)

        # Try getting lat/lng from the address if exist with Nominatim too
        get_lat_lng_for_prediction(df, input_dict)
        
        # --- Adjust display settings to show all columns ---
        pd.set_option('display.max_columns', None)  # No limit on the number of columns to display
        pd.set_option('display.width', None)        # No limit on the width of the display
        pd.set_option('display.max_colwidth', None) # No truncation of column content
        # --- print the list of columns of dataframe ---
        print(f"columns : {df.columns.to_list}")
        # --- print the head of the dataframe ---
        print(f"Head : {df.head()}")

        print("Pipeline creation")
        # --- Load the pipeline from the file ---
        pipeline = DataPipeline(df=df,target_columns_name="price",path_to_save_pipeline="pipeline/xgboost_pipeline.pipeline")
        # --- Use the pipeline to prepare data for the model ---
        test_data = pipeline.prepare_data_for_prediction()

        # --- Make the prediction ---
        prediction = xgbmodel.predict(test_data)
        print(f"Prediction : {prediction}")

        # --- Return the raw prediction(s) as a list (to ensure JSON serializability) ---
        return {"prediction": prediction.tolist()}

    except Exception as e:
        # --- Log the error for debugging ---
        print(f"An error occurred during prediction: {e}")
        return {"error": f"An internal server error occurred: {e}"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port)

# --- How to run the application ---
# Run the application from your terminal using:
# uvicorn api:app --reload
#
# The '--reload' flag is useful during development as it restarts the server
# whenever you make changes to the code.
#
# Once running, you can access the interactive API documentation (Swagger UI) at:
# http://127.0.0.1:8000/docs

