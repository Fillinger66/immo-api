// script.js

function initializeApp() {
    console.log("initializeApp function called.");
    // Get reference to the loading popup
    const loadingPopup = document.getElementById('loadingPopup');

    // ---- TomSelect Initialization ----
    const localityElement = document.getElementById("locality");
    if (localityElement) {
        new TomSelect(localityElement, {
            create: false,
            sortField: {
                field: "text",
                direction: "asc"
            }
        });
        console.log("TomSelect initialized for #locality");
    } else {
        console.warn("Element with ID 'locality' not found for TomSelect initialization.");
    }

    const postCodeElement = document.getElementById("postCode");
    if (postCodeElement) {
        new TomSelect(postCodeElement, {
            create: false,
            sortField: {
                field: "text",
                direction: "asc"
            }
        });
        console.log("TomSelect initialized for #postCode");
    } else {
        console.warn("Element with ID 'postCode' not found for TomSelect initialization.");
    }

    // ---- Prediction Form Logic ----
    const predictionForm = document.getElementById('predictionForm');
    if (predictionForm) {
        predictionForm.addEventListener('submit', async function(event) {
            event.preventDefault();

             // Show loading popup
            if (loadingPopup) {
                loadingPopup.classList.remove('hidden');
            }

            const form = event.target;
            const formData = new FormData(form);
            const jsonData = {};

            for (const [key, value] of formData.entries()) {
                if (['bedroomCount', 'bathroomCount', 'postCode', 'buildingConstructionYear',
                     'facedeCount', 'parkingCountIndoor', 'parkingCountOutdoor',
                     'toiletCount', 'cadastralIncome', 'primaryEnergyConsumptionPerSqm'].includes(key)) {
                    jsonData[key] = value ? parseInt(value) : null;
                } else if (['habitableSurface', 'landSurface', 'gardenSurface', 'terraceSurface',
                            'latitude', 'longitude'].includes(key)) {
                    jsonData[key] = value ? parseFloat(value) : null;
                } else if (['hasLift', 'hasHeatPump', 'hasPhotovoltaicPanels', 'hasThermicPanels',
                          'hasGarden', 'hasAirConditioning', 'hasArmoredDoor', 'hasVisiophone',
                          'hasOffice', 'hasSwimmingPool', 'hasFireplace', 'hasTerrace'].includes(key)) {
                    jsonData[key] = (value === 'true' || value === 'on');
                } else {
                    jsonData[key] = value || null;
                }
            }

            const booleanFields = ['hasLift', 'hasHeatPump', 'hasPhotovoltaicPanels', 'hasThermicPanels',
                                   'hasGarden', 'hasAirConditioning', 'hasArmoredDoor', 'hasVisiophone',
                                   'hasOffice', 'hasSwimmingPool', 'hasFireplace', 'hasTerrace'];
            booleanFields.forEach(field => {
                if (jsonData[field] === undefined) {
                    jsonData[field] = false;
                }
            });

            console.log("Sending JSON data:", jsonData);
            const MAE = 46000;
            const predictionResultDiv = document.getElementById('predictionResult');

            if (!predictionResultDiv) {
                console.error("Element with ID 'predictionResult' not found.");
                return;
            }

            predictionResultDiv.classList.add('hidden');
            predictionResultDiv.innerHTML = "";

            try {
                const response = await fetch('/predict/', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json; charset=UTF-8',
                        'accept': 'application/json'
                    },
                    body: JSON.stringify(jsonData)
                });

                if (!response.ok) {
                    let errorData;
                    try {
                        errorData = await response.json();
                    } catch (e) {
                        errorData = { detail: "Could not parse error response." };
                    }
                    throw new Error(`API error: ${response.status} ${response.statusText} - ${JSON.stringify(errorData.detail || errorData)}`);
                }

                const result = await response.json();
                console.log("Received prediction:", result);

                let selectedLocality = "";
                if (localityElement && localityElement.value) {
                    selectedLocality = "in " + localityElement.value;
                }

                if (result && result.prediction !== undefined) {
                    const predictedPrice = Array.isArray(result.prediction) ? result.prediction[0] : result.prediction;
                    predictionResultDiv.innerHTML = `Estimate Price: <strong>${predictedPrice.toFixed(0)} &euro;</strong><br>
                    <div id="predictionText">Our model's average prediction error (Mean Absolute Error - MAE) across the Belgian housing market is approximately ${MAE}&euro;.<br><br>
                    While our estimated price is ${predictedPrice.toFixed(0)} &euro;, the actual selling price could reasonably be expected to fall within a range of approximately <br><br><span style=" font-size: 1.2em;"><center><b> ${Math.max(0, parseFloat(predictedPrice.toFixed(0))-MAE)}&euro;</b> to <b>${parseFloat(predictedPrice.toFixed(0))+MAE}&euro;</center></b></span><br>
                    While our model provides an estimate, the actual price ${selectedLocality} could vary more significantly due to the characteristics of the local market compared to the national average.<br><br>
                    We recommend considering these regional differences and potentially consulting local real estate experts for a more precise understanding of the ${selectedLocality} market.</div>`;
                    predictionResultDiv.classList.remove('hidden');
                    predictionResultDiv.style.backgroundColor = '#d4edda';
                    predictionResultDiv.style.borderColor = '#c3e6cb';
                    predictionResultDiv.style.color = '#155724';
                } else {
                    predictionResultDiv.innerHTML = `Prediction received, but format is unexpected: ${JSON.stringify(result)}`;
                    predictionResultDiv.classList.remove('hidden');
                    predictionResultDiv.style.backgroundColor = '#fff3cd';
                    predictionResultDiv.style.borderColor = '#ffeeba';
                    predictionResultDiv.style.color = '#856404';
                }

            } catch (error) {
                console.error("Error fetching prediction:", error);
                predictionResultDiv.innerHTML = `Error getting prediction: ${error.message}`;
                predictionResultDiv.classList.remove('hidden');
                predictionResultDiv.style.backgroundColor = '#f8d7da';
                predictionResultDiv.style.borderColor = '#f5c6cb';
                predictionResultDiv.style.color = '#721c24';
            }
            finally {
                // ALWAYS hide loading popup when done (success or error)
                if (loadingPopup) {
                    loadingPopup.classList.add('hidden');
                }
            }
        });
    } else {
        console.warn("Element with ID 'predictionForm' not found. Prediction script not fully initialized.");
    }
}