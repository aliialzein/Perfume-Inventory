AI-Powered Inventory & Product Recognition System Built for Speed, Automation & Accuracy

This POS is an intelligent, end-to-end inventory management system that eliminates manual data entry through real-time AI predictions. Users simply upload a product image, and the system automatically identifies the item, predicts its price, stores the data, and updates inventory quantities — all in a seamless workflow.

Built from scratch with custom AI models, a clean React interface, and a FastAPI backend, Korban POS is designed for clarity, efficiency, and automation.

🎯 What POS Does

Automatic Product Recognition — Upload a product image and the AI identifies brand, model, and gender.

Market Price Prediction — A machine learning regression model instantly estimates a realistic product price.

AI + Database Integration — Images stored in MongoDB GridFS, with live product updates.

Real-Time Inventory Management — Quantities update as items are added or modified.

Fast, Clean React Interface — Search, filter, sort, and manage inventory with speed and clarity.

🧠 AI Components
1. Image Classification Model (EfficientNetB0)

Custom CNN + transfer learning pipeline.

Identifies perfume brand, model, and gender directly from the uploaded image.

Enhanced with augmentation, preprocessing, and class balancing.

2. Price Prediction Model (Random Forest Regression)

Trained on structured perfume metadata.

Provides an estimated market price that users can modify before saving.

3. Real-Time AI Predictions

Both models run inside the FastAPI backend for smooth, low-latency predictions.

💻 Tech Stack Overview
Frontend (React.js)

Clean, minimal UI

Search & filtering (brand, model, gender, price)

Sortable price column

Fully responsive

Tight integration with the FastAPI backend

Backend (FastAPI + MongoDB)

REST API for classification, price estimation, CRUD operations

GridFS for efficient image storage

Live inventory updates with UUID product IDs

Database (MongoDB + GridFS)

Each item document includes:

product_id (UUID)

Image stored in GridFS

Predicted price

Adjusted price

Quantity

Date added

📦 Installation & Usage (For End Users)

Follow these steps if you want to run the system without modifying the code.

1. Clone the repository
git clone https://github.com/aliialzein/Perfume-Inventory.git
cd Perfume-Inventory

2. Start the backend (FastAPI)

Install Python dependencies:

pip install -r requirements.txt

requirment.txt:absl-py==2.3.1
annotated-doc==0.0.4
annotated-types==0.7.0
anyio==4.12.0
astunparse==1.6.3
certifi==2025.11.12
charset-normalizer==3.4.4
click==8.3.1
colorama==0.4.6
dnspython==2.8.0
fastapi==0.123.0
flatbuffers==25.9.23
gast==0.7.0
google-pasta==0.2.0
grpcio==1.76.0
h11==0.16.0
h5py==3.15.1
httptools==0.7.1
idna==3.11
joblib==1.5.2
keras==3.12.0
libclang==18.1.1
Markdown==3.10
markdown-it-py==4.0.0
MarkupSafe==3.0.3
mdurl==0.1.2
ml_dtypes==0.5.4
namex==0.1.0
numpy==2.3.5
opt_einsum==3.4.0
optree==0.18.0
packaging==25.0
pandas==2.3.3
pillow==12.0.0
protobuf==6.33.1
pydantic==2.12.5
pydantic_core==2.41.5
Pygments==2.19.2
pymongo==4.15.4
python-dateutil==2.9.0.post0
python-dotenv==1.2.1
python-multipart==0.0.20
pytz==2025.2
PyYAML==6.0.3
requests==2.32.5
rich==14.2.0
scikit-learn==1.6.1
scipy==1.16.3
setuptools==80.9.0
six==1.17.0
starlette==0.50.0
tensorboard==2.20.0
tensorboard-data-server==0.7.2
tensorflow==2.20.0
termcolor==3.2.0
threadpoolctl==3.6.0
typing-inspection==0.4.2
typing_extensions==4.15.0
tzdata==2025.2
urllib3==2.5.0
uvicorn==0.38.0
watchfiles==1.1.1
websockets==15.0.1
Werkzeug==3.1.4
wheel==0.45.1
wrapt==2.0.1



Run the API:

uvicorn app.main:app --reload

3. Start the frontend (React)
cd frontend
npm install
npm start

4. Access the system

Open:

http://localhost:3000


Upload a product → Get predictions → Save to inventory.

🛠️ Installation & Usage (For Contributors)

If you want to work on the project or extend it:

1. Clone the repo
git clone https://github.com/aliialzein/Perfume-Inventory.git
cd Perfume-Inventory

2. Backend Setup

Install Python 3.2 and run:

pip install -r requirements.txt
uvicorn app.main:app --reload


Configure environment variables for:

MongoDB connection string

GridFS parameters

Model file paths

3. Frontend Setup
cd frontend
npm install
npm start

4. AI Model Setup

Ensure the following model files are placed correctly:

\Backend\ai_models\Perfumes_brand.h5

\Backend\ai_models\Perfume_NameGender_final.keras

\Backend\ai_models\perfume_price_model.pkl

🤝 Contributor Guidelines

We welcome improvements! Please follow these expectations:

Open an issue before major feature work

Create a new branch for every feature

Write clear commit messages (squash if appropriate)

Ensure backend + frontend code stays consistent

Update the documentation if your feature changes user behavior

Submit a PR using the project’s pull request template


🎯 Project Goal

To eliminate manual data entry and create a workflow where inventory management becomes:

Take a photo → AI recognizes it → Predict price → Save → Done.

Simple, fast, and intelligent.
