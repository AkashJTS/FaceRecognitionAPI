🚀 FaceRecognitionAPI
<p align="center"> <b>FastAPI-based Face Recognition REST API using InsightFace</b><br> Supports Image Upload + Base64 + Bearer Token Authentication </p>
📌 Features

✅ Upload Reference Face Image
✅ Compare Two Faces Against Reference
✅ Supports:

🖼️ Form-Data Image Upload (JPEG / PNG)

🔐 Bearer Token Authentication

📦 Base64 Image Support

🎯 Adjustable Match Threshold
✅ Cosine Similarity Face Matching
✅ 8MB File Size Limit
✅ Environment-based Secret Token (.env)

🛠 Tech Stack

⚡ FastAPI

🧠 InsightFace (buffalo_l model)

🖼 OpenCV

🔢 NumPy

🔐 python-dotenv

🚀 Uvicorn

⚙️ Setup Guide (Step-by-Step)
1️⃣ Clone Repository
git clone https://github.com/sathish-1507/FaceRecognitionAPI.git
cd FaceRecognitionAPI

2️⃣ Create Virtual Environment (Recommended)
Windows
python -m venv venv
venv\Scripts\activate

Linux / macOS
python3 -m venv venv
source venv/bin/activate

3️⃣ Install Dependencies
pip install fastapi uvicorn numpy opencv-python-headless insightface python-dotenv


Or if you have requirements.txt:

pip install -r requirements.txt

4️⃣ Create .env File

In the root folder create a file named:

.env


Add:

AUTH_TOKEN=your-secret-token


Example:

AUTH_TOKEN=my-secure-api-token

🚀 Start the Server
python app.py


Server will start at:

http://localhost:8000


Swagger UI available at:

http://localhost:8000/docs

🔐 Authentication

All APIs require Bearer Token in header:

Authorization: Bearer your-secret-token


If token is invalid or missing → 401 Unauthorized

📡 API Endpoints
🖼 1. Upload Reference Image
Endpoint:
POST /uploadImage

Headers:
Authorization: Bearer your-secret-token

Body (form-data):
Key	Type
reference	File
Success Response:
{
  "status": true,
  "message": "Reference uploaded successfully"
}

👤 2. Compare Faces
Endpoint:
POST /employeeFaceCompare

Headers:
Authorization: Bearer your-secret-token

Option A — Form Data (File Upload)
Key	Type
file1	File
file2	File
threshold	Number (optional, default 70)
Option B — Base64 JSON
{
  "file1_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD...",
  "file2_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD...",
  "threshold": 80
}

Response Example
{
  "status": true,
  "message": "Face recognition successful",
  "data": {
    "face_match": true,
    "match_score": 92.07
  }
}

📏 Rules & Validations
Rule	Value
Allowed Formats	JPEG, PNG
Max File Size	8MB
Default Threshold	70%
Timeout	60 seconds
🧠 How Matching Works

Face detected using InsightFace

Face embedding generated

Cosine similarity calculated

Score converted to percentage

Compared with threshold

🛑 Common Errors
401 Unauthorized

Missing Bearer token

Wrong token

400 Bad Request

No face detected

Invalid image format

File size exceeds 8MB

📂 Project Structure
FaceRecognitionAPI/
│
├── app.py
├── .env
├── temp/
└── README.md

⭐ Example Postman Header
Key: Authorization
Value: Bearer my-secure-api-token

🏁 You're Ready!

Your API is now:

🔐 Secure

⚡ Fast

🧠 AI-powered

🖼 Multi-format compatible

If you want, I can also:

✅ Create a clean requirements.txt

✅ Add GitHub badges

✅ Add Docker support

✅ Add Deployment guide (AWS / Render / Railway)

Just tell me 😄