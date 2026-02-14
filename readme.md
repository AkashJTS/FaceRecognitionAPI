# 🚀 FaceRecognitionAPI

A FastAPI-based Face Recognition & Comparison API using InsightFace.

---

## 📦 1️⃣ Clone the Repository

```bash
git clone https://github.com/sathish-1507/FaceRecognitionAPI.git
cd FaceRecognitionAPI
🐍 2️⃣ Create Virtual Environment (Recommended)
python -m venv venv
▶ Activate Environment
Windows

venv\Scripts\activate
Mac/Linux

source venv/bin/activate
📥 3️⃣ Install Required Packages
pip install --upgrade pip
pip install fastapi uvicorn insightface opencv-python numpy python-multipart python-dotenv
🔐 4️⃣ Create .env File
Inside project root folder create a file named:

.env
Add this inside it:

AUTH_TOKEN=your-secret-token
🚀 5️⃣ Start the Server
python app.py
OR

uvicorn app:app --host 0.0.0.0 --port 8000 --reload
🌐 6️⃣ Open Swagger UI
After server starts, open:

http://127.0.0.1:8000/docs
✅ Server is now running successfully.