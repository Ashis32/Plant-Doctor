# Tree-Doctor Project Setup Instructions

This guide provides step-by-step instructions to set up and run the Tree-Doctor project locally. The project consists of two parts:
1.  **Backend (API)**: A FastAPI application serving the machine learning model.
2.  **Frontend**: A Next.js web application for the user interface.

Files are organized as follows:
- `api/`: Contains the Python backend code.
- `frontend/`: Contains the Next.js frontend code.

## Prerequisites

Ensure you have the following installed on your system:
-   **Python 3.9+**: [Download Python](https://www.python.org/downloads/)
-   **Node.js 18+** (LTS recommended): [Download Node.js](https://nodejs.org/)

---

## 1. Backend Setup (API)

The backend runs on port `8000`.

### Step 1.1: Navigate to the API directory
Open your terminal and check that you are in the project root, then move to the `api` folder:

```bash
cd api
```

### Step 1.2: Create a Virtual Environment (Optional but Recommended)
It is good practice to use a virtual environment to manage dependencies.

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 1.3: Install Dependencies
Install the required Python packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Step 1.4: specific for macos (m1/m2/m3)
If you are using a mac with Apple Silicon (M1/M2/M3), you need to install tensorflow-macos and tensorflow-metal using the following command:

```bash
pip install tensorflow-macos tensorflow-metal
```

### Step 1.5: Run the Backend Server
Start the FastAPI server:

```bash
python main.py
```
*Alternatively, you can use uvicorn directly:* `uvicorn main:app --reload`

You should see output indicating the server is running, typically at `http://localhost:8000`.
You can check if it's working by visiting: [http://localhost:8000/ping](http://localhost:8000/ping)

---

## 2. Frontend Setup

The frontend runs on port `3000`.

### Step 2.1: Open a New Terminal
Leave the backend running in the first terminal. Open a **new** terminal window or tab for the frontend.

### Step 2.2: Navigate to the Frontend directory

```bash
cd frontend
```

### Step 2.3: Install Dependencies
Install the Node.js packages:

```bash
npm install
# or if you use yarn:
# yarn install
```

### Step 2.4: Run the Development Server
Start the Next.js development server:

```bash
npm run dev
# or
# yarn dev
```

The application should now be accessible at [http://localhost:3000](http://localhost:3000).

---

## 3. Usage

1.  Open your browser and search [http://localhost:3000](http://localhost:3000).
2.  You will see the Tree-Doctor interface.
3.  Upload an image of a potato leaf.
4.  The frontend will send the image to the backend running on `localhost:8000`.
5.  The backend will process the image (using the model or mock logic if the model is missing) and return the diagnosis.
6.  The result will be displayed on the screen.

**Note:** If the TensorFlow model is not found in `api/models/potato_model.keras`, the backend will run in "Mock Mode" and provide simulated predictions for testing purposes.
