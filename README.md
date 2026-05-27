# README

## Project Overview

This project consists of:

- **Backend** — Python-based API service
- **Frontend** — React.js web application
- **Firebase Firestore** — Database and vector storage

---

# Prerequisites

Before starting the project, make sure the following tools are installed on your system.

## Required Tools

### Backend Requirements

- Python 3.10+
- pip
- virtualenv (included with Python)

Download Python:  
https://www.python.org/downloads/

---

### Frontend Requirements

- Node.js
- npm

Download Node.js:  
https://nodejs.org/

---

### Firebase & Google Cloud

You also need:

- A Firebase project
- Firestore enabled

---

### Tesseract OCR EXE

You also need:

- install executable file from https://github.com/UB-Mannheim/tesseract/wiki
- Setup and install on your device

# Windows PowerShell Policy Fix (React.js)

If you are using Windows, PowerShell may block npm scripts.

Run the following command in **PowerShell as Administrator** before using npm:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

# Environment Variables Setup

Both frontend and backend require `.env` files.

1. Locate the provided:

```bash
sample.env
```

2. Create your own `.env` file based on it.

3. Fill in all required environment variables.

---

# Firebase Setup

To allow the system to connect to your Firebase project:

1. Go to Firebase Console:

https://console.firebase.google.com/

2. Select your Firebase project

3. Navigate to:

```text
Project Settings -> Service Accounts
```

4. Click:

```text
Generate New Private Key
```

5. A JSON file will be downloaded

6. Place the downloaded JSON file inside:

```text
./backend
```

7. Rename the file to:

```text
firebase-key.json
```

---

# Backend Setup (First Time)

Open a terminal and navigate to the backend folder:

```bash
cd backend
```

## Create Virtual Environment

```bash
py -m venv venv
```

## Activate Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

### Mac/Linux

```bash
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r packages.txt
```

---

# Firestore Catalog Migration

The system requires importing the product catalog into Firestore during first-time setup.

Inside the `backend` folder, run:

```bash
python catalog_migration.py
```

Wait until all catalog data has been imported successfully.

---

# Firestore Vector Index Configuration

After catalog import is completed, configure the Firestore vector index.

Run the following command in your terminal:

```bash
gcloud firestore indexes composite create \
--project={your firestore project bucket} \
--collection-group=catalog \
--query-scope=COLLECTION \
--field-config=order=ASCENDING,field-path=category \
--field-config=vector-config='{"dimension":"768","flat": "{}"}',field-path=embedding_vector
```

Replace:

```text
{your firestore project bucket}
```

with your actual Firebase/Firestore project ID.

---

# Frontend Setup (First Time)

Open another terminal and navigate to the frontend folder:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

---

# Running the Project

## Start Backend

Inside the `backend` folder:

```bash
uvicorn main:app --reload
```

The backend server will start running.

---

## Start Frontend

Open a new terminal.

Navigate to the frontend folder:

```bash
cd frontend
```

Run:

```bash
npm start
```

The React frontend application will start running.

---

# Project Structure

```text
project-root/
│
├── backend/
│   ├── firebase-key.json
│   └── ...
│
├── frontend/
│   └── ...
│
├── .env
└── sample.env
```

---

# Notes

- Ensure Firebase Firestore is enabled before running migrations
- Ensure `.env` values are correctly configured
- Ensure the vector index creation command completes successfully