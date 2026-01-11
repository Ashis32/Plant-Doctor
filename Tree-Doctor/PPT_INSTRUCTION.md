# Tree-Doctor Project: Presentation Instructions

This document outlines the structure and content for your minor project presentation. It includes the workflow, system architecture, and key technical details.

## Slide 1: Title Slide
- **Project Name**: Tree-Doctor
- **Subtitle**: AI-Powered Plant Disease Detection & Treatment Recommendation
- **Presented by**: [Your Name/Team Members]
- **Date**: [Date]

## Slide 2: Introduction & Problem Statement
- **Problem**: Plant diseases like "Early Blight" and "Late Blight" in potatoes significantly reduce crop yield. Early detection is difficult for farmers without expert knowledge.
- **Solution**: "Tree-Doctor" – A web-based application that uses Deep Learning (CNN) to detect diseases from leaf images and suggests specific treatments immediately.

## Slide 3: Objectives
- To automate the detection of potato leaf diseases.
- To provide an easy-to-use interface for farmers/users.
- To offer actionable treatment recommendations (medicines & dosage).
- To achieve high accuracy using Convolutional Neural Networks (CNN).

## Slide 4: Technology Stack
- **Frontend (UI)**: Next.js (React Framework), HTML/CSS.
- **Backend (API)**: Python, FastAPI.
- **Machine Learning**: TensorFlow, Keras.
- **Image Processing**: Pillow (PIL), NumPy.
- **Dataset**: PlantVillage Dataset (Potato classes).

## Slide 5: System Architecture
*Use the diagram below to explain how the system components interact.*

```mermaid
graph LR
    User[User] -- Uploads Image --> Frontend[Frontend (Next.js)]
    Frontend -- POST /predict --> Backend[Backend API (FastAPI)]
    
    subgraph "Server Side"
        Backend -- Preprocessing --> Preprocessor[Resize/Rescale]
        Preprocessor -- Input (256x256) --> Model[CNN Model (TensorFlow)]
        Model -- Prediction --> Logic[Recommendation Engine]
        Logic -- JSON Response --> Backend
    end
    
    Backend -- Disease + Medicine Info --> Frontend
    Frontend -- Displays Result --> User
```

**Key Components:**
1.  **Client**: The user interface where images are uploaded.
2.  **API Server**: Handles requests, loads the trained model, and processes images.
3.  **ML Model**: The "Brain" that classifies the image into Healthy, Early Blight, or Late Blight.

## Slide 6: System Workflow
1.  **Image Upload**: User selects a photo of a potato leaf.
2.  **Transmission**: Image is sent to the backend server via a REST API call.
3.  **Preprocessing**: Server resizes the image to 256x256 pixels and normalizes pixel values.
4.  **Inference**: The CNN model analyzes the image and outputs a confidence score for each class.
5.  **Recommendation**: Based on the highest confidence class, the system retrieves specific treatment advice (e.g., "Mancozeb" for Early Blight).
6.  **Result**: The diagnosis, confidence score, and medicine dosage are displayed to the user.

## Slide 7: Model Architecture (The "Brain")
*Explain the CNN structure used in `train.py`.*
- **Input Layer**: 256x256 RGB Images.
- **Convolutional Layers (Conv2D)**: Extract features like edges, textures, and spots.
- **Pooling Layers (MaxPooling2D)**: Reduce data size while keeping important features.
- **Flatten Layer**: Converts 2D feature maps to a 1D vector.
- **Dense Layers**: Fully connected layers for classification.
- **Output Layer**: Softmax activation with 3 neurons (representing the 3 classes).

## Slide 8: Implementation Details
- **Training**:
    - **Dataset Split**: 80% Training, 10% Validation, 10% Testing.
    - **Epochs**: 10 (or as configured).
    - **Batch Size**: 32.
- **Mock Mode**: A fail-safe feature in the backend that allows development even if TensorFlow is not installed (returns simulated results).

## Slide 9: Results & Screenshots
- *[Insert Screenshot of Home Page]*
- *[Insert Screenshot of Uploading an Image]*
- *[Insert Screenshot of Result Card with "Early Blight" detection and Medicine recommendation]*

## Slide 10: Conclusion & Future Scope
- **Conclusion**: The project successfully demonstrates how AI can assist in agriculture by providing quick and accurate disease diagnosis.
- **Future Scope**:
    - Mobile App development (already mobile-responsive web).
    - Adding more crops (Tomato, Corn, etc.).
    - Multi-language support for regional farmers.

## Slide 11: Q&A
- Thank the audience and invite questions.
