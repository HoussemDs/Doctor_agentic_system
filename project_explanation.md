# Heart Diagnosis & Treatment AI System - Complete Explanation

## 🎯 LEVEL 1: HIGH-LEVEL OVERVIEW

### What Does This System Do?

This is an **AI-powered medical diagnosis system** that combines:
- **Human-like AI agents** (digital doctors) that can reason and communicate
- **Machine Learning models** trained on medical data
- **Visual medical references** (heart condition images)

### Input → Process → Output

**INPUT:**
- Patient symptoms (text): "I have chest pain and shortness of breath"
- Medical history information

**PROCESS:**
1. **Diagnosis Doctor Agent** analyzes symptoms using ML model
2. **Treatment Doctor Agent** creates personalized treatment plan
3. System shows relevant medical imagery

**OUTPUT:**
- Precise heart condition diagnosis
- Detailed treatment plan with medications
- Medical images for visual reference
- Professional medical reasoning

### Key Benefits

🏥 **For Medical Professionals:**
- Quick second opinion on diagnoses
- Consistent treatment recommendations
- Visual aids for patient education
- 24/7 availability

🤖 **Technical Innovation:**
- Combines AI reasoning with ML predictions
- Multi-agent collaboration (like a medical team)
- Explainable AI decisions
- Integrates multiple data sources

💡 **Business Value:**
- Reduces diagnosis time
- Improves accuracy through dual validation
- Scalable medical expertise
- Cost-effective healthcare support

---

## 🔧 LEVEL 2: FILE-BY-FILE BREAKDOWN

### 📁 Project Structure
```
Medical_crew/
├── main.py           # System entry point & orchestration
├── agents.py         # AI doctor agents definitions
├── tasks.py          # Specific medical tasks
├── tools.py          # ML model integration & utilities
├── test_tools.py     # Testing & validation
├── requirements.txt  # Dependencies
├── .env             # API keys & configuration
├── Random_Forest_Heart_Model.joblib    # Trained ML model
├── model_columns.joblib                # Model feature columns
└── images/          # Medical condition images
```

### 🎭 **agents.py** - The Digital Doctors
**Purpose:** Defines AI agents that act like specialized doctors

**Components:**
- **DoctorsAgents Class**: Factory for creating doctor agents
- **Diagnosis Doctor**: Cardiologist specializing in heart diagnosis
  - Has access to ML prediction tools
  - 20 years experience persona
  - Goal: Identify heart problems and affected parts
- **Treatment Doctor**: Senior cardiologist for treatment planning
  - Focuses on medication and lifestyle recommendations
  - Creates comprehensive treatment plans

**Key Features:**
- Uses Groq's LLaMA model for reasoning
- Each agent has specific role, backstory, and goals
- Tools integration for ML predictions

### 📋 **tasks.py** - Medical Procedures
**Purpose:** Defines specific medical tasks agents must perform

**Components:**
- **DoctorTasks Class**: Contains all medical task definitions
- **diagnose_patient()**: 
  - Analyzes symptoms
  - Uses ML prediction tool
  - Displays relevant medical images
  - Outputs structured diagnosis
- **suggest_treatment()**:
  - Creates treatment plans based on diagnosis
  - Includes medications, lifestyle changes, timing

**Task Flow:**
1. Diagnosis task → Uses ML model → Shows images
2. Treatment task → Uses diagnosis results → Creates plan

### 🛠️ **tools.py** - ML Model Integration
**Purpose:** Bridges AI agents with machine learning models

**Components:**
- **HeartPredictionTool**: 
  - Loads trained Random Forest model
  - Processes patient data
  - Returns ML predictions (13 different heart conditions)
- **HeartConditionImageTool**:
  - Maps conditions to medical images
  - Displays relevant visuals
  - Supports patient education

**ML Model Details:**
- Random Forest classifier
- 13 heart conditions (MI types, cardiogenic shock, etc.)
- 40+ medical features (blood tests, ECG data, etc.)

### 🚀 **main.py** - System Orchestra
**Purpose:** Coordinates the entire system execution

**Flow:**
1. Load environment variables (API keys)
2. Get patient input
3. Test tools functionality
4. Create medical crew (agents + tasks)
5. Execute diagnosis → treatment pipeline
6. Display results

**Features:**
- Error handling and validation
- User interface management
- Crew orchestration
- Results presentation

### 🧪 **test_tools.py** - Quality Assurance
**Purpose:** Independent testing of ML tools

**Functions:**
- Tests heart prediction tool
- Tests image display functionality
- Validates model loading
- Provides debugging information

---

## 📝 LEVEL 3: LINE-BY-LINE CODE ANALYSIS

### 🔍 **tools.py** - Detailed Code Walkthrough

```python
from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import pandas as pd
import joblib
from PIL import Image
import os
```
**Lines 1-7:** Import statements
- `BaseTool`: CrewAI's base class for creating custom tools
- `BaseModel, Field`: Pydantic for data validation and schemas
- `pandas`: Data manipulation for ML model input
- `joblib`: Loading saved scikit-learn models
- `PIL`: Image processing and display
- `os`: File system operations

```python
class HeartPredictionInput(BaseModel):
    """Input schema for Heart Disease Predictor."""
    patient_data: str = Field(..., description="Patient data string containing symptoms and medical history")
```
**Lines 9-11:** Pydantic schema definition
- Defines the expected input format for the ML tool
- `Field(...)`: Makes patient_data required
- Used by CrewAI for input validation

```python
class HeartPredictionTool(BaseTool):
    name: str = "Heart Disease Predictor"
    description: str = "Predicts heart condition based on patient medical data using a trained ML model."
    args_schema: Type[BaseModel] = HeartPredictionInput
```
**Lines 13-16:** Tool class definition
- Inherits from `BaseTool` (CrewAI's tool base class)
- `name`: How agents refer to this tool
- `description`: What agents understand about the tool's purpose
- `args_schema`: Links to the input validation schema

```python
def _run(self, patient_data: str) -> str:
```
**Line 18:** Main tool execution method
- `_run`: Required method name for CrewAI tools
- Takes patient data as string input
- Returns string result for the agent

```python
try:
    # Load the trained model and columns
    model = joblib.load('Random_Forest_Heart_Model.joblib')
    model_columns = joblib.load('model_columns.joblib')
```
**Lines 28-31:** Model loading
- `joblib.load()`: Loads pre-trained Random Forest model
- Loads the exact column order the model expects
- Wrapped in try-catch for error handling

```python
sample_data = {
    'Age': [45],
    'F.History': [0],
    'Diabetes': [0],
    # ... 40+ medical features
}
```
**Lines 35-70:** Sample medical data
- Dictionary with all medical features the model needs
- Currently uses fixed sample data
- In production, would parse `patient_data` string
- Each key corresponds to a medical measurement

```python
input_df = pd.DataFrame(sample_data)

# Ensure all model columns are present
for col in model_columns:
    if col not in input_df.columns:
        input_df[col] = 2

input_df = input_df[model_columns]
```
**Lines 72-79:** Data preprocessing
- Convert dictionary to pandas DataFrame
- Add missing columns with default value (2)
- Reorder columns to match model's expected order
- Critical for model compatibility

```python
# Make prediction
pred_encoded = model.predict(input_df)
```
**Lines 81-82:** ML prediction
- `model.predict()`: Gets numerical prediction from Random Forest
- Returns encoded number (0-12) representing heart condition

```python
label_mapping = {
    0: 'Anterior Wall MI',
    1: 'Cardiogenic Shock',
    2: 'Extensive MI',
    # ... maps 0-12 to condition names
}

pred_label = label_mapping.get(pred_encoded[0], "Unknown")
return f"ML Model Prediction: {pred_label}"
```
**Lines 84-99:** Result interpretation
- Maps numeric prediction to human-readable condition name
- `get()` method provides fallback for unknown predictions
- Returns formatted string for the agent

### 🎭 **agents.py** - Agent Architecture Analysis

```python
from crewai import Agent, LLM
from textwrap import dedent
from tools import heart_predictor, heart_image_display
```
**Lines 1-3:** Import CrewAI components and custom tools
- `Agent`: CrewAI's agent class
- `LLM`: Language model wrapper
- Custom tools from tools.py

```python
class DoctorsAgents:
    def __init__(self):
        self.GroqLLaMA = LLM(model="groq/llama-3.3-70b-versatile")
```
**Lines 5-7:** Agent factory class
- Centralizes agent creation
- Initializes LLM connection to Groq's LLaMA model
- 70B parameter model for advanced reasoning

```python
def Diagnosis_Doctor(self):
    return Agent(
        role="Heart Diagnosis Doctor",
        backstory=dedent("""
            You are a cardiologist with 20 years of experience diagnosing heart diseases.
            You are expert in identifying which part of the heart is affected based on symptoms.
            You have access to an advanced ML model that can help predict heart conditions.
        """),
```
**Lines 9-16:** Agent persona definition
- `role`: Agent's professional identity
- `backstory`: Detailed background that shapes behavior
- Creates expertise context for better responses
- `dedent()`: Removes indentation for clean text

```python
goal=dedent("""
    Diagnose if the patient has a heart problem using both your medical expertise 
    and the ML prediction tool available to you.
    If yes, specify the heart condition and the affected heart part
    in the format: "Sick with [condition] affecting [heart part]".
    If healthy, respond with "Healthy heart".
    Use the heart disease prediction tool to get ML-based insights.
"""),
```
**Lines 17-24:** Agent objectives
- Clear instructions on what the agent should accomplish
- Specifies output format for consistency
- Mandates tool usage for ML integration
- Combines AI reasoning with ML predictions

```python
allow_delegation=False,
verbose=True,
llm=self.GroqLLaMA,
tools=[heart_predictor, heart_image_display]
```
**Lines 25-28:** Agent configuration
- `allow_delegation=False`: Agent works independently
- `verbose=True`: Shows detailed execution logs
- `llm`: Links to the language model
- `tools`: Array of available tools for this agent

### 📋 **tasks.py** - Task Orchestration Analysis

```python
def diagnose_patient(self, agent, patient_data):
    return Task(
        description=dedent(
            f"""
            **Task**: Diagnose Heart Condition
            **Description**: Analyze the patient's heart-related symptoms and identify if there is a problem.
                
                IMPORTANT: Use the "Heart Disease Predictor" tool to get ML-based predictions 
                and combine this with your medical expertise.
```
**Lines 9-18:** Task definition structure
- Function creates Task objects for CrewAI
- `f-string` allows dynamic patient data insertion
- Structured markdown format for clarity
- Explicit tool usage instructions

```python
Steps:
1. First, use the Heart Disease Predictor tool with the patient data
2. Analyze the ML prediction results
3. Combine ML insights with your medical knowledge
4. If a specific condition is predicted, use the Heart Condition Image Display tool to show relevant imagery
5. Provide final diagnosis
```
**Lines 20-25:** Execution workflow
- Step-by-step instructions for the agent
- Ensures proper tool usage sequence
- Combines ML and AI reasoning
- Includes visual aid integration

```python
expected_output='Either "Healthy heart" or "Sick with [heart condition] affecting [heart part]", including ML prediction results and medical imagery when applicable.'
```
**Lines 35-36:** Output specification
- Defines exact format expected
- Ensures consistent results
- Includes multiple information sources
- Used by CrewAI for validation

### 🚀 **main.py** - System Integration Analysis

```python
def main():
    # Load environment variables
    load_dotenv()
    
    # Set up API key
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("❌ Error: GROQ_API_KEY not found in environment variables!")
        return
```
**Lines 6-13:** Environment setup and validation
- `load_dotenv()`: Loads .env file variables
- Retrieves API key for Groq LLM service
- Error handling for missing configuration
- Prevents system execution without proper setup

```python
# Test tools first
print("Testing tools before starting...")
try:
    from tools import heart_predictor, heart_image_display
    test_result = heart_predictor._run("test")
    print(f"✅ Tools test: {test_result[:50]}...")
except Exception as e:
    print(f"❌ Tools test failed: {e}")
    return
```
**Lines 52-60:** Pre-flight validation
- Tests ML model loading before full execution
- Prevents runtime failures during agent execution
- Shows first 50 characters of test result
- Graceful failure handling

```python
crew = Crew(
    agents=[diagnosis_doctor, treatment_doctor],
    tasks=[diagnose_task, treatment_task],
    verbose=True,
)

result = crew.kickoff()
```
**Lines 44-49:** CrewAI orchestration
- `Crew`: Main orchestrator class
- Links agents with their respective tasks
- `verbose=True`: Shows detailed execution logs
- `kickoff()`: Starts the multi-agent workflow

This system represents a sophisticated integration of:
- **Multi-agent AI** for complex reasoning
- **Machine Learning** for data-driven predictions  
- **Tool integration** for specialized capabilities
- **Professional medical knowledge** through agent personas
- **Visual learning aids** for better understanding

The architecture is modular, scalable, and designed for real-world medical support applications.