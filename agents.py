from crewai import Agent, LLM
from textwrap import dedent
from tools import heart_predictor, heart_image_display

class DoctorsAgents:
    def __init__(self):
        self.GroqLLaMA = LLM(model="groq/llama-3.3-70b-versatile")

    def Diagnosis_Doctor(self):
        return Agent(
            role="Heart Diagnosis Doctor",
            backstory=dedent("""
                You are a cardiologist with 20 years of experience diagnosing heart diseases.
                You are expert in identifying which part of the heart is affected based on symptoms.
                You have access to an advanced ML model that can help predict heart conditions.
            """),
            goal=dedent("""
                Diagnose if the patient has a heart problem using both your medical expertise 
                and the ML prediction tool available to you.
                If yes, specify the heart condition and the affected heart part
                in the format: "Sick with [condition] affecting [heart part]".
                If healthy, respond with "Healthy heart".
                Use the heart disease prediction tool to get ML-based insights.
            """),
            allow_delegation=False,
            verbose=True,
            llm=self.GroqLLaMA,
            tools=[heart_predictor, heart_image_display]
        )

    def Treatment_Doctor(self):
        return Agent(
            role="Heart Treatment Doctor",
            backstory=dedent("""
                You are a senior cardiologist with 20 years of experience
                in prescribing treatment plans for heart conditions.
            """),
            goal=dedent("""
                Suggest a clear and effective treatment plan
                based on the diagnosis, including medication,
                lifestyle changes, and treatment timing.
            """),
            allow_delegation=False,
            verbose=True,
            llm=self.GroqLLaMA
        )