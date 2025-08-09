from crewai import Agent, LLM
from textwrap import dedent

class DoctorsAgents:
    def __init__(self):
        # Using Groq LLaMA model
        self.GroqLLaMA = LLM(model="groq/llama-3.3-70b-versatile")

    def Diagnosis_Doctor(self):
        return Agent(
            role="Diagnosis Doctor",
            backstory=dedent("""
                You are an experienced medical doctor with 20 years of experience
                in diagnosing patients. You can accurately identify illnesses
                based on symptoms, lab results, and medical history.
            """),
            goal=dedent("""
                Determine whether the patient is sick or healthy. If sick,
                provide the illness name in the format:
                'Sick with [diagnosis]'.
                If healthy, simply state 'Healthy'.
            """),
            allow_delegation=False,
            verbose=True,
            llm=self.GroqLLaMA
        )

    def Treatment_Doctor(self):
        return Agent(
            role="Treatment Doctor",
            backstory=dedent("""
                You are a senior medical specialist with over 20 years of
                experience in treatment planning. You have extensive knowledge
                of medical guidelines, medications, and best practices for
                curing diseases effectively and safely.
            """),
            goal=dedent("""
                Based on the patient's diagnosis, create a clear treatment plan.
                Include medication or therapy recommendations and specify the
                timing or duration of the treatment.
            """),
            allow_delegation=False,
            verbose=True,
            llm=self.GroqLLaMA
        )
