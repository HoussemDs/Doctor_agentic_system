from crewai import Task
from textwrap import dedent

class DoctorTasks:
    def __tip_section(self):
        return "Provide the most accurate and clear medical advice possible."

    def diagnose_patient(self, agent, patient_data):
        return Task(
            description=dedent(
                f"""
                **Task**: Diagnose the Patient
                **Description**: Based on the provided patient data, determine whether the patient
                    is healthy or sick. If sick, identify the illness with a clear and concise diagnosis.
                    Your diagnosis should be supported by reasoning based on the symptoms provided.
                    Avoid unnecessary details and keep the explanation medically relevant.

                **Parameters**:
                    - Patient Data: {patient_data}

                **Note**: {self.__tip_section()}
                """
            ),
            agent=agent,
            expected_output="Either 'Healthy' or 'Sick with [diagnosis]' based on the patient's symptoms."
        )

    def suggest_treatment(self, agent, diagnosis):
        return Task(
            description=dedent(
                f"""
                **Task**: Suggest Treatment and Timing
                **Description**: Based on the provided diagnosis, suggest an appropriate treatment plan,
                    including medication (if any), lifestyle recommendations, and the optimal timing or
                    duration for the treatment. Make the advice clear and easy to follow.

                **Parameters**:
                    - Diagnosis: {diagnosis}

                **Note**: {self.__tip_section()}
                """
            ),
            agent=agent,
            expected_output="A treatment plan in the format: 'We must use [treatment] to cure him at [time].'"
        )
