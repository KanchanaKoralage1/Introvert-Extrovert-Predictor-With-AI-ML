from pydantic import BaseModel


class PersonalityRequest(BaseModel):
    Time_spent_Alone: float
    Stage_fear: str
    Social_event_attendance: float
    Going_outside: float
    Drained_after_socializing: str
    Friends_circle_size: float
    Post_frequency: float
    Social_Activity_Score: float
    Isolation_Index: float