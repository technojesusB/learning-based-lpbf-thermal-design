from pydantic import BaseModel, Field


class InstrumentationConfig(BaseModel):
    """Controls the full-system-scan instrumentation stack."""

    system_metrics: bool = False
    flight_recorder: bool = False
    progress_reporter: bool = True
    nvml_sample_interval_steps: int = Field(default=1, ge=1)
    flight_recorder_history: int = Field(default=5, ge=1)
