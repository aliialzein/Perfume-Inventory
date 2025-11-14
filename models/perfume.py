from dataclasses import Field
from datetime import datetime
from typing import Optional
from pydantic import BaseModel # type: ignore

class PerfumeBase(BaseModel):
    product_type: str              
    price_predicted: float
    price_modified: Optional[float] = None
    quantity: int
    brand: Optional[str] = None
    model_name: Optional[str] = None

class PerfumeCreate(PerfumeBase):
    image_gridfs_id: Optional[str] = None

class PerfumeUpdate(BaseModel):
    product_name: Optional[str] = None
    product_type: Optional[str] = None
    price_predicted: Optional[float] = None
    price_modified: Optional[float] = None
    quantity: Optional[int] = None
    image_gridfs_id: Optional[str] = None

class PerfumeInDB(PerfumeBase):
    id: str = Field(alias="_id")
    product_id: str
    image_gridfs_id: Optional[str] = None
    date_added: datetime

    class Config:
        allow_population_by_field_name = True

