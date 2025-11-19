# models/perfume.py
from datetime import datetime
from typing import Optional
from pydantic import BaseModel   # type: ignore


class PerfumeBase(BaseModel):
    product_type: str
    price_predicted: float
    price_modified: Optional[float] = None
    quantity: int
    brand: Optional[str] = None
    model_name: Optional[str] = None

    # 🔹 NEW FIELDS
    capacity_ml: Optional[int] = None       # e.g. 50, 100
    perfume_type: Optional[str] = None      # e.g. "after bath", "eau de parfum"


class PerfumeCreate(PerfumeBase):
    image_gridfs_id: Optional[str] = None


class PerfumeUpdate(BaseModel):
    product_type: Optional[str] = None
    price_predicted: Optional[float] = None
    price_modified: Optional[float] = None
    quantity: Optional[int] = None
    image_gridfs_id: Optional[str] = None
    brand: Optional[str] = None
    model_name: Optional[str] = None

    # 🔹 NEW FIELDS
    capacity_ml: Optional[int] = None
    perfume_type: Optional[str] = None


class PerfumeInDB(PerfumeBase):
    id: str                     # string version of Mongo _id
    product_id: str             # UUID string
    image_gridfs_id: Optional[str] = None
    date_added: datetime
