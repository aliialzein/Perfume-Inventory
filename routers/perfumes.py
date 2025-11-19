from fastapi import APIRouter, HTTPException, UploadFile, File, Query # type: ignore
from typing import List, Optional
from uuid import uuid4
from datetime import datetime
from pydantic import BaseModel # type: ignore
from pymongo import ReturnDocument # type: ignore

from config.database import perfumes_collection, fs
from models.perfume import PerfumeCreate, PerfumeUpdate, PerfumeInDB

router = APIRouter(
    prefix="/perfumes",
    tags=["perfumes"]
)


# --- helper to convert Mongo doc -> PerfumeInDB ---

def perfume_helper(doc) -> PerfumeInDB:
    """Convert a MongoDB document into PerfumeInDB."""
    return PerfumeInDB(
        id=str(doc.get("_id")),
        product_id=str(doc.get("product_id", "")),
        product_type=doc.get("product_type", ""),
        price_predicted=float(doc.get("price_predicted", 0)),
        price_modified=doc.get("price_modified"),
        quantity=int(doc.get("quantity", 0)),
        brand=doc.get("brand"),
        model_name=doc.get("model_name"),
        image_gridfs_id=doc.get("image_gridfs_id"),
        date_added=doc.get("date_added", datetime.utcnow()),
        capacity_ml=doc.get("capacity_ml"),
        perfume_type=doc.get("perfume_type"),
    )




# --- CREATE ---

@router.post("/", response_model=PerfumeInDB)
def create_perfume(perfume: PerfumeCreate):
    data = perfume.dict()
    data["product_id"] = str(uuid4())         # UUID as string
    data["date_added"] = datetime.utcnow()

    result = perfumes_collection.insert_one(data)
    data["_id"] = result.inserted_id

    return perfume_helper(data)


# --- LIST ALL ---

@router.get("/", response_model=List[PerfumeInDB])
def list_perfumes():
    perfumes = []
    for doc in perfumes_collection.find():
        perfumes.append(perfume_helper(doc))
    return perfumes


# --- GET ONE BY product_id ---

@router.get("/{product_id}", response_model=PerfumeInDB)
def get_perfume(product_id: str):
    doc = perfumes_collection.find_one({"product_id": product_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Perfume not found")
    return perfume_helper(doc)


# --- UPDATE BY product_id ---

@router.put("/{product_id}", response_model=PerfumeInDB)
def update_perfume(product_id: str, update: PerfumeUpdate):
    update_data = {k: v for k, v in update.dict().items() if v is not None}
    if not update_data:
        raise HTTPException(status_code=400, detail="No data to update")

    doc = perfumes_collection.find_one_and_update(
        {"product_id": product_id},
        {"$set": update_data},
        return_document=ReturnDocument.AFTER
    )

    if not doc:
        raise HTTPException(status_code=404, detail="Perfume not found")

    return perfume_helper(doc)


# --- DELETE BY product_id ---

@router.delete("/{product_id}")
def delete_perfume(product_id: str):
    result = perfumes_collection.delete_one({"product_id": product_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Perfume not found")

    return {"message": "Perfume deleted successfully"}

@router.post("/upload-image")
def upload_image(file: UploadFile = File(...)):
    """
    Upload a perfume image, store it in GridFS, return image_gridfs_id.
    """
    contents = file.file.read()
    gridfs_id = fs.put(
        contents,
        filename=file.filename,
        content_type=file.content_type
    )
    return {"image_gridfs_id": str(gridfs_id)}

@router.get("/check", response_model=PerfumeInDB | None)
def check_existing_perfume(
    brand: str = Query(...),
    model_name: str = Query(...)
):
    """
    Check if a perfume exists by brand + model_name.
    Return the perfume if found, otherwise null.
    """
    doc = perfumes_collection.find_one({"brand": brand, "model_name": model_name})
    if not doc:
        # FastAPI will serialize 'None' as null for the client
        return None
    return perfume_helper(doc)

class PriceUpdate(BaseModel):
    price_modified: float

@router.patch("/{product_id}/price", response_model=PerfumeInDB)
def update_price(product_id: str, body: PriceUpdate):
    doc = perfumes_collection.find_one_and_update(
        {"product_id": product_id},
        {"$set": {"price_modified": body.price_modified}},
        return_document=ReturnDocument.AFTER
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Perfume not found")
    return perfume_helper(doc)

class QuantityChange(BaseModel):
    delta: int

@router.patch("/{product_id}/quantity", response_model=PerfumeInDB)
def change_quantity(product_id: str, body: QuantityChange):
    """
    Increment or decrement quantity.
    Example: {"delta": 5} to add 5 units.
    """
    doc = perfumes_collection.find_one_and_update(
        {"product_id": product_id},
        {"$inc": {"quantity": body.delta}},
        return_document=ReturnDocument.AFTER
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Perfume not found")
    return perfume_helper(doc)

# ---------- AI PLACEHOLDER ENDPOINTS ----------

class ClassificationResult(BaseModel):
    image_gridfs_id: str
    product_type: str
    brand: str
    model_name: str


@router.post("/classify-image", response_model=ClassificationResult)
def classify_image(file: UploadFile = File(...)):
    ...
    # TODO: replace with real model inference
    product_type = "perfume"
    brand = "Dummy Brand"
    model_name = "Dummy Model"
    capacity_ml = 100                
    perfume_type = "after bath"       

    return ClassificationResult(
        image_gridfs_id=str(gridfs_id), # type: ignore
        product_type=product_type,
        brand=brand,
        model_name=model_name,
        capacity_ml=capacity_ml,
        perfume_type=perfume_type,
    )


class PricePredictionRequest(BaseModel):
    product_type: str
    brand: str
    model_name: str
    capacity_ml: int 
    perfume_type:str


class PricePredictionResponse(BaseModel):
    price_predicted: float


@router.post("/predict-price", response_model=PricePredictionResponse)
def predict_price(body: PricePredictionRequest):
    """
    1) Take text output from classification model
       (product_type, brand, model_name)
    2) (Future) Run price prediction model
       -> price_predicted
    """
    # TODO: replace with real price model
    # for now just return a dummy price
    dummy_price = 100.0

    return PricePredictionResponse(price_predicted=dummy_price)

class MetaUpdate(BaseModel):
    product_type: str
    brand: str
    model_name: str
    capacity_ml: int
    perfume_type: str


@router.patch("/{product_id}/meta", response_model=PerfumeInDB)
def update_metadata(product_id: str, body: MetaUpdate):
    """
    Edit classification-related fields after the first AI model:
    brand, model_name, capacity_ml, perfume_type, product_type.
    """
    update_data = {k: v for k, v in body.dict().items() if v is not None}
    if not update_data:
        raise HTTPException(status_code=400, detail="No data to update")

    doc = perfumes_collection.find_one_and_update(
        {"product_id": product_id},
        {"$set": update_data},
        return_document=ReturnDocument.AFTER
    )

    if not doc:
        raise HTTPException(status_code=404, detail="Perfume not found")

    return perfume_helper(doc)
@router.get("/filter", response_model=List[PerfumeInDB])
def filter_perfumes(
    brand: Optional[str] = None,
    model_name: Optional[str] = None,
    product_type: Optional[str] = None,
    perfume_type: Optional[str] = None,
    min_capacity: Optional[int] = None,
    max_capacity: Optional[int] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
):
    query: dict = {}

    if brand:
        query["brand"] = brand

    if model_name:
        query["model_name"] = model_name

    if product_type:
        query["product_type"] = product_type

    if perfume_type:
        query["perfume_type"] = perfume_type

    if min_capacity is not None or max_capacity is not None:
        cap_cond: dict = {}
        if min_capacity is not None:
            cap_cond["$gte"] = min_capacity
        if max_capacity is not None:
            cap_cond["$lte"] = max_capacity
        query["capacity_ml"] = cap_cond

    if min_price is not None or max_price is not None:
        price_cond: dict = {}
        if min_price is not None:
            price_cond["$gte"] = min_price
        if max_price is not None:
            price_cond["$lte"] = max_price
        query["price_predicted"] = price_cond

    docs = perfumes_collection.find(query)
    return [perfume_helper(doc) for doc in docs]