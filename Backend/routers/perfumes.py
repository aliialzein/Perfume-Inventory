from fastapi import APIRouter, HTTPException, UploadFile, File, Query  # type: ignore
from typing import List, Optional
from uuid import uuid4
from datetime import datetime
from pydantic import BaseModel  # type: ignore
from pymongo import ReturnDocument  # type: ignore

from config.database import perfumes_collection, fs
from models.perfume import PerfumeCreate, PerfumeUpdate, PerfumeInDB
from ai_models.perfume_classifier import classify_perfume
from ai_models.priceprediction import predict_price_from_features


router = APIRouter(
    prefix="/perfumes",
    tags=["perfumes"]
)


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
        gender=doc.get("gender"),
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


# ---------- AI ENDPOINTS ----------

class ClassificationResult(BaseModel):
    image_gridfs_id: str
    product_type: str
    brand: str
    model_name: str
    gender: str


@router.post("/classify-image", response_model=ClassificationResult)
def classify_image(file: UploadFile = File(...)):
    # 1) Read image bytes
    contents = file.file.read()

    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    # 2) Save image to GridFS
    gridfs_id = fs.put(
        contents,
        filename=file.filename,
        content_type=file.content_type
    )

    # 3) Run AI model
    try:
        result = classify_perfume(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {e}")

    # 4) Return structured result
    return ClassificationResult(
        image_gridfs_id=str(gridfs_id),
        product_type=result["product_type"],
        brand=result["brand"],
        model_name=result["model_name"],
        gender=result["gender"],
    )


class PricePredictionRequest(BaseModel):
    # comes from classify-image
    product_type: str
    brand: str
    model_name: str
    gender: str
    capacity_ml: int


class PricePredictionResponse(BaseModel):
    price_predicted: float


@router.post("/predict-price", response_model=PricePredictionResponse)
def predict_price(body: PricePredictionRequest):
    """
    1) Takes classification output:
       - product_type, brand, model_name, gender
    2) Also takes capacity_ml (chosen by the user)
    3) Runs the trained price prediction model
       -> returns price_predicted
    """
    try:
        features = body.dict()
        price = predict_price_from_features(features)
    except RuntimeError as e:
        # usually model file not loaded / missing
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error while running price prediction: {e}",
        )

    return PricePredictionResponse(price_predicted=price)


class MetaUpdate(BaseModel):
    product_type: str
    brand: str
    model_name: str
    capacity_ml: int
    gender: str


@router.patch("/{product_id}/meta", response_model=PerfumeInDB)
def update_metadata(product_id: str, body: MetaUpdate):
    """
    Edit classification-related fields after the first AI model:
    brand, model_name, gender, product_type.
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
    gender: Optional[str] = None,
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

    if gender:
        query["gender"] = gender  # fixed from "perfume_type"

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


class AutoCreateResponse(BaseModel):
    perfume: PerfumeInDB
    created: bool  # true if new perfume, false if already existed


@router.post("/auto-create-from-image", response_model=AutoCreateResponse)
def auto_create_from_image(file: UploadFile = File(...)):
    """
    1) Upload an image
    2) Classify (brand, model_name, gender)
    3) Store image in GridFS
    4) If perfume with same brand+model_name exists -> reuse it
       Otherwise create a new Perfume with default price/quantity.
    """
    contents = file.file.read()

    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    gridfs_id = fs.put(
        contents,
        filename=file.filename,
        content_type=file.content_type
    )

    try:
        result = classify_perfume(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {e}")

    brand = result["brand"]
    model_name = result["model_name"]
    gender = result["gender"]
    product_type = result["product_type"]

    # 1) Check if perfume already exists
    existing = perfumes_collection.find_one(
        {"brand": brand, "model_name": model_name}
    )

    if existing:
        # just update image / gender if you want
        perfumes_collection.update_one(
            {"_id": existing["_id"]},
            {"$set": {"image_gridfs_id": str(gridfs_id), "gender": gender}},
        )
        updated = perfumes_collection.find_one({"_id": existing["_id"]})
        return AutoCreateResponse(perfume=perfume_helper(updated), created=False)

    # 2) Create new perfume (dummy price & quantity for now)
    data = {
        "product_id": str(uuid4()),
        "product_type": product_type,
        "price_predicted": 0.0,   # will be replaced by price model later
        "price_modified": None,
        "quantity": 0,
        "brand": brand,
        "model_name": model_name,
        "capacity_ml": None,
        "gender": gender,
        "image_gridfs_id": str(gridfs_id),
        "date_added": datetime.utcnow(),
    }

    result_insert = perfumes_collection.insert_one(data)
    data["_id"] = result_insert.inserted_id

    return AutoCreateResponse(perfume=perfume_helper(data), created=True)


class ResolveImageResponse(BaseModel):
    image_gridfs_id: str
    product_type: str
    brand: str
    model_name: str
    gender: str
    capacity_ml: Optional[int] = None
    exists: bool
    perfume: Optional[PerfumeInDB] = None
    price_predicted: Optional[float] = None


@router.post("/resolve-image", response_model=ResolveImageResponse)
def resolve_image(
    file: UploadFile = File(...),
    capacity_ml: Optional[int] = Query(
        None,
        description="Capacity in ml to use for price prediction if perfume does not exist yet",
    ),
):
    """
    1) Upload image
    2) Classify to (product_type, brand, model_name, gender)
    3) Check if a perfume with same brand + model_name exists in DB
       - If EXISTS: return existing perfume (with price & quantity)
       - If NOT: optionally run price prediction (if capacity_ml is given)
                 and return classification + predicted price so frontend
                 can let user edit and then call POST /perfumes.
    """
    contents = file.file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    # Save image to GridFS
    gridfs_id = fs.put(
        contents,
        filename=file.filename,
        content_type=file.content_type,
    )

    # Run classifier
    try:
        result = classify_perfume(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {e}")

    product_type = result["product_type"]
    brand = result["brand"]
    model_name = result["model_name"]
    gender = result["gender"]

    # 1) Check if perfume already exists (brand + model_name)
    existing = perfumes_collection.find_one(
        {"brand": brand, "model_name": model_name}
    )

    if existing:
        current_price = existing.get("price_modified") or existing.get("price_predicted", 0.0)

        return ResolveImageResponse(
            image_gridfs_id=str(gridfs_id),
            product_type=product_type,
            brand=brand,
            model_name=model_name,
            gender=gender,
            capacity_ml=existing.get("capacity_ml"),
            exists=True,
            perfume=perfume_helper(existing),
            price_predicted=float(current_price),
        )

    # 2) Not existing: optionally run price model if capacity_ml provided
    predicted_price: Optional[float] = None
    if capacity_ml is not None:
        features = {
            "product_type": product_type,
            "brand": brand,
            "model_name": model_name,
            "gender": gender,
            "capacity_ml": capacity_ml,
        }
        try:
            predicted_price = predict_price_from_features(features)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Price prediction error: {e}",
            )

    return ResolveImageResponse(
        image_gridfs_id=str(gridfs_id),
        product_type=product_type,
        brand=brand,
        model_name=model_name,
        gender=gender,
        capacity_ml=capacity_ml,
        exists=False,
        perfume=None,
        price_predicted=predicted_price,
    )
