import secrets

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from config import settings
from inference.markup import parse_chartreader_markup
from inference.service import ClusterParams, InferenceService


app = FastAPI(title="Line Grouping API", version="1.0.0")
service = InferenceService(
    checkpoint_path=settings.model_checkpoint_path,
    emb_dim=settings.model_emb_dim,
    device=(settings.model_device or None),
)
security = HTTPBasic()


def require_basic_auth(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    if not settings.api_auth_enabled:
        return "auth-disabled"

    user_ok = secrets.compare_digest(credentials.username, settings.api_auth_username)
    pass_ok = secrets.compare_digest(credentials.password, settings.api_auth_password)
    if not (user_ok and pass_ok):
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model/info")
def model_info() -> dict[str, str | int]:
    return {
        "checkpoint_path": str(service.checkpoint_path),
        "emb_dim": int(service.emb_dim),
        "device": str(service.device),
    }


@app.post("/v1/cluster")
async def cluster_image(
    _: str = Depends(require_basic_auth),
    file: UploadFile = File(...),
    markup: str | None = Form(None),
    score_thr: float = Form(0.4),
    category_idx: int = Form(0),
    clustering_method: str = Form("hdbscan"),
    eps: float = Form(0.3),
    min_samples: int = Form(12),
    l2_normalize: bool = Form(True),
    auto_eps: bool = Form(False),
    auto_eps_k: int = Form(10),
    auto_eps_q: float = Form(0.90),
    use_spatial: bool = Form(True),
    spatial_weight: float = Form(0.25),
    candidate_method: str = Form("non_white"),
    white_threshold: int = Form(245),
    canny_threshold1: int = Form(80),
    canny_threshold2: int = Form(180),
    canny_aperture_size: int = Form(3),
    canny_l2gradient: bool = Form(False),
    canny_dilate_iter: int = Form(1),
    max_candidate_points: int = Form(0),
):
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(raw) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max allowed bytes: {settings.max_upload_bytes}",
        )

    params = ClusterParams(
        clustering_method=clustering_method,
        eps=eps,
        min_samples=min_samples,
        l2_normalize=l2_normalize,
        auto_eps=auto_eps,
        auto_eps_k=auto_eps_k,
        auto_eps_q=auto_eps_q,
        use_spatial=use_spatial,
        spatial_weight=spatial_weight,
        candidate_method=candidate_method,
        white_threshold=white_threshold,
        canny_threshold1=canny_threshold1,
        canny_threshold2=canny_threshold2,
        canny_aperture_size=canny_aperture_size,
        canny_l2gradient=canny_l2gradient,
        canny_dilate_iter=canny_dilate_iter,
        max_candidate_points=max_candidate_points,
    )
    markup_provided = bool(markup and markup.strip())
    candidate_coords = parse_chartreader_markup(
        markup_raw=markup,
        score_thr=score_thr,
        category_idx=category_idx,
        target_image_name=file.filename,
    )
    used_markup = len(candidate_coords) > 0
    fallback_reason = None
    if markup_provided and not used_markup:
        fallback_reason = "markup_empty_or_unparsed_after_score_filter"

    try:
        # If markup is provided and parsed successfully, use those candidates.
        # Otherwise fallback to candidate_method-based extraction.
        coords = candidate_coords if used_markup else None
        result = service.predict(image_bytes=raw, params=params, candidate_coords=coords)
        result.setdefault("meta", {})
        result["meta"].update(
            {
                "candidate_source": "markup" if used_markup else "candidate_method",
                "markup_provided": bool(markup_provided),
                "markup_points_after_score_thr": int(len(candidate_coords)),
                "requested_file_name": str(file.filename),
                "fallback_reason": fallback_reason,
            }
        )
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}") from e
