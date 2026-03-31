from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
import threading
from typing import Any, Optional

from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .datasets import get_dataset_record
from .exceptions import YoloCtlError
from .review import (
    REVIEW_STATUSES,
    apply_prediction_to_draft,
    build_prediction_issue_index,
    build_session_summary,
    filter_review_items,
    generate_thumbnail,
    get_item_payload,
    load_review_session,
    replace_draft_with_predictions,
    save_item_draft,
    update_item_status,
)

TEMPLATES_DIR = Path(__file__).resolve().parent / "review_templates"
STATIC_DIR = Path(__file__).resolve().parent / "review_static"


def _as_http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, YoloCtlError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, FileNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


def create_review_app(session_dir: Path, start_indexer: bool = True) -> FastAPI:
    session = load_review_session(session_dir)
    record = get_dataset_record(session.dataset_key, version_id=session.base_version_id)
    static_version = str(
        max(
            int((STATIC_DIR / "app.js").stat().st_mtime),
            int((STATIC_DIR / "styles.css").stat().st_mtime),
        )
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        current = load_review_session(session_dir)
        should_index = (
            start_indexer
            and current.weights
            and current.prediction_index.get("state") in {"pending", "running"}
        )
        if should_index:
            def _runner() -> None:
                try:
                    build_prediction_issue_index(session_dir)
                except Exception:  # pragma: no cover - surfaced in session summary for manual debugging
                    failed = load_review_session(session_dir)
                    failed.prediction_index["state"] = "error"
                    failed.prediction_index["message"] = "Prediction indexing failed"
                    from .review import save_review_session

                    save_review_session(session_dir, failed)

            threading.Thread(target=_runner, daemon=True).start()
        yield

    app = FastAPI(title="yoloctl-review", docs_url=None, redoc_url=None, lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    @app.get("/")
    def review_index(request: Request) -> Any:
        summary = build_session_summary(session_dir)
        return templates.TemplateResponse(
            request,
            "review.html",
            {
                "session": summary["session"],
                "summary": summary["counts"],
                "dataset": record.to_mapping(),
                "statuses": sorted(REVIEW_STATUSES),
                "splits": list(session.selected_splits),
                "class_names": list(record.class_names),
                "static_version": static_version,
            },
        )

    @app.get("/api/session")
    def api_session() -> dict[str, Any]:
        return {
            "session": load_review_session(session_dir).to_mapping(),
            "summary": build_session_summary(session_dir)["counts"],
            "dataset": record.to_mapping(),
            "statuses": sorted(REVIEW_STATUSES),
            "splits": list(session.selected_splits),
            "class_names": list(record.class_names),
        }

    @app.get("/api/items")
    def api_items(
        split: Optional[str] = None,
        status: Optional[str] = None,
        issue_type: Optional[str] = Query(default=None, alias="issueType"),
        gt_class: Optional[str] = Query(default=None, alias="gtClass"),
        pred_class: Optional[str] = Query(default=None, alias="predClass"),
        edited_only: bool = Query(default=False, alias="editedOnly"),
        query: Optional[str] = None,
        order_by: str = Query(default="issue_severity", alias="orderBy"),
    ) -> dict[str, Any]:
        try:
            items = filter_review_items(
                session_dir=session_dir,
                split=split,
                status=status,
                issue_type=issue_type,
                gt_class=gt_class,
                pred_class=pred_class,
                edited_only=edited_only,
                query=query,
                order_by=order_by,
            )
        except Exception as exc:  # pragma: no cover - converted to API error
            raise _as_http_error(exc) from exc
        return {"items": items}

    @app.get("/api/items/{image_id}")
    def api_item(image_id: str) -> dict[str, Any]:
        try:
            return get_item_payload(session_dir, image_id)
        except Exception as exc:  # pragma: no cover - converted to API error
            raise _as_http_error(exc) from exc

    @app.get("/api/image/{image_id}")
    def api_image(image_id: str) -> FileResponse:
        try:
            payload = get_item_payload(session_dir, image_id)
        except Exception as exc:  # pragma: no cover - converted to API error
            raise _as_http_error(exc) from exc
        return FileResponse(payload["item"]["image_path"])

    @app.get("/api/thumbnail/{image_id}")
    def api_thumbnail(image_id: str) -> FileResponse:
        try:
            thumbnail = generate_thumbnail(session_dir, image_id)
        except Exception as exc:  # pragma: no cover - converted to API error
            raise _as_http_error(exc) from exc
        return FileResponse(thumbnail)

    @app.post("/api/items/{image_id}/save")
    def api_save_item(image_id: str, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        try:
            return save_item_draft(
                session_dir=session_dir,
                image_id=image_id,
                draft_objects=list(payload.get("draft_objects", [])),
                status=payload.get("status"),
                note=payload.get("note"),
            )
        except Exception as exc:  # pragma: no cover - converted to API error
            raise _as_http_error(exc) from exc

    @app.post("/api/items/{image_id}/status")
    def api_status(image_id: str, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        try:
            item = update_item_status(
                session_dir=session_dir,
                image_id=image_id,
                status=str(payload.get("status", "")),
                note=payload.get("note"),
            )
        except Exception as exc:  # pragma: no cover - converted to API error
            raise _as_http_error(exc) from exc
        return {"item": item.to_mapping()}

    @app.post("/api/items/{image_id}/predict")
    def api_predict(image_id: str) -> dict[str, Any]:
        try:
            return get_item_payload(session_dir, image_id, force_predict=True)
        except Exception as exc:  # pragma: no cover - converted to API error
            raise _as_http_error(exc) from exc

    @app.post("/api/items/{image_id}/apply-prediction")
    def api_apply_prediction(image_id: str, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        try:
            return apply_prediction_to_draft(
                session_dir=session_dir,
                image_id=image_id,
                prediction_id=str(payload.get("prediction_id", "")),
                mode=str(payload.get("mode", "append")),
                target_draft_id=str(payload["target_draft_id"]) if payload.get("target_draft_id") else None,
            )
        except Exception as exc:  # pragma: no cover - converted to API error
            raise _as_http_error(exc) from exc

    @app.post("/api/items/{image_id}/replace-with-predictions")
    def api_replace_predictions(image_id: str) -> dict[str, Any]:
        try:
            return replace_draft_with_predictions(session_dir=session_dir, image_id=image_id)
        except Exception as exc:  # pragma: no cover - converted to API error
            raise _as_http_error(exc) from exc

    @app.get("/api/session/summary")
    def api_summary() -> dict[str, Any]:
        return build_session_summary(session_dir)

    return app
