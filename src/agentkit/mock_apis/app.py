from __future__ import annotations

import copy
import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel

DATA_DIR = Path(os.environ.get("AGENTKIT_DATA_DIR", Path(__file__).resolve().parents[3] / "data"))

ACME_QUEUES = ["general", "onboarding", "payouts", "disputes", "risk-limits", "compliance"]
CONTOSO_TEAMS = ["service-desk", "kyc-ops", "payments-ops", "compliance"]


def load_seed(customer: str) -> dict[str, Any]:
    return json.loads((DATA_DIR / customer / "seed.json").read_text())


def build_warehouse(seed: dict[str, Any], path: Path) -> Path:
    """Materialise the Contoso data warehouse (accounts, transfers) as SQLite."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    with sqlite3.connect(path) as conn:
        for table in ("accounts", "transfers"):
            rows = seed[table]
            cols = list(rows[0])
            conn.execute(f"CREATE TABLE {table} ({', '.join(cols)})")
            conn.executemany(
                f"INSERT INTO {table} VALUES ({', '.join('?' for _ in cols)})",
                [tuple(r[c] for c in cols) for r in rows],
            )
    return path


class TicketUpdate(BaseModel):
    queue: str | None = None
    priority: str | None = None
    status: str | None = None
    note: str | None = None


class Note(BaseModel):
    note: str


class CaseUpdate(BaseModel):
    state: str | None = None
    severity: str | None = None
    team: str | None = None
    comment: str | None = None


def acme_router(store: dict[str, Any]) -> APIRouter:
    router = APIRouter(prefix="/acme")
    clients = {c["client_id"]: c for c in store["clients"]}
    tickets = {t["ticket_id"]: t for t in store["tickets"]}

    @router.get("/crm/clients/{client_id}")
    def get_client(client_id: str) -> dict[str, Any]:
        if client_id not in clients:
            raise HTTPException(404, "client not found")
        return clients[client_id]

    @router.get("/crm/clients")
    def search_clients(name: str = "") -> list[dict[str, Any]]:
        return [c for c in clients.values() if name.lower() in c["legal_name"].lower()]

    @router.post("/crm/clients/{client_id}/notes")
    def add_note(client_id: str, body: Note) -> dict[str, Any]:
        if client_id not in clients:
            raise HTTPException(404, "client not found")
        clients[client_id].setdefault("notes", []).append(body.note)
        return {"ok": True, "client_id": client_id}

    @router.get("/tickets/{ticket_id}")
    def get_ticket(ticket_id: str) -> dict[str, Any]:
        if ticket_id not in tickets:
            raise HTTPException(404, "ticket not found")
        return tickets[ticket_id]

    @router.get("/tickets")
    def list_tickets(status: str = "", client_id: str = "") -> list[dict[str, Any]]:
        return [
            t
            for t in tickets.values()
            if (not status or t["status"] == status) and (not client_id or t["client_id"] == client_id)
        ]

    @router.patch("/tickets/{ticket_id}")
    def update_ticket(ticket_id: str, body: TicketUpdate) -> dict[str, Any]:
        if ticket_id not in tickets:
            raise HTTPException(404, "ticket not found")
        if body.queue and body.queue not in ACME_QUEUES:
            raise HTTPException(422, f"unknown queue {body.queue!r}")
        ticket = tickets[ticket_id]
        for field in ("queue", "priority", "status"):
            if (value := getattr(body, field)) is not None:
                ticket[field] = value
        if body.note:
            ticket["notes"].append(body.note)
        return ticket

    return router


def contoso_router(store: dict[str, Any]) -> APIRouter:
    router = APIRouter(prefix="/contoso/api/v2")
    cases = {c["case_ref"]: c for c in store["cases"]}

    @router.get("/cases/{case_ref}")
    def get_case(case_ref: str) -> dict[str, Any]:
        if case_ref not in cases:
            raise HTTPException(404, "case not found")
        return cases[case_ref]

    @router.patch("/cases/{case_ref}")
    def update_case(case_ref: str, body: CaseUpdate) -> dict[str, Any]:
        if case_ref not in cases:
            raise HTTPException(404, "case not found")
        if body.team and body.team not in CONTOSO_TEAMS:
            raise HTTPException(422, f"unknown team {body.team!r}")
        case = cases[case_ref]
        for field in ("state", "severity", "team"):
            if (value := getattr(body, field)) is not None:
                case[field] = value
        if body.comment:
            case["comments"].append(body.comment)
        return case

    return router


def create_app() -> FastAPI:
    app = FastAPI(title="Mock customer systems (synthetic data)")
    app.include_router(acme_router(copy.deepcopy(load_seed("acme-payments"))))
    app.include_router(contoso_router(copy.deepcopy(load_seed("contoso-wealth"))))

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
