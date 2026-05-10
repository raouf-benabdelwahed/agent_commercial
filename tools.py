import csv
import os
from datetime import datetime
from typing import Optional, Dict, Any

PRICING_CSV = os.path.join("data", "pricing.csv")
LEADS_CSV = "leads.csv"


def compute_quote(product: str, qty: int, option: Optional[str] = None) -> Dict[str, Any]:
    base_price = None
    option_price = 0.0

    with open(PRICING_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # prix de base (option vide)
    for r in rows:
        if r["product"] == product and r["option"] == "":
            base_price = float(r["base_price"])
            break

    if base_price is None:
        return {"ok": False, "error": "Produit inconnu"}

    # option (si fournie)
    if option:
        for r in rows:
            if r["product"] == product and r["option"] == option:
                option_price = float(r["option_price"])
                break

    unit_price = base_price + option_price
    total = unit_price * qty

    return {
        "ok": True,
        "product": product,
        "qty": qty,
        "option": option,
        "unit_price": unit_price,
        "total": total,
    }


def create_lead(name: str, email: str, need: str, quote_total: float) -> Dict[str, Any]:
    now = datetime.utcnow().isoformat()
    headers = ["timestamp", "name", "email", "need", "quote_total"]
    file_exists = os.path.exists(LEADS_CSV)

    with open(LEADS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            w.writeheader()
        w.writerow(
            {
                "timestamp": now,
                "name": name,
                "email": email,
                "need": need,
                "quote_total": quote_total,
            }
        )

    return {"ok": True, "timestamp": now}