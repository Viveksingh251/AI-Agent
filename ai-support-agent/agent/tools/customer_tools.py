"""Customer Record and Order History Lookup Tools with Sensitive Data Clearance."""

from typing import Any, Dict, List, Optional
from langchain_core.tools import tool


@tool
def get_customer_profile_tool(
    customer_id: str,
    caller_role: str = "customer",
    caller_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Retrieves customer account profile and subscription status.
    
    Customers can only inspect their own profile. Support staff and Admins can view any customer profile.
    """
    # Permission verification
    if caller_role == "customer" and caller_id and str(caller_id) != str(customer_id):
        return {
            "error": "Permission Denied: Customers cannot inspect other accounts.",
            "authorized": False,
        }

    return {
        "customer_id": customer_id,
        "name": "Jane Doe",
        "email": "jane.doe@example.com",
        "tier": "Gold Member",
        "account_status": "active",
        "authorized": True,
    }


@tool
def get_order_history_tool(
    customer_id: str,
    limit: int = 5,
    caller_role: str = "customer",
    caller_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Fetches recent orders placed by a customer.
    
    Customers can only view their own orders; staff can view orders for any customer.
    """
    if caller_role == "customer" and caller_id and str(caller_id) != str(customer_id):
        return [{"error": "Permission Denied: Customers cannot view orders of other users."}]

    return [
        {
            "order_id": "ORD-8821",
            "date": "2026-10-25",
            "status": "delivered",
            "items": ["Ergonomic Wireless Keyboard"],
            "total_usd": 129.99,
            "tracking_number": "TRK-99281928",
        }
    ]


@tool
def view_sensitive_financials_tool(
    customer_id: str,
    caller_role: str = "customer",
) -> Dict[str, Any]:
    """Inspects credit balances, transaction chargebacks, and payment methods.
    
    Strictly restricted to 'support_agent' or 'admin' roles. Denied to regular customers.
    """
    if caller_role not in ["support_agent", "admin"]:
        return {
            "error": "Access Denied: Sensitive financial telemetry is restricted to support_agent and admin roles.",
            "authorized": False,
        }

    return {
        "customer_id": customer_id,
        "credit_balance_usd": 45.00,
        "last_payment_method": "Visa ending in 4022",
        "chargeback_risk_score": "low",
        "authorized": True,
    }
