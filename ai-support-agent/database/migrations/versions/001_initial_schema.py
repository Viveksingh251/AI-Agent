"""Initial database migration: Customer, Order, SupportTicket, TicketMessage, Employee, AgentActionLog.

Revision ID: 001_initial_schema
Revises: 
Create Date: 2026-11-01 12:00:00.000000
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "001_initial_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Create 'customers' table
    op.create_table(
        "customers",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("phone", sa.String(length=32), nullable=True),
        sa.Column("company", sa.String(length=128), nullable=True),
        sa.Column("tier", sa.String(length=32), nullable=False, server_default="free"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_customers_email", "customers", ["email"], unique=True)
    op.create_index("ix_customers_name", "customers", ["name"], unique=False)
    op.create_index("ix_customers_company", "customers", ["company"], unique=False)
    op.create_index("ix_customers_tier", "customers", ["tier"], unique=False)
    op.create_index("ix_customers_created_at", "customers", ["created_at"], unique=False)

    # 2. Create 'employees' table
    op.create_table(
        "employees",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("role", sa.String(length=32), nullable=False, server_default="support"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_employees_email", "employees", ["email"], unique=True)
    op.create_index("ix_employees_name", "employees", ["name"], unique=False)
    op.create_index("ix_employees_role", "employees", ["role"], unique=False)
    op.create_index("ix_employees_is_active", "employees", ["is_active"], unique=False)

    # 3. Create 'orders' table
    op.create_table(
        "orders",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("customer_id", sa.Integer(), nullable=False),
        sa.Column("order_number", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="pending"),
        sa.Column("total_amount", sa.Numeric(precision=10, scale=2), nullable=False, server_default="0.00"),
        sa.Column("items", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["customer_id"], ["customers.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_orders_order_number", "orders", ["order_number"], unique=True)
    op.create_index("ix_orders_customer_id", "orders", ["customer_id"], unique=False)
    op.create_index("ix_orders_status", "orders", ["status"], unique=False)
    op.create_index("ix_orders_created_at", "orders", ["created_at"], unique=False)
    op.create_index("idx_orders_customer_status", "orders", ["customer_id", "status"], unique=False)

    # 4. Create 'support_tickets' table
    op.create_table(
        "support_tickets",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("ticket_number", sa.String(length=64), nullable=False),
        sa.Column("customer_id", sa.Integer(), nullable=False),
        sa.Column("subject", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="open"),
        sa.Column("priority", sa.String(length=32), nullable=False, server_default="medium"),
        sa.Column("assigned_to", sa.Integer(), nullable=True),
        sa.Column("created_by_agent", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["customer_id"], ["customers.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["assigned_to"], ["employees.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_support_tickets_ticket_number", "support_tickets", ["ticket_number"], unique=True)
    op.create_index("ix_support_tickets_customer_id", "support_tickets", ["customer_id"], unique=False)
    op.create_index("ix_support_tickets_status", "support_tickets", ["status"], unique=False)
    op.create_index("ix_support_tickets_priority", "support_tickets", ["priority"], unique=False)
    op.create_index("ix_support_tickets_assigned_to", "support_tickets", ["assigned_to"], unique=False)
    op.create_index("ix_support_tickets_created_at", "support_tickets", ["created_at"], unique=False)
    op.create_index("idx_tickets_customer_status", "support_tickets", ["customer_id", "status"], unique=False)
    op.create_index("idx_tickets_assigned_status", "support_tickets", ["assigned_to", "status"], unique=False)

    # 5. Create 'ticket_messages' table
    op.create_table(
        "ticket_messages",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("ticket_id", sa.Integer(), nullable=False),
        sa.Column("sender_type", sa.String(length=32), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["ticket_id"], ["support_tickets.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_ticket_messages_ticket_id", "ticket_messages", ["ticket_id"], unique=False)
    op.create_index("ix_ticket_messages_sender_type", "ticket_messages", ["sender_type"], unique=False)
    op.create_index("ix_ticket_messages_created_at", "ticket_messages", ["created_at"], unique=False)

    # 6. Create 'agent_action_logs' table
    op.create_table(
        "agent_action_logs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("customer_id", sa.Integer(), nullable=True),
        sa.Column("action_type", sa.String(length=64), nullable=False),
        sa.Column("details", sa.JSON(), nullable=False),
        sa.Column("success", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["customer_id"], ["customers.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_agent_action_logs_customer_id", "agent_action_logs", ["customer_id"], unique=False)
    op.create_index("ix_agent_action_logs_action_type", "agent_action_logs", ["action_type"], unique=False)
    op.create_index("ix_agent_action_logs_success", "agent_action_logs", ["success"], unique=False)
    op.create_index("ix_agent_action_logs_created_at", "agent_action_logs", ["created_at"], unique=False)
    op.create_index("idx_action_logs_customer_created", "agent_action_logs", ["customer_id", "created_at"], unique=False)


def downgrade() -> None:
    # Drop tables in reverse dependency order
    op.drop_table("agent_action_logs")
    op.drop_table("ticket_messages")
    op.drop_table("support_tickets")
    op.drop_table("orders")
    op.drop_table("employees")
    op.drop_table("customers")

