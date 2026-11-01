"""Setup configuration for editable and standard package installation."""

from setuptools import find_packages, setup

setup(
    name="ai-support-agent",
    version="0.1.0",
    description="Customer Support AI Agent using LangGraph, FastAPI, and RAG",
    packages=find_packages(),
    python_requires=">=3.11",
)

