Python Code Analyzer
A web application that analyzes Python code for quality, complexity, and potential issues. This tool helps developers improve their code by providing detailed reports on various metrics and identifying areas for improvement.

Key Features
Code Analysis: Analyzes Python code by parsing its Abstract Syntax Tree (AST) to generate metrics on total lines, code lines, function/class counts, and cyclomatic complexity.

Issue Detection: Scans code for common issues such as high complexity, overly long functions, missing docstrings, bare except clauses, and potential security vulnerabilities like hardcoded secrets and dangerous function calls (eval, exec).

User-Friendly Interface: Provides a modern, responsive front-end for users to either upload .py files or paste code directly into the application for immediate analysis.

Web Application: The project is a web-based service, accessible via a URL, and hosted on a platform like Heroku.

Technologies Used
Backend: Flask, Python

Frontend: HTML, CSS, JavaScript

Core Logic: Python's ast module for code parsing

Deployment: Heroku
