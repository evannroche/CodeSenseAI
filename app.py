from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import ast
import traceback
import re
import sys
import logging # Import logging module
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure CORS for specific origins or allow all for development (adjust for production)
# For production, replace '*' with a list of allowed origins, e.g., origins=["http://localhost:3000"]
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'py'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@dataclass
class FunctionInfo:
    name: str
    line_start: int
    line_end: int
    args_count: int
    returns_count: int
    complexity: int
    calls: List[str]
    variables: Set[str]
    docstring: Optional[str] = None

@dataclass
class ClassInfo:
    name: str
    line_start: int
    line_end: int
    methods: List[str]
    attributes: Set[str]
    inheritance: List[str]
    docstring: Optional[str] = None

class EnhancedPythonAnalyzer(ast.NodeVisitor):
    """Enhanced Python code analyzer combining simple and advanced features."""
    
    def __init__(self, source_code: str, file_path: str = "input.py"):
        self.source_code = source_code
        self.file_path = file_path
        self.lines = source_code.split('\n')
        
        # Analysis results
        self.functions = []
        self.classes = []
        self.imports = []
        self.issues = []
        
        # Context tracking
        self.current_class = None
        self.current_function = None
        self.complexity_stack = [0]
        self.loop_depth = 0
        self.try_depth = 0
        
        # Pattern tracking
        self.variable_assignments = defaultdict(list)
        self.function_calls = defaultdict(list)
        self.string_literals = []
        self.numeric_literals = []
        
        # Line counting
        self.total_lines = len(self.lines)
        self.code_lines = 0
        self.comment_lines = 0
        self.blank_lines = 0
        self._count_lines()
    
    def analyze(self) -> Dict[str, Any]:
        """Main analysis entry point"""
        try:
            tree = ast.parse(self.source_code, filename=self.file_path)
            self.visit(tree)
            self._detect_additional_patterns()
            
            # Calculate metrics
            total_complexity = sum(f.complexity for f in self.functions)
            avg_complexity = total_complexity / len(self.functions) if self.functions else 0
            
            return {
                'file_path': self.file_path,
                'statistics': {
                    'total_lines': self.total_lines,
                    'code_lines': self.code_lines,
                    'comment_lines': self.comment_lines,
                    'blank_lines': self.blank_lines,
                    'function_count': len(self.functions),
                    'class_count': len(self.classes),
                    'complexity_score': round(avg_complexity, 2)
                },
                'issues': self.issues,
                'functions': [self._function_to_dict(f) for f in self.functions],
                'classes': [self._class_to_dict(c) for c in self.classes]
            }
            
        except SyntaxError as e:
            app.logger.error(f"Syntax error in {self.file_path}: {str(e)}") # Log error
            return {
                'file_path': self.file_path,
                'error': f'Python syntax error: {str(e)} at line {e.lineno}',
                'statistics': {
                    'total_lines': self.total_lines,
                    'code_lines': 0,
                    'comment_lines': 0,
                    'blank_lines': 0,
                    'function_count': 0,
                    'class_count': 0,
                    'complexity_score': 0
                },
                'issues': [],
                'functions': [],
                'classes': []
            }
        except Exception as e:
            app.logger.error(f"Analysis error for {self.file_path}: {e}", exc_info=True) # Log error with traceback
            return {
                'file_path': self.file_path,
                'error': f'Analysis error: An unexpected error occurred during analysis.', # Generic error for user
                'statistics': {
                    'total_lines': self.total_lines,
                    'code_lines': 0,
                    'comment_lines': 0,
                    'blank_lines': 0,
                    'function_count': 0,
                    'class_count': 0,
                    'complexity_score': 0
                },
                'issues': [],
                'functions': [],
                'classes': []
            }
    
    def _count_lines(self):
        """Count different types of lines"""
        for line in self.lines:
            stripped = line.strip()
            if not stripped:
                self.blank_lines += 1
            elif stripped.startswith('#'):
                self.comment_lines += 1
            else:
                self.code_lines += 1
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Analyze function definitions"""
        # Calculate complexity
        complexity = self._calculate_function_complexity(node)
        
        # Count returns
        returns_count = len([n for n in ast.walk(node) if isinstance(n, ast.Return)])
        
        # Get function calls
        calls = []
        variables = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and hasattr(child.func, 'id'):
                calls.append(child.func.id)
            elif isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                variables.add(child.id)
        
        func_info = FunctionInfo(
            name=node.name,
            line_start=node.lineno,
            line_end=getattr(node, 'end_lineno', node.lineno + len(node.body)),
            args_count=len(node.args.args),
            returns_count=returns_count,
            complexity=complexity,
            calls=calls,
            variables=variables,
            docstring=ast.get_docstring(node)
        )
        
        # Add to current class or global functions
        if self.current_class:
            self.current_class.methods.append(func_info.name)
        
        self.functions.append(func_info)
        self._check_function_issues(func_info)
        
        # Visit children
        old_function = self.current_function
        self.current_function = func_info
        self.generic_visit(node)
        self.current_function = old_function
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Handle async functions"""
        self.visit_FunctionDef(node)
    
    def visit_ClassDef(self, node: ast.ClassDef):
        """Analyze class definitions"""
        class_info = ClassInfo(
            name=node.name,
            line_start=node.lineno,
            line_end=getattr(node, 'end_lineno', node.lineno + len(node.body)),
            methods=[],
            attributes=set(),
            inheritance=[base.id for base in node.bases if hasattr(base, 'id')],
            docstring=ast.get_docstring(node)
        )
        
        # Find attributes
        for child in ast.walk(node):
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                        if target.value.id == 'self':
                            class_info.attributes.add(target.attr)
        
        old_class = self.current_class
        self.current_class = class_info
        
        self.classes.append(class_info)
        self._check_class_issues(class_info)
        
        self.generic_visit(node)
        self.current_class = old_class
    
    def visit_Import(self, node: ast.Import):
        """Track imports"""
        for alias in node.names:
            self.imports.append(alias.name)
            self._check_import_security(alias.name, node.lineno)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Track from imports"""
        module = node.module or ""
        for alias in node.names:
            import_name = f"{module}.{alias.name}"
            self.imports.append(import_name)
            self._check_import_security(import_name, node.lineno)
        self.generic_visit(node)
    
    def visit_Call(self, node: ast.Call):
        """Check function calls for security and pattern issues"""
        if hasattr(node.func, 'id'):
            func_name = node.func.id
            self.function_calls[func_name].append(node.lineno)
            self._check_dangerous_calls(func_name, node.lineno)
        self.generic_visit(node)
    
    def visit_Assign(self, node: ast.Assign):
        """Check assignments for security issues"""
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.variable_assignments[target.id].append(node.lineno)
                # Check for hardcoded secrets
                if isinstance(node.value, (ast.Str, ast.Constant)):
                    value = node.value.s if hasattr(node.value, 's') else str(node.value.value)
                    if self._is_potential_secret(target.id, value):
                        self._add_issue('security_hardcoded_secret', 'error',
                                      f'Potential hardcoded secret in variable {target.id}',
                                      node.lineno,
                                      'Use environment variables or secure configuration files')
        self.generic_visit(node)
    
    def visit_Try(self, node: ast.Try):
        """Check exception handling patterns"""
        self.try_depth += 1
        
        # Check for bare except clauses
        for handler in node.handlers:
            if handler.type is None:
                self._add_issue('bare_except', 'warning',
                              'Bare except clause catches all exceptions',
                              handler.lineno,
                              'Catch specific exceptions instead of using bare except')
            
            # Check for empty except blocks
            if len(handler.body) == 1 and isinstance(handler.body[0], ast.Pass):
                self._add_issue('empty_except', 'warning',
                              'Empty except block',
                              handler.lineno,
                              'Handle the exception appropriately or log it')
        
        self.generic_visit(node)
        self.try_depth -= 1
    
    def visit_Str(self, node: ast.Str):
        """Check string literals"""
        self.string_literals.append((node.s, node.lineno))
        self._check_string_patterns(node.s, node.lineno)
        self.generic_visit(node)
    
    def visit_Constant(self, node: ast.Constant):
        """Check constants (Python 3.8+)"""
        if isinstance(node.value, str):
            self.string_literals.append((node.value, node.lineno))
            self._check_string_patterns(node.value, node.lineno)
        elif isinstance(node.value, (int, float)):
            self.numeric_literals.append((node.value, node.lineno))
            self._check_magic_numbers(node.value, node.lineno)
        self.generic_visit(node)
    
    def visit_Num(self, node: ast.Num):
        """Check numeric literals (Python < 3.8)"""
        self.numeric_literals.append((node.n, node.lineno))
        self._check_magic_numbers(node.n, node.lineno)
        self.generic_visit(node)
    
    def _calculate_function_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity for a function"""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler,
                                ast.With, ast.AsyncWith, ast.ListComp, ast.DictComp,
                                ast.SetComp, ast.GeneratorExp)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _check_function_issues(self, func_info: FunctionInfo):
        """Check for function-specific issues"""
        # Too many parameters
        if func_info.args_count > 7:
            self._add_issue('too_many_parameters', 'warning',
                          f'Function {func_info.name} has too many parameters ({func_info.args_count})',
                          func_info.line_start,
                          'Consider reducing parameters or using a configuration object')
        
        # High complexity
        if func_info.complexity > 10:
            self._add_issue('too_complex', 'warning',
                          f'Function {func_info.name} is too complex (complexity: {func_info.complexity})',
                          func_info.line_start,
                          'Consider breaking this function into smaller functions')
        
        # Long function
        function_length = func_info.line_end - func_info.line_start
        if function_length > 50:
            self._add_issue('long_function', 'warning',
                          f'Function {func_info.name} is too long ({function_length} lines)',
                          func_info.line_start,
                          'Consider breaking this function into smaller functions')
        
        # Missing docstring for public functions
        if not func_info.docstring and not func_info.name.startswith('_'):
            self._add_issue('missing_docstring', 'info',
                          f'Public function {func_info.name} missing docstring',
                          func_info.line_start,
                          'Add a docstring to document the function purpose')
        
        # Too many return statements
        if func_info.returns_count > 3:
            self._add_issue('multiple_returns', 'info',
                          f'Function {func_info.name} has multiple return statements ({func_info.returns_count})',
                          func_info.line_start,
                          'Consider refactoring to reduce complexity')
    
    def _check_class_issues(self, class_info: ClassInfo):
        """Check for class-specific issues"""
        method_count = len(class_info.methods)
        
        # Too many methods
        if method_count > 20:
            self._add_issue('too_many_methods', 'warning',
                          f'Class {class_info.name} has too many methods ({method_count})',
                          class_info.line_start,
                          'Consider splitting this class into multiple smaller classes')
        
        # No methods (data class candidate)
        if method_count == 0:
            self._add_issue('no_methods', 'info',
                          f'Class {class_info.name} has no methods',
                          class_info.line_start,
                          'Consider using a dataclass or namedtuple')
        
        # Missing docstring
        if not class_info.docstring:
            self._add_issue('missing_docstring', 'info',
                          f'Class {class_info.name} missing docstring',
                          class_info.line_start,
                          'Add a docstring to document the class purpose')
    
    def _check_dangerous_calls(self, func_name: str, line: int):
        """Check for dangerous function calls"""
        dangerous_functions = {
            'eval': 'Use ast.literal_eval() for safe evaluation or find alternative approaches',
            'exec': 'Avoid dynamic code execution or use safer alternatives',
            'input': 'Validate and sanitize user input',
            '__import__': 'Use importlib for dynamic imports'
        }
        
        if func_name in dangerous_functions:
            self._add_issue(f'security_{func_name}', 'error',
                          f'Use of {func_name}() is dangerous and should be avoided',
                          line,
                          dangerous_functions[func_name])
        
        # Print statements
        if func_name == 'print':
            self._add_issue('print_statement', 'info',
                          'Print statement found',
                          line,
                          'Use logging instead of print for production code')
    
    def _check_import_security(self, import_name: str, line: int):
        """Check for potentially insecure imports"""
        insecure_modules = ['pickle', 'marshal', 'shelve']
        if any(module in import_name for module in insecure_modules):
            self._add_issue('insecure_import', 'info',
                          f'Potentially insecure module import: {import_name}',
                          line,
                          'Be cautious when deserializing untrusted data')
    
    def _check_string_patterns(self, string_value: str, line: int):
        """Check string patterns for security issues"""
        # SQL injection patterns
        sql_patterns = [
            r'SELECT\s+.*\s+FROM\s+',
            r'INSERT\s+INTO\s+',
            r'UPDATE\s+.*\s+SET\s+',
            r'DELETE\s+FROM\s+'
        ]
        
        for pattern in sql_patterns:
            if re.search(pattern, string_value, re.IGNORECASE):
                if '+' in string_value:  # String concatenation
                    self._add_issue('sql_injection', 'error',
                                  'Possible SQL injection vulnerability',
                                  line,
                                  'Use parameterized queries instead of string concatenation')
    
    def _check_magic_numbers(self, number, line: int):
        """Check for magic numbers"""
        common_numbers = {0, 1, 2, 10, 100, 1000}
        if isinstance(number, (int, float)) and number not in common_numbers:
            if abs(number) > 10:
                self._add_issue('magic_number', 'info',
                              f'Magic number: {number}',
                              line,
                              'Consider defining as a named constant')
    
    def _is_potential_secret(self, var_name: str, value: str) -> bool:
        """Check if a variable might contain a secret"""
        secret_keywords = ['password', 'secret', 'key', 'token', 'api_key']
        return (any(keyword in var_name.lower() for keyword in secret_keywords) and
                len(str(value)) > 5)
    
    def _detect_additional_patterns(self):
        """Detect additional patterns in the code"""
        # Check for long lines
        for i, line in enumerate(self.lines, 1):
            if len(line) > 120:
                self._add_issue('long_line', 'info',
                              f'Line too long ({len(line)} characters)',
                              i,
                              'Break long lines for better readability')
            
            # Check for TODO comments
            if any(keyword in line.upper() for keyword in ['TODO', 'FIXME', 'HACK']):
                self._add_issue('todo_comment', 'info',
                              'TODO/FIXME comment found',
                              i,
                              'Address the TODO item or create a proper issue')
    
    def _add_issue(self, issue_type: str, severity: str, message: str, line: int, suggestion: str = ""):
        """Add an issue to the issues list"""
        issue = {
            'type': issue_type,
            'severity': severity,
            'message': message,
            'line': line,
            'column': 0,
            'function': self.current_function.name if self.current_function else None,
            'class_name': self.current_class.name if self.current_class else None,
            'suggestion': suggestion,
            'rule_id': issue_type
        }
        self.issues.append(issue)
    
    def _function_to_dict(self, func: FunctionInfo) -> Dict[str, Any]:
        """Convert FunctionInfo to dictionary"""
        return {
            'name': func.name,
            'line_start': func.line_start,
            'line_end': func.line_end,
            'args_count': func.args_count,
            'complexity': func.complexity,
            'has_docstring': bool(func.docstring)
        }
    
    def _class_to_dict(self, cls: ClassInfo) -> Dict[str, Any]:
        """Convert ClassInfo to dictionary"""
        return {
            'name': cls.name,
            'line_start': cls.line_start,
            'line_end': cls.line_end,
            'method_count': len(cls.methods),
            'has_docstring': bool(cls.docstring),
            'inheritance': cls.inheritance
        }

def analyze_python_code_enhanced(source_code: str, file_path: str = "input.py") -> dict:
    """Enhanced Python code analyzer."""
    analyzer = EnhancedPythonAnalyzer(source_code, file_path)
    return analyzer.analyze()

# Add security headers
@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    # IMPORTANT: 'unsafe-inline' for script-src is added to allow existing inline event handlers in index.html.
    # For true production readiness, refactor frontend to avoid inline scripts and remove 'unsafe-inline'.
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self';"
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response

@app.route('/')
def index():
    """Serve the main HTML interface."""
    return send_from_directory('.', 'index.html')

@app.route('/api/health')
def health_check():
    """API health check endpoint."""
    app.logger.info("Health check requested.") # Log request
    return jsonify({
        'status': 'healthy',
        'message': 'Python Code Analyzer API is running (Enhanced Mode)',
        'version': '2.0.0',
        'analyzer': 'enhanced'
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_files():
    """Analyze uploaded Python files."""
    try:
        if 'files' not in request.files:
            app.logger.warning("No files uploaded in analyze_files request.") # Log warning
            return jsonify({'success': False, 'error': 'No files uploaded'}), 400
        
        files = request.files.getlist('files')
        if not files or all(file.filename == '' for file in files):
            app.logger.warning("No files selected for analysis.") # Log warning
            return jsonify({'success': False, 'error': 'No files selected'}), 400
        
        results = []
        total_lines = 0
        total_issues = 0
        complexity_scores = []
        
        for file in files:
            if file and allowed_file(file.filename):
                try:
                    # Read file content
                    source_code = file.read().decode('utf-8')
                    filename = secure_filename(file.filename)
                    app.logger.info(f"Analyzing uploaded file: {filename}") # Log file analysis
                    
                    # Analyze the code
                    result = analyze_python_code_enhanced(source_code, filename)
                    results.append(result)
                    
                    # Update summary statistics
                    if 'statistics' in result and result['statistics']:
                        total_lines += result['statistics'].get('total_lines', 0)
                        total_issues += len(result.get('issues', []))
                        complexity_scores.append(result['statistics'].get('complexity_score', 0))
                    
                except UnicodeDecodeError:
                    app.logger.error(f"File encoding error for {file.filename}.") # Log error
                    results.append({
                        'file_path': file.filename,
                        'error': 'File encoding error. Please ensure the file is UTF-8 encoded.',
                        'statistics': {
                            'total_lines': 0,
                            'code_lines': 0,
                            'comment_lines': 0,
                            'blank_lines': 0,
                            'function_count': 0,
                            'class_count': 0,
                            'complexity_score': 0
                        },
                        'issues': [],
                        'functions': [],
                        'classes': []
                    })
                except Exception as e:
                    app.logger.error(f"Error processing file {file.filename}: {e}", exc_info=True) # Log error with traceback
                    results.append({
                        'file_path': file.filename,
                        'error': f'Error processing file: An unexpected error occurred.', # Generic error for user
                        'statistics': {
                            'total_lines': 0,
                            'code_lines': 0,
                            'comment_lines': 0,
                            'blank_lines': 0,
                            'function_count': 0,
                            'class_count': 0,
                            'complexity_score': 0
                        },
                        'issues': [],
                        'functions': [],
                        'classes': []
                    })
        
        # Calculate summary
        avg_complexity = sum(complexity_scores) / len(complexity_scores) if complexity_scores else 0
        
        app.logger.info(f"Successfully analyzed {len(results)} files.") # Log success
        return jsonify({
            'success': True,
            'summary': {
                'total_files': len(results),
                'total_lines': total_lines,
                'total_issues': total_issues,
                'average_complexity': round(avg_complexity, 2)
            },
            'files': results
        })
        
    except Exception as e:
        app.logger.error(f"Error in analyze_files endpoint: {e}", exc_info=True) # Log server error with traceback
        return jsonify({
            'success': False, 
            'error': f'Server error: An unexpected server error occurred.' # Generic error for user
        }), 500

@app.route('/api/analyze-text', methods=['POST'])
def analyze_text():
    """Analyze Python code from text input."""
    try:
        data = request.get_json()
        if not data or 'code' not in data:
            app.logger.warning("No code provided in analyze_text request.") # Log warning
            return jsonify({'success': False, 'error': 'No code provided'}), 400
        
        source_code = data['code']
        filename = data.get('filename', 'pasted_code.py')
        
        if not source_code.strip():
            app.logger.warning("Empty code provided for analysis.") # Log warning
            return jsonify({'success': False, 'error': 'Empty code provided'}), 400
        
        app.logger.info(f"Analyzing pasted code (filename: {filename}).") # Log pasted code analysis
        # Analyze the code
        result = analyze_python_code_enhanced(source_code, filename)
        
        # Calculate summary
        total_lines = result['statistics'].get('total_lines', 0) if 'statistics' in result else 0
        total_issues = len(result.get('issues', []))
        complexity_score = result['statistics'].get('complexity_score', 0) if 'statistics' in result else 0
        
        return jsonify({
            'success': True,
            'summary': {
                'total_files': 1,
                'total_lines': total_lines,
                'total_issues': total_issues,
                'average_complexity': complexity_score
            },
            'files': [result]
        })
        
    except Exception as e:
        app.logger.error(f"Error in analyze_text endpoint: {e}", exc_info=True) # Log server error with traceback
        return jsonify({
            'success': False, 
            'error': f'Server error: An unexpected server error occurred.' # Generic error for user
        }), 500

@app.errorhandler(413)
def too_large(e):
    app.logger.warning("Request entity too large (413).") # Log 413 error
    return jsonify({'success': False, 'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    app.logger.warning(f"Endpoint not found: {request.url} (404).") # Log 404 error
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    app.logger.error(f"Internal server error (500): {e}", exc_info=True) # Log 500 error with traceback
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.logger.info("🐍 Starting Python Code Analyzer Web Server (Enhanced Mode)...")
    app.logger.info("📁 Make sure index.html is in the same directory as this file")
    app.logger.info("🌐 Open your browser to: http://localhost:5000")
    app.logger.info("⚙️  Using enhanced analyzer with advanced pattern detection")
    app.logger.info("\nFeatures:")
    app.logger.info("  ✅ Cyclomatic complexity calculation")
    app.logger.info("  ✅ Security vulnerability detection")
    app.logger.info("  ✅ Code smell detection")
    app.logger.info("  ✅ Documentation coverage analysis")
    app.logger.info("  ✅ Pattern-based issue detection")
    app.logger.info("\nStarting server...")
    app.run(debug=False, host='0.0.0.0', port=5000)
