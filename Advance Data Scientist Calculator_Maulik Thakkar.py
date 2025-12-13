import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import math
import re
import numpy as np
import pandas as pd
import datetime
from datetime import datetime

def _parse_date_flexible(s):
    s = s.strip()
    formats = [
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%Y/%m/%d",
        "%d.%m.%Y",
    ]
    for f in formats:
        try:
            return datetime.strptime(s, f)
        except Exception:
            pass
    raise ValueError("Invalid date format")


import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

import webbrowser
import base64
from io import BytesIO


BG = "#1e1e1e"
FG = "#ffffff"
BTN = "#333333"
ACCENT = "#4ea1d3"

MAX_HISTORY = 50
CALC_HISTORY = []

FONT_SMALL = ("Segoe UI", 10)
FONT_MED = ("Segoe UI", 12)
FONT_LARGE = ("Segoe UI", 16, "bold")
FONT_TITLE = ("Segoe UI", 22, "bold")

SIGNATURE_TEXT = "Developed by Maulik Thakkar"

SIGNATURE_FONT = ("Segoe Script", 13, "bold")

def add_to_history(calc_type, expression, result):
    """Store calculation history (keeps last MAX_HISTORY)."""
    try:
        ts = datetime.now().strftime("%H:%M:%S")
    except Exception:
        ts = ""
    CALC_HISTORY.append({"type": calc_type, "expr": str(expression), "result": str(result), "time": ts})
    if len(CALC_HISTORY) > MAX_HISTORY:
        del CALC_HISTORY[0:len(CALC_HISTORY)-MAX_HISTORY]


class HistoryPanel:
    """Reusable history list panel."""
    def __init__(self, parent, on_select, *, title="History", width=34, height=18):
        self.on_select = on_select
        self.frame = tk.LabelFrame(parent, text=title, bg=BG, fg=FG, font=FONT_MED, bd=1, relief="groove")
        self.frame.pack(side="right", fill="y", padx=(8, 12), pady=(6, 10))

        list_frame = tk.Frame(self.frame, bg=BG)
        list_frame.pack(fill="both", expand=True, padx=6, pady=6)

        self.listbox = tk.Listbox(
            list_frame,
            width=width,
            height=height,
            bg="#000000",
            fg=FG,
            selectbackground=ACCENT,
            selectforeground="black",
            activestyle="none",
            font=("Consolas", 11)
        )
        self.listbox.pack(side="left", fill="both", expand=True)

        yscroll = tk.Scrollbar(list_frame, orient="vertical", command=self.listbox.yview)
        yscroll.pack(side="right", fill="y")
        self.listbox.configure(yscrollcommand=yscroll.set)

        btn_row = tk.Frame(self.frame, bg=BG)
        btn_row.pack(fill="x", padx=6, pady=(0, 6))
        tk.Button(btn_row, text="Clear", bg=BTN, fg=FG, width=10, command=self.clear).pack(side="left")
        tk.Button(btn_row, text="Refresh", bg=BTN, fg=FG, width=10, command=self.refresh).pack(side="left", padx=(6, 0))

        self.listbox.bind("<<ListboxSelect>>", self._select)
        self.refresh()

    def refresh(self):
        self.listbox.delete(0, tk.END)
        # show newest first
        for item in reversed(CALC_HISTORY):
            txt = f"[{item.get('time','')}] {item.get('expr','')} = {item.get('result','')}"
            self.listbox.insert(tk.END, txt)

    def clear(self):
        CALC_HISTORY.clear()
        self.refresh()

    def _select(self, _event=None):
        sel = self.listbox.curselection()
        if not sel:
            return
        idx = len(CALC_HISTORY) - 1 - sel[0]
        if 0 <= idx < len(CALC_HISTORY):
            self.on_select(CALC_HISTORY[idx])


def make_scrolled_text(parent, height=12, width=120):
    """Create a black output text area with consistent font + scrollbars."""
    frame = tk.Frame(parent, bg=BG)
    frame.pack(pady=10, fill="both", expand=True)

    text = tk.Text(
        frame,
        height=height,
        width=width,
        bg="#000000",
        fg=FG,
        insertbackground=FG,
        font=("Consolas", 14),
        wrap="none",
        padx=10,
        pady=10
    )
    text.pack(side="left", fill="both", expand=True)

    yscroll = tk.Scrollbar(frame, orient="vertical", command=text.yview)
    xscroll = tk.Scrollbar(frame, orient="horizontal", command=text.xview)
    text.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
    yscroll.pack(side="right", fill="y")
    xscroll.pack(side="bottom", fill="x")
    return frame, text

_ALLOWED_MATH = {
    "math": math,
    "np": np,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow
}

def safe_eval(expr: str):
    """Safer eval for calculator-like expressions."""
    return eval(expr, {"__builtins__": {}}, _ALLOWED_MATH)

class AdvancedCalculatorApp:
    def __init__(self, root):
        self.root = root
        root.title("Advanced Data Science Calculator")
        root.geometry("1500x900")
        root.configure(bg=BG)

        style = ttk.Style()
        style.theme_use("default")
        style.configure("TNotebook", background=BG)
        style.configure("TNotebook.Tab", background=BTN, foreground=FG, padding=8)
        style.map("TNotebook.Tab", background=[("selected", ACCENT)])

        root.grid_rowconfigure(0, weight=1) 
        root.grid_rowconfigure(1, weight=0)   
        root.grid_columnconfigure(0, weight=1)

        self.notebook = ttk.Notebook(root)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        footer = tk.Frame(root, bg=BG)
        footer.grid(row=1, column=0, sticky="ew")
        footer.grid_columnconfigure(0, weight=1)

        signature_font = ("Segoe Script", 13, "italic")
        try:
            import tkinter.font as tkfont
            available = set(tkfont.families())
            if "Segoe Script" not in available:
                signature_font = ("Lucida Handwriting", 12, "italic") if "Lucida Handwriting" in available else ("Segoe UI", 11, "italic")
        except Exception:
            signature_font = ("Segoe UI", 11, "italic")

        sig = tk.Label(
            footer,
            text=SIGNATURE_TEXT,
            bg=BG,
            fg="#e9e13d",          
            font=signature_font
        )
        sig.grid(row=0, column=0, sticky="e", padx=18, pady=(1, 2))


        self._add_tab("Basic Calculator", BasicCalculatorTab)
        self._add_tab("Scientific Calculator", ScientificCalculatorTab)
        self._add_tab("Matrix Tools", MatrixTab)
        self._add_tab("Data Tools", DataTab)
        self._add_tab("Statistics", StatisticsTab)
        self._add_tab("Machine Learning", MLTab)
        self._add_tab("Finance", FinanceTab)
        self._add_tab("Forecasting", ForecastTab)
        self._add_tab("Plotting", PlotTab)
        self._add_tab("Converters", ConverterTab)
        self._add_tab("Formula Editor", FormulaEditorTab)
        self._add_tab("Reports", ReportTab)

        self.root.after(200, lambda: self.focus_basic())

    def _add_tab(self, name, TabClass):
        page = tk.Frame(self.notebook, bg=BG)
        self.notebook.add(page, text=name)

        page.grid_rowconfigure(0, weight=1)
        page.grid_columnconfigure(0, weight=1)
        inner = tk.Frame(page, bg=BG)
        inner.grid(row=0, column=0, sticky="nsew")

        tab_obj = TabClass(inner)
        if name == "Basic Calculator":
            self.basic_tab = tab_obj

    def focus_basic(self):
        try:
            self.notebook.select(0)
            if hasattr(self, "basic_tab") and hasattr(self.basic_tab, "focus_input"):
                self.basic_tab.focus_input()
        except Exception:
            pass

class BasicCalculatorTab:
    def __init__(self, parent):
        self.parent = parent
        self.expression = ""
        self.just_evaluated = False
        self.memory = 0.0

        main = tk.Frame(parent, bg=BG)
        main.pack(fill="both", expand=True)

        left = tk.Frame(main, bg=BG)
        left.pack(side="left", fill="both", expand=True)

        self.history = HistoryPanel(main, self.load_from_history, title="History (Basic)", width=36, height=20)

        tk.Label(left, text="Basic Calculator", bg=BG, fg=FG, font=FONT_TITLE).pack(pady=8)
        self.display_var = tk.StringVar(value="")
        self.display = tk.Entry(
            left, textvariable=self.display_var,
            font=("Segoe UI", 24), bd=4, relief="sunken",
            bg="white", fg="black", justify="right",
            state="readonly"
        )
        self.display.pack(fill="x", padx=20, pady=(6, 10), ipady=10)

        btns = [
            ["7", "8", "9", "/", "MC"],
            ["4", "5", "6", "*", "M+"],
            ["1", "2", "3", "-", "M-"],
            ["0", ".", "C", "+", "MR"],
            ["(", ")", "%", "//", "="],
        ]

        frame = tk.Frame(left, bg=BG)
        frame.pack(pady=8)

        for row in btns:
            row_frame = tk.Frame(frame, bg=BG)
            row_frame.pack()
            for btn in row:
                tk.Button(
                    row_frame, text=btn, width=8, height=2,
                    bg=BTN, fg=FG, font=FONT_MED,
                    command=lambda b=btn: self.on_press(b)
                ).pack(side="left", padx=5, pady=4)
        self.display.focus_set()
        self.display.bind("<KeyPress>", self.handle_key)
        self.display.bind("<<Paste>>", self.handle_paste)

    def focus_input(self):
        self.display.focus_set()

    def load_from_history(self, item):
        expr = str(item.get("expr", ""))
        if expr:
            self.expression = expr
            self.update_display()

    def update_display(self):
        self.display_var.set(self.expression)

    def on_press(self, value):
        if value == "=":
            self.calculate()
            return
        if value == "C":
            self.expression = ""
            self.update_display()
            return

        if value == "M+":
            try:
                self.memory += float(self._safe_eval(self.expression or "0"))
            except Exception:
                pass
            return
        if value == "M-":
            try:
                self.memory -= float(self._safe_eval(self.expression or "0"))
            except Exception:
                pass
            return
        if value == "MR":
            self.expression = str(self.memory)
            self.update_display()
            return
        if value == "MC":
            self.memory = 0.0
            return
        if self.just_evaluated and (str(value).isdigit() or value == "."):
            self.expression = ""
        self.just_evaluated = False
        self.expression += value
        self.update_display()

    def _safe_eval(self, expr: str):
        return eval(expr, {"__builtins__": {}}, {"math": math})

    def calculate(self):
        expr = self.expression
        try:
            result = self._safe_eval(expr)
            self.expression = str(result)
            # history
            add_to_history("Basic", expr, result)
            self.history.refresh()
            self.just_evaluated = True
        except Exception:
            self.expression = "Error"
            self.just_evaluated = True
            self.just_evaluated = True
        self.update_display()

    def handle_paste(self, _event):
        try:
            s = self.parent.clipboard_get()
        except Exception:
            return "break"
        allowed = "0123456789+-*/().% "
        cleaned = "".join(ch for ch in s if ch in allowed).replace(" ", "")
        if cleaned:
            self.expression += cleaned
            self.update_display()
        return "break"

    def handle_key(self, event):
        key = event.char

        if key in "0123456789+-*/().%":
            if self.just_evaluated and (key.isdigit() or key == "."):
                self.expression = ""
            self.just_evaluated = False
            self.expression += key
            self.update_display()
            return "break"

        if event.keysym in ("Return", "KP_Enter"):
            self.calculate()
            return "break"

        if event.keysym == "BackSpace":
            self.expression = self.expression[:-1]
            self.update_display()
            return "break"

        if event.keysym == "Escape":
            self.expression = ""
            self.update_display()
            return "break"

        return "break"



class ScientificCalculatorTab:
    def __init__(self, parent):
        self.parent = parent
        self.expression = ""
        self.just_evaluated = False
        self.angle_mode = "DEG"
        main = tk.Frame(parent, bg=BG)
        main.pack(fill="both", expand=True)

        left = tk.Frame(main, bg=BG)
        left.pack(side="left", fill="both", expand=True)

        self.history = HistoryPanel(main, self.load_from_history, title="History (Sci)", width=36, height=20)

        tk.Label(left, text="Scientific Calculator", bg=BG, fg=FG, font=FONT_TITLE).pack(pady=4)

        self.display_var = tk.StringVar(value="")
        self.display = tk.Entry(
            left,
            textvariable=self.display_var,
            font=("Segoe UI", 24),
            bd=4,
            relief="sunken",
            bg="white",
            fg="black",
            justify="right",
            state="readonly"
        )
        self.display.pack(fill="x", padx=20, pady=4, ipady=10)

        self.mode_label = tk.Label(left, text="Mode: DEG", bg=BG, fg=ACCENT, font=FONT_MED)
        self.mode_label.pack(pady=2)

        btns = [
            ["7", "8", "9", "/", "C"],
            ["4", "5", "6", "*", "("],
            ["1", "2", "3", "-", ")"],
            ["0", ".", "%", "+", "="],

            ["sin", "cos", "tan", "log", "ln"],
            ["asin", "acos", "atan", "√", "x²"],
            ["sinh", "cosh", "tanh", "x³", "^"],
            ["π", "e", "RAD/DEG", "//", "DEL"],
        ]

        frame = tk.Frame(left, bg=BG)
        frame.pack(pady=4)

        for row in btns:
            row_frame = tk.Frame(frame, bg=BG)
            row_frame.pack()
            for btn in row:
                tk.Button(
                    row_frame, text=btn, width=8, height=2,
                    bg=BTN, fg=FG, font=FONT_MED,
                    command=lambda b=btn: self.on_press(b)
                ).pack(side="left", padx=5, pady=3)

        self.display.focus_set()
        self.display.bind("<KeyPress>", self.handle_key)
        self.display.bind("<<Paste>>", self.handle_paste)

    def load_from_history(self, item):
        expr = str(item.get("expr", ""))
        if expr:
            self.expression = expr
            self.update_display()

    def update_display(self):
        self.display_var.set(self.expression)

    def _env(self):
        if self.angle_mode == "DEG":
            sin = lambda x: math.sin(math.radians(x))
            cos = lambda x: math.cos(math.radians(x))
            tan = lambda x: math.tan(math.radians(x))
            asin = lambda x: math.degrees(math.asin(x))
            acos = lambda x: math.degrees(math.acos(x))
            atan = lambda x: math.degrees(math.atan(x))
        else:
            sin, cos, tan = math.sin, math.cos, math.tan
            asin, acos, atan = math.asin, math.acos, math.atan

        return {
            "pi": math.pi,
            "e": math.e,
            "sin": sin, "cos": cos, "tan": tan,
            "asin": asin, "acos": acos, "atan": atan,
            "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
            "log": lambda x: math.log10(x),
            "ln": lambda x: math.log(x),
            "sqrt": math.sqrt,
            "abs": abs,
            "round": round,
        }

    def _needs_mul_before_const(self):
        if not self.expression:
            return False
        last = self.expression[-1]
        return last.isdigit() or last == "." or last == ")" or last == "e"

    def _append_token(self, token: str):
        if token in ("pi", "e", "(") and self._needs_mul_before_const():
            self.expression += "*"
        self.expression += token
        self.update_display()

    def _get_current_value(self):
        if not self.expression.strip():
            return 0.0
        return float(eval(self.expression, {"__builtins__": {}}, self._env()))

    def _set_value(self, value):
        self.expression = str(value)
        self.update_display()

    def _apply_unary(self, name, func):
        expr_before = self.expression or "0"
        try:
            x = self._get_current_value()
            res = func(x)
            self._set_value(res)
            add_to_history(f"Sci:{name}", expr_before, res)
            self.history.refresh()
        except Exception:
            self._set_value("Error")

    def _safe_eval(self, expr: str):
        """Safely evaluate a scientific expression using a restricted env."""
        cleaned = expr.strip()
        if not cleaned:
            return 0
        cleaned = cleaned.replace("^", "**")
        if re.search(r"[^0-9\+\-\*\/\(\)\.\%pie\s]", cleaned):
            raise ValueError("Invalid characters")
        return eval(cleaned, {"__builtins__": {}}, self._env())

    def calculate(self):
        expr = self.expression
        try:
            result = self._safe_eval(expr)
            self.expression = str(result)

            add_to_history("Sci", expr, result)
            self.history.refresh()
            self.just_evaluated = True

            self.just_evaluated = True
        except Exception:
            self.expression = "Error"
            self.just_evaluated = True

        self.update_display()

    def on_press(self, value):
        if value == "C":
            self.expression = ""
            self.update_display()
            return
        if value == "DEL":
            self.expression = self.expression[:-1]
            self.update_display()
            return
        if value == "=":
            self.calculate()
            return
        if value == "RAD/DEG":
            self.angle_mode = "RAD" if self.angle_mode == "DEG" else "DEG"
            self.mode_label.config(text=f"Mode: {self.angle_mode}")
            return

        if value == "π":
            self._append_token("pi")
            return
        if value == "e":
            self._append_token("e")
            return

        env = self._env()

        if value in ("sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh"):
            self._apply_unary(value, env[value])
            return
        if value == "log":
            self._apply_unary("log", env["log"])
            return
        if value == "ln":
            self._apply_unary("ln", env["ln"])
            return
        if value == "√":
            self._apply_unary("sqrt", env["sqrt"])
            return
        if value == "x²":
            self._apply_unary("square", lambda x: x**2)
            return
        if value == "x³":
            self._apply_unary("cube", lambda x: x**3)
            return

        if value == "^":
            self.expression += "**"
            self.update_display()
            return

        if value == "(":
            self._append_token("(")
        else:
            if self.just_evaluated and (str(value).isdigit() or value == "."):
                self.expression = ""
            self.just_evaluated = False
            self.expression += value
            self.update_display()

    def handle_paste(self, _event):
        try:
            s = self.parent.clipboard_get()
        except Exception:
            return "break"
        allowed = "0123456789+-*/().% "
        cleaned = "".join(ch for ch in s if ch in allowed).replace(" ", "")
        if cleaned:
            self.expression += cleaned
            self.update_display()
        return "break"

    def handle_key(self, event):
        key = event.char
        if key in "0123456789+-*/().%":
            if self.just_evaluated and (key.isdigit() or key == "."):
                self.expression = ""
            self.just_evaluated = False
            self.expression += key
            self.update_display()
            return "break"
        if key == "^":
            self.expression += "**"
            self.update_display()
            return "break"
        if event.keysym in ("Return", "KP_Enter"):
            self.calculate()
            return "break"
        if event.keysym == "BackSpace":
            self.expression = self.expression[:-1]
            self.update_display()
            return "break"
        if event.keysym == "Escape":
            self.expression = ""
            self.update_display()
            return "break"
        return "break"


class MatrixTab:
    def __init__(self, parent):
        self.parent = parent
        tk.Label(parent, text="Matrix Tools", bg=BG, fg=FG, font=FONT_TITLE).pack(pady=10)

        self.frame_inputs = tk.Frame(parent, bg=BG)
        self.frame_inputs.pack(pady=8)

        tk.Label(self.frame_inputs, text="Rows (A):", bg=BG, fg=FG).grid(row=0, column=0)
        tk.Label(self.frame_inputs, text="Cols (A):", bg=BG, fg=FG).grid(row=0, column=1)
        self.a_rows = tk.Entry(self.frame_inputs, width=5)
        self.a_cols = tk.Entry(self.frame_inputs, width=5)
        self.a_rows.grid(row=1, column=0)
        self.a_cols.grid(row=1, column=1)
        tk.Button(self.frame_inputs, text="Create A", bg=BTN, fg=FG, command=self.create_matrix_a)\
            .grid(row=1, column=2, padx=10)

        tk.Label(self.frame_inputs, text="Rows (B):", bg=BG, fg=FG).grid(row=0, column=3)
        tk.Label(self.frame_inputs, text="Cols (B):", bg=BG, fg=FG).grid(row=0, column=4)
        self.b_rows = tk.Entry(self.frame_inputs, width=5)
        self.b_cols = tk.Entry(self.frame_inputs, width=5)
        self.b_rows.grid(row=1, column=3)
        self.b_cols.grid(row=1, column=4)
        tk.Button(self.frame_inputs, text="Create B", bg=BTN, fg=FG, command=self.create_matrix_b)\
            .grid(row=1, column=5, padx=10)

        self.matrix_a_frame = tk.LabelFrame(parent, text="Matrix A", bg=BG, fg=FG)
        self.matrix_a_frame.pack(pady=5)

        self.matrix_b_frame = tk.LabelFrame(parent, text="Matrix B", bg=BG, fg=FG)
        self.matrix_b_frame.pack(pady=5)

        self.matrix_a_entries = []
        self.matrix_b_entries = []

        ops_frame = tk.Frame(parent, bg=BG)
        ops_frame.pack(pady=10)

        operations_a = [
            ("A + B", self.matrix_add),
            ("A - B", self.matrix_subtract),
            ("A × B", self.matrix_multiply),
            ("Transpose A", self.transpose_a),
            ("Det(A)", self.det_a),
            ("Inverse A", self.inverse_a),
            ("Rank A", self.rank_a),
            ("Eigen(A)", self.eigen_a),
            ("Solve Ax=b", self.solve_ax_b),
        ]
        for label, func in operations_a:
            tk.Button(ops_frame, text=label, width=14, bg=BTN, fg=FG, command=func)\
                .pack(side="left", padx=4, pady=4)

        ops_frame_b = tk.Frame(parent, bg=BG)
        ops_frame_b.pack(pady=6)

        operations_b = [
            ("Transpose B", self.transpose_b),
            ("Det(B)", self.det_b),
            ("Inverse B", self.inverse_b),
            ("Rank B", self.rank_b),
            ("Eigen(B)", self.eigen_b),
            ("Reset", self.reset_matrices),
        ]
        for label, func in operations_b:
            tk.Button(ops_frame_b, text=label, width=14, bg=BTN, fg=FG, command=func)\
                .pack(side="left", padx=4, pady=4)

        self.output_frame = tk.Frame(parent, bg=BG)
        self.output_frame.pack(pady=10, fill='x')

        self.output = tk.Text(
            self.output_frame,
            height=12,
            bg="#000000",
            fg=FG,
            insertbackground=FG,
            font=("Consolas", 14),
            wrap="none",
            padx=10,
            pady=10
        )
        self.output.pack(side='left', fill='both', expand=True)
        yscroll = tk.Scrollbar(self.output_frame, orient='vertical', command=self.output.yview)
        xscroll = tk.Scrollbar(self.output_frame, orient='horizontal', command=self.output.xview)
        self.output.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        yscroll.pack(side='right', fill='y')
        xscroll.pack(side='bottom', fill='x')

        tk.Label(
            parent,
            text="Tip: For Solve Ax=b, set Matrix B as a column vector (Cols(B)=1).",
            bg=BG, fg="#cfcfcf", font=FONT_SMALL
        ).pack(pady=(0, 8))

    def reset_matrices(self):
        """Clear matrices A/B, their entry widgets, size boxes, and output."""
        for e in (self.a_rows, self.a_cols, self.b_rows, self.b_cols):
            try:
                e.delete(0, "end")
            except Exception:
                pass

        for frame in (self.matrix_a_frame, self.matrix_b_frame):
            try:
                for w in frame.winfo_children():
                    w.destroy()
            except Exception:
                pass

        self.matrix_a_entries = []
        self.matrix_b_entries = []

        try:
            self.output.configure(state="normal")
            self.output.delete("1.0", "end")
            self.output.configure(state="disabled")
        except Exception:
            try:
                self.output.delete("1.0", "end")
            except Exception:
                pass

    def show_output(self, obj):
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")

        try:
            arr = np.array(obj, dtype=float)
            if arr.ndim == 2:
                lines = []
                for row in arr:
                    lines.append("  ".join([f"{v: .6g}" for v in row]))
                text = "\n".join(lines)
                self.output.insert("end", text)
                self.output.configure(state="disabled")
                return
            if arr.ndim == 1:
                text = "\n".join([f"{v: .6g}" for v in arr])
                self.output.insert("end", text)
                self.output.configure(state="disabled")
                return
        except Exception:
            pass

        self.output.insert("end", str(obj))
        self.output.configure(state="disabled")


    def _validate_created(self, entries, name):
        if not entries:
            messagebox.showerror("Error", f"Please create {name} first.")
            return False
        return True

    def _validate_square(self, M, name):
        if M.ndim != 2:
            raise ValueError(f"{name} must be a 2D matrix")
        if M.shape[0] != M.shape[1]:
            raise ValueError(f"{name} must be square (rows = cols)")
        return True

    def get_matrix(self, entries):
        if not entries:
            return None
        try:
            rows = []
            for row_entries in entries:
                row = []
                for e in row_entries:
                    s = e.get().strip()
                    if s == "":
                        s = "0"
                    row.append(float(s))
                rows.append(row)
            return np.array(rows, dtype=float)
        except Exception:
            messagebox.showerror("Error", "Invalid matrix input (use numbers only).")
            return None

    def create_matrix_a(self):
        try:
            r = int(self.a_rows.get()); c = int(self.a_cols.get())
            if r <= 0 or c <= 0:
                raise ValueError
        except Exception:
            return messagebox.showerror("Error", "Enter valid positive integers for Rows/Cols of A")

        for w in self.matrix_a_frame.winfo_children():
            w.destroy()
        self.matrix_a_entries = []
        for i in range(r):
            row_entries = []
            for j in range(c):
                e = tk.Entry(self.matrix_a_frame, width=8)
                e.grid(row=i, column=j, padx=2, pady=2)
                row_entries.append(e)
            self.matrix_a_entries.append(row_entries)

    def create_matrix_b(self):
        try:
            r = int(self.b_rows.get()); c = int(self.b_cols.get())
            if r <= 0 or c <= 0:
                raise ValueError
        except Exception:
            return messagebox.showerror("Error", "Enter valid positive integers for Rows/Cols of B")

        for w in self.matrix_b_frame.winfo_children():
            w.destroy()
        self.matrix_b_entries = []
        for i in range(r):
            row_entries = []
            for j in range(c):
                e = tk.Entry(self.matrix_b_frame, width=8)
                e.grid(row=i, column=j, padx=2, pady=2)
                row_entries.append(e)
            self.matrix_b_entries.append(row_entries)

    def matrix_add(self):
        A = self.get_matrix(self.matrix_a_entries)
        B = self.get_matrix(self.matrix_b_entries)
        if A is None or B is None:
            return
        if A.shape != B.shape:
            return messagebox.showerror("Shape Error", f"A and B must have same shape. A={A.shape}, B={B.shape}")
        self.show_output(A + B)

    def matrix_subtract(self):
        A = self.get_matrix(self.matrix_a_entries)
        B = self.get_matrix(self.matrix_b_entries)
        if A is None or B is None:
            return
        if A.shape != B.shape:
            return messagebox.showerror("Shape Error", f"A and B must have same shape. A={A.shape}, B={B.shape}")
        self.show_output(A - B)

    def matrix_multiply(self):
        A = self.get_matrix(self.matrix_a_entries)
        B = self.get_matrix(self.matrix_b_entries)
        if A is None or B is None:
            return
        if A.shape[1] != B.shape[0]:
            return messagebox.showerror("Shape Error", f"Invalid multiply: A cols must equal B rows. A={A.shape}, B={B.shape}")
        self.show_output(A @ B)

    def transpose_a(self):
        A = self.get_matrix(self.matrix_a_entries)
        if A is None:
            return
        self.show_output(A.T)

    def transpose_b(self):
        B = self.get_matrix(self.matrix_b_entries)
        if B is None:
            return
        self.show_output(B.T)

    def det_a(self):
        A = self.get_matrix(self.matrix_a_entries)
        if A is None:
            return
        try:
            self._validate_square(A, "A")
            self.show_output(np.linalg.det(A))
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def det_b(self):
        B = self.get_matrix(self.matrix_b_entries)
        if B is None:
            return
        try:
            self._validate_square(B, "B")
            self.show_output(np.linalg.det(B))
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def inverse_a(self):
        A = self.get_matrix(self.matrix_a_entries)
        if A is None:
            return
        try:
            self._validate_square(A, "A")
            self.show_output(np.linalg.inv(A))
        except Exception as e:
            messagebox.showerror("Error", f"Cannot invert A: {e}")

    def inverse_b(self):
        B = self.get_matrix(self.matrix_b_entries)
        if B is None:
            return
        try:
            self._validate_square(B, "B")
            self.show_output(np.linalg.inv(B))
        except Exception as e:
            messagebox.showerror("Error", f"Cannot invert B: {e}")

    def rank_a(self):
        A = self.get_matrix(self.matrix_a_entries)
        if A is None:
            return
        try:
            if A.ndim != 2:
                raise ValueError("A must be 2D")
            self.show_output(np.linalg.matrix_rank(A))
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def rank_b(self):
        B = self.get_matrix(self.matrix_b_entries)
        if B is None:
            return
        try:
            if B.ndim != 2:
                raise ValueError("B must be 2D")
            self.show_output(np.linalg.matrix_rank(B))
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def eigen_a(self):
        A = self.get_matrix(self.matrix_a_entries)
        if A is None:
            return
        try:
            self._validate_square(A, "A")
            vals, vecs = np.linalg.eig(A)
            self.show_output("Eigenvalues:\n" + str(vals) + "\n\nEigenvectors:\n" + str(vecs))
        except Exception as e:
            messagebox.showerror("Error", f"Eigen(A) failed: {e}")

    def eigen_b(self):
        B = self.get_matrix(self.matrix_b_entries)
        if B is None:
            return
        try:
            self._validate_square(B, "B")
            vals, vecs = np.linalg.eig(B)
            self.show_output("Eigenvalues:\n" + str(vals) + "\n\nEigenvectors:\n" + str(vecs))
        except Exception as e:
            messagebox.showerror("Error", f"Eigen(B) failed: {e}")

    def solve_ax_b(self):
        A = self.get_matrix(self.matrix_a_entries)
        B = self.get_matrix(self.matrix_b_entries)
        if A is None or B is None:
            return
        try:
            self._validate_square(A, "A")
            if B.ndim != 2:
                raise ValueError("B must be 2D (use Cols(B)=1 for a vector).")
            if B.shape[0] != A.shape[0]:
                raise ValueError(f"B rows must equal A rows. A={A.shape}, B={B.shape}")

            x = np.linalg.solve(A, B)
            self.show_output(x)
        except Exception as e:
            messagebox.showerror("Error", f"Cannot solve Ax=b: {e}")

class DataTab:
    def __init__(self, parent):
        self.parent = parent
        self.df = None

        tk.Label(parent, text="Data Tools", bg=BG, fg=FG, font=FONT_TITLE).pack(pady=10)

        file_frame = tk.Frame(parent, bg=BG)
        file_frame.pack(pady=5)
        tk.Button(file_frame, text="Load CSV", bg=BTN, fg=FG, command=self.load_csv).pack(side="left", padx=8)
        tk.Button(file_frame, text="Load Excel", bg=BTN, fg=FG, command=self.load_excel).pack(side="left", padx=8)
        tk.Button(file_frame, text="Save Cleaned CSV", bg=BTN, fg=FG, command=self.save_csv).pack(side="left", padx=8)

        self.text = tk.Text(parent, width=140, height=18, bg="#000000", fg=FG)
        self.text.pack(pady=8)

        col_frame = tk.Frame(parent, bg=BG)
        col_frame.pack(pady=5)
        tk.Label(col_frame, text="Column:", bg=BG, fg=FG).pack(side="left")
        self.column_var = tk.StringVar()
        self.column_menu = ttk.Combobox(col_frame, textvariable=self.column_var, width=30)
        self.column_menu.pack(side="left", padx=10)

        ops_frame = tk.Frame(parent, bg=BG)
        ops_frame.pack(pady=8)
        tk.Button(ops_frame, text="Describe", bg=BTN, fg=FG, command=self.describe_column).pack(side="left", padx=5)
        tk.Button(ops_frame, text="Drop NA Rows", bg=BTN, fg=FG, command=self.drop_na_rows).pack(side="left", padx=5)
        tk.Button(ops_frame, text="Fill NA (Mean)", bg=BTN, fg=FG, command=self.fill_mean).pack(side="left", padx=5)
        tk.Button(ops_frame, text="Unique Values", bg=BTN, fg=FG, command=self.unique_values).pack(side="left", padx=5)
        tk.Button(ops_frame, text="Value Counts", bg=BTN, fg=FG, command=self.value_counts).pack(side="left", padx=5)

        filter_frame = tk.Frame(parent, bg=BG)
        filter_frame.pack(pady=8)
        tk.Label(filter_frame, text="Filter (pandas query):", bg=BG, fg=FG).pack(side="left")
        self.filter_entry = tk.Entry(filter_frame, width=50)
        self.filter_entry.pack(side="left", padx=10)
        tk.Button(filter_frame, text="Apply Filter", bg=BTN, fg=FG, command=self.apply_filter).pack(side="left")

    def load_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not path: return
        try:
            self.df = pd.read_csv(path)
            self.update_preview(self.df)
            self.update_columns()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def load_excel(self):
        path = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xlsx")])
        if not path: return
        try:
            self.df = pd.read_excel(path)
            self.update_preview(self.df)
            self.update_columns()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def save_csv(self):
        if self.df is None: return
        path = filedialog.asksaveasfilename(defaultextension=".csv")
        if not path: return
        try:
            self.df.to_csv(path, index=False)
            messagebox.showinfo("Saved", "Dataset saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def update_columns(self):
        if self.df is None: return
        self.column_menu["values"] = list(self.df.columns)

    def update_preview(self, df_or_series):
        self.text.delete("1.0", tk.END)
        try:
            if hasattr(df_or_series, "to_string"):
                self.text.insert(tk.END, df_or_series.head().to_string() if isinstance(df_or_series, pd.DataFrame) else df_or_series.to_string())
            else:
                self.text.insert(tk.END, str(df_or_series))
        except Exception:
            self.text.insert(tk.END, str(df_or_series))

    def describe_column(self):
        if self.df is None: return
        col = self.column_var.get()
        if not col: return
        self.update_preview(self.df[col].describe(include="all"))

    def drop_na_rows(self):
        if self.df is None: return
        col = self.column_var.get()
        if not col: return
        self.df = self.df.dropna(subset=[col])
        self.update_preview(self.df)

    def fill_mean(self):
        if self.df is None: return
        col = self.column_var.get()
        if not col: return
        try:
            mean_val = pd.to_numeric(self.df[col], errors="coerce").mean()
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(mean_val)
            self.update_preview(self.df)
        except Exception:
            messagebox.showerror("Error", "Column must be numeric")

    def unique_values(self):
        if self.df is None: return
        col = self.column_var.get()
        if not col: return
        result = self.df[col].dropna().unique()
        self.update_preview(result)

    def value_counts(self):
        if self.df is None: return
        col = self.column_var.get()
        if not col: return
        result = self.df[col].value_counts(dropna=False)
        self.update_preview(result)

    def apply_filter(self):
        if self.df is None: return
        query_str = self.filter_entry.get().strip()
        if not query_str:
            self.update_preview(self.df)
            return
        try:
            filtered = self.df.query(query_str)
            self.update_preview(filtered)
        except Exception:
            messagebox.showerror("Error", "Invalid filter expression")

class StatisticsTab:
    def __init__(self, parent):
        self.parent = parent
        self.df = None

        tk.Label(parent, text="Statistics Tools", bg=BG, fg=FG, font=FONT_TITLE).pack(pady=10)

        file_frame = tk.Frame(parent, bg=BG)
        file_frame.pack(pady=5)
        tk.Button(file_frame, text="Load CSV", bg=BTN, fg=FG, command=self.load_csv).pack(side="left", padx=8)
        tk.Button(file_frame, text="Load Excel", bg=BTN, fg=FG, command=self.load_excel).pack(side="left", padx=8)

        self.text = tk.Text(parent, width=140, height=16, bg="#000000", fg=FG)
        self.text.pack(pady=8)

        col_frame = tk.Frame(parent, bg=BG)
        col_frame.pack(pady=5)

        tk.Label(col_frame, text="Column 1:", bg=BG, fg=FG).pack(side="left")
        self.col1 = ttk.Combobox(col_frame, width=30)
        self.col1.pack(side="left", padx=10)

        tk.Label(col_frame, text="Column 2:", bg=BG, fg=FG).pack(side="left")
        self.col2 = ttk.Combobox(col_frame, width=30)
        self.col2.pack(side="left", padx=10)

        stats_frame = tk.Frame(parent, bg=BG)
        stats_frame.pack(pady=5)

        buttons = [
            ("Describe", self.describe_df),
            ("Mean", self.mean_col),
            ("Median", self.median_col),
            ("Mode", self.mode_col),
            ("Variance", self.variance_col),
            ("Std Dev", self.std_col),
            ("Skewness", self.skew_col),
            ("Kurtosis", self.kurt_col),
            ("Correlation Matrix", self.corr_matrix),
            ("Covariance Matrix", self.cov_matrix),
        ]
        for label, func in buttons:
            tk.Button(stats_frame, text=label, bg=BTN, fg=FG, command=func).pack(side="left", padx=4, pady=4)

        test_frame = tk.Frame(parent, bg=BG)
        test_frame.pack(pady=8)
        tk.Label(test_frame, text="Hypothesis Testing:", bg=BG, fg=ACCENT, font=FONT_MED).pack(side="left", padx=10)
        tk.Button(test_frame, text="T-Test", bg=BTN, fg=FG, command=self.t_test).pack(side="left", padx=5)
        tk.Button(test_frame, text="Chi-Square", bg=BTN, fg=FG, command=self.chi_square).pack(side="left", padx=5)
        tk.Button(test_frame, text="ANOVA", bg=BTN, fg=FG, command=self.anova_test).pack(side="left", padx=5)

    def load_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not path: return
        self.df = pd.read_csv(path)
        self.update_preview()
        self.update_columns()

    def load_excel(self):
        path = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xlsx")])
        if not path: return
        self.df = pd.read_excel(path)
        self.update_preview()
        self.update_columns()

    def update_preview(self):
        if self.df is None: return
        self.text.delete("1.0", tk.END)
        self.text.insert(tk.END, self.df.head().to_string())

    def update_columns(self):
        if self.df is None: return
        cols = list(self.df.columns)
        self.col1["values"] = cols
        self.col2["values"] = cols

    def describe_df(self):
        if self.df is None: return
        self.text.delete("1.0", tk.END)
        self.text.insert(tk.END, str(self.df.describe(include="all")))

    def _get_series(self, colname):
        if self.df is None:
            messagebox.showerror("Error", "No dataset loaded.")
            return None
        if not colname:
            messagebox.showerror("Error", "Select a column.")
            return None
        return pd.to_numeric(self.df[colname], errors="coerce")

    def mean_col(self):
        c = self._get_series(self.col1.get())
        if c is not None:
            self._show(f"Mean = {c.mean()}")

    def median_col(self):
        c = self._get_series(self.col1.get())
        if c is not None:
            self._show(f"Median = {c.median()}")

    def mode_col(self):
        c = self._get_series(self.col1.get())
        if c is not None:
            m = c.mode(dropna=True)
            self._show(f"Mode = {m.iloc[0] if len(m) else 'No unique mode'}")

    def variance_col(self):
        c = self._get_series(self.col1.get())
        if c is not None:
            self._show(f"Variance = {c.var()}")

    def std_col(self):
        c = self._get_series(self.col1.get())
        if c is not None:
            self._show(f"Std Dev = {c.std()}")

    def skew_col(self):
        c = self._get_series(self.col1.get())
        if c is not None:
            self._show(f"Skewness = {c.skew()}")

    def kurt_col(self):
        c = self._get_series(self.col1.get())
        if c is not None:
            self._show(f"Kurtosis = {c.kurt()}")

    def corr_matrix(self):
        if self.df is None: return
        self._show(self.df.corr(numeric_only=True))

    def cov_matrix(self):
        if self.df is None: return
        self._show(self.df.cov(numeric_only=True))

    def t_test(self):
        if self.df is None: return
        c1, c2 = self.col1.get(), self.col2.get()
        if not c1 or not c2: return
        try:
            a = pd.to_numeric(self.df[c1], errors="coerce")
            b = pd.to_numeric(self.df[c2], errors="coerce")
            res = stats.ttest_ind(a, b, nan_policy="omit")
            self._show(f"T-Test:\nStatistic = {res.statistic}\nP-value = {res.pvalue}")
        except Exception:
            messagebox.showerror("Error", "T-test failed.")

    def chi_square(self):
        if self.df is None: return
        c1, c2 = self.col1.get(), self.col2.get()
        if not c1 or not c2: return
        try:
            table = pd.crosstab(self.df[c1], self.df[c2])
            chi2, p, dof, _exp = stats.chi2_contingency(table)
            self._show(f"Chi-Square Test:\nChi2 = {chi2}\nP-value = {p}\nDoF = {dof}\n\nTable:\n{table}")
        except Exception:
            messagebox.showerror("Error", "Chi-Square test failed.")

    def anova_test(self):
        if self.df is None: return
        group_col, value_col = self.col1.get(), self.col2.get()
        if not group_col or not value_col: return
        try:
            groups = [pd.to_numeric(g[value_col], errors="coerce").dropna() for _, g in self.df.groupby(group_col)]
            res = stats.f_oneway(*groups)
            self._show(f"ANOVA:\nF-statistic = {res.statistic}\nP-value = {res.pvalue}")
        except Exception:
            messagebox.showerror("Error", "ANOVA test failed.")

    def _show(self, obj):
        self.text.delete("1.0", tk.END)
        self.text.insert(tk.END, str(obj))


class MLTab:
    def __init__(self, parent):
        self.parent = parent
        self.df = None
        self.model = None
        self._last = {}
        self._dummy_columns = None  
        self._x_cols = []           
        self._y_col = None

        tk.Label(parent, text="Machine Learning Tools", bg=BG, fg=FG, font=FONT_TITLE).pack(pady=10)

        main = tk.Frame(parent, bg=BG)
        main.pack(fill="both", expand=True, padx=16, pady=10)

        main.grid_rowconfigure(0, weight=3)
        main.grid_rowconfigure(1, weight=2)
        main.grid_columnconfigure(0, weight=1)
        main.grid_columnconfigure(1, weight=2)

        panel_bg = "#232323"
        border_fg = "#444444"

        step_panel = tk.LabelFrame(
            main, text="Step", bg=panel_bg, fg=FG,
            font=FONT_MED, bd=2, relief="groove", labelanchor="nw"
        )
        step_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=(0, 10))
        step_panel.grid_rowconfigure(2, weight=1)
        step_panel.grid_columnconfigure(0, weight=1)

        help_text = """1) Load CSV/Excel
2) Select Target column (Y)
3) Select Feature columns (X) (Ctrl/Shift for multi-select)
4) Choose Algorithm
5) Click Train Model
Tip: For prediction, enter values in the same order as selected Features."""
        tk.Label(step_panel, text=help_text, bg=panel_bg, fg="#cfcfcf",
                 justify="left", font=FONT_SMALL).grid(row=0, column=0, sticky="nw", padx=10, pady=(8, 6))

        btns = tk.Frame(step_panel, bg=panel_bg)
        btns.grid(row=1, column=0, sticky="w", padx=10, pady=(0, 8))
        tk.Button(btns, text="Load CSV", bg=BTN, fg=FG, command=self.load_csv).pack(side="left", padx=(0, 8))
        tk.Button(btns, text="Load Excel", bg=BTN, fg=FG, command=self.load_excel).pack(side="left")

        data_panel = tk.LabelFrame(
            main, text="File data", bg=panel_bg, fg=FG,
            font=FONT_MED, bd=2, relief="groove", labelanchor="n"
        )
        data_panel.grid(row=0, column=1, sticky="nsew", pady=(0, 10))
        data_panel.grid_rowconfigure(0, weight=1)
        data_panel.grid_columnconfigure(0, weight=1)

        self.preview_frame, self.preview = make_scrolled_text(data_panel, height=10, width=110)
        self.preview_frame.configure(bg=panel_bg)
        self.preview.configure(
            bg="#000000",
            fg=FG,
            insertbackground=FG,
            font=("Consolas", 11),   
            wrap="none"
        )

        input_panel = tk.LabelFrame(
            main, text="inputs and selection", bg=panel_bg, fg=FG,
            font=FONT_MED, bd=2, relief="groove", labelanchor="n"
        )
        input_panel.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        input_panel.grid_columnconfigure(1, weight=1)

        tk.Label(input_panel, text="Target (Y):", bg=panel_bg, fg=FG).grid(row=0, column=0, sticky="w", padx=10, pady=(10, 4))
        self.target_var = tk.StringVar(value="")
        self.target_menu = ttk.Combobox(input_panel, textvariable=self.target_var, state="readonly", width=28, values=[])
        self.target_menu.grid(row=0, column=1, sticky="w", padx=10, pady=(10, 4))

        tk.Label(input_panel, text="Features (X):", bg=panel_bg, fg=FG).grid(row=1, column=0, sticky="nw", padx=10, pady=(6, 4))
        self.feature_list = tk.Listbox(input_panel, selectmode="extended", height=8, width=34)
        self.feature_list.grid(row=1, column=1, sticky="w", padx=10, pady=(6, 4))

        tk.Label(input_panel, text="Algorithm:", bg=panel_bg, fg=FG).grid(row=2, column=0, sticky="w", padx=10, pady=(10, 4))
        self.alg_var = tk.StringVar(value="Logistic Regression")
        ttk.Combobox(
            input_panel,
            textvariable=self.alg_var,
            state="readonly",
            values=["Linear Regression", "Logistic Regression"],
            width=26
        ).grid(row=2, column=1, sticky="w", padx=10, pady=(10, 4))

        action_row1 = tk.Frame(input_panel, bg=panel_bg)
        action_row1.grid(row=3, column=0, columnspan=2, sticky="w", padx=10, pady=(4, 4))

        tk.Button(action_row1, text="Train Model", width=14,
                bg=ACCENT, fg="black", command=self.train_model)\
            .pack(side="left", padx=(0, 8))

        tk.Button(action_row1, text="Reset", width=10,
                bg=BTN, fg=FG, command=self.reset_ml)\
            .pack(side="left", padx=(0, 8))

        tk.Button(action_row1, text="Feature Importance", width=20,
                bg=BTN, fg=FG, command=self.show_feature_importance)\
            .pack(side="left", padx=(0, 8))


        action_row2 = tk.Frame(input_panel, bg=panel_bg)
        action_row2.grid(row=4, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 6))

        # Use grid so buttons don't get clipped on small widths
        btn_cm = tk.Button(action_row2, text="Confusion Matrix Heatmap", width=24,
                           bg=BTN, fg=FG, command=self.plot_confusion_heatmap)
        btn_cm.grid(row=0, column=0, sticky="w", padx=(0, 8), pady=(0, 4))

        btn_export = tk.Button(action_row2, text="Export ML Report", width=18,
                               bg=BTN, fg=FG, command=self.export_report)
        btn_export.grid(row=0, column=1, sticky="w", padx=(0, 8), pady=(0, 4))

        pred_box = tk.LabelFrame(input_panel, text="Prediction", bg=panel_bg, fg=FG, bd=1, relief="groove")
        pred_box.grid(row=5, column=0, columnspan=2, sticky="ew", padx=10, pady=(8, 8))
        pred_box.grid_columnconfigure(1, weight=1)

        tk.Label(pred_box, text="Values (comma-separated):", bg=panel_bg, fg=FG).grid(row=0, column=0, sticky="w", padx=10, pady=10)
        self.predict_entry = tk.Entry(pred_box)
        self.predict_entry.grid(row=0, column=1, sticky="ew", padx=10, pady=10)
        tk.Button(pred_box, text="Predict", bg=BTN, fg=FG, command=self.predict_value).grid(row=0, column=2, sticky="e", padx=10, pady=10)

        out_panel = tk.LabelFrame(
            main, text="result of regression", bg=panel_bg, fg=FG,
            font=FONT_MED, bd=2, relief="groove", labelanchor="n"
        )
        out_panel.grid(row=1, column=1, sticky="nsew")
        out_panel.grid_rowconfigure(0, weight=1)
        out_panel.grid_columnconfigure(0, weight=1)

        self.metrics_frame, self.metrics = make_scrolled_text(out_panel, height=10, width=110)
        self.metrics_frame.configure(bg=panel_bg)
        self.metrics.configure(
            bg="#000000",
            fg=FG,
            insertbackground=FG,
            font=("Consolas", 13),
            wrap="word"
        )

        self._init_metric_tags()
        self._write_hint()

    def reset_ml(self):
        self.df = None
        self.model = None
        self._last = {}
        self._dummy_columns = None
        self._x_cols = []
        self._y_col = None

        try:
            self.preview.delete("1.0", tk.END)
        except Exception:
            pass
        try:
            self.metrics.configure(state="normal")
            self.metrics.delete("1.0", tk.END)
            self.metrics.configure(state="normal")
        except Exception:
            pass
        try:
            self.target_menu["values"] = []
            self.target_var.set("")
            self.feature_list.delete(0, tk.END)
        except Exception:
            pass
        try:
            self.predict_entry.delete(0, tk.END)
        except Exception:
            pass

        self._write_hint()

    def _write_hint(self):
        self._insert_line("READY", "header")
        self._insert_line("-" * 50, "neutral")
        self._insert_line("1) Load CSV/Excel", "neutral")
        self._insert_line("2) Select Target (Y)", "neutral")
        self._insert_line("3) Select Feature columns (X) (Ctrl/Shift for multi-select)", "neutral")
        self._insert_line("4) Choose Algorithm", "neutral")
        self._insert_line("5) Click Train Model", "neutral")
        self._insert_line("", "neutral")
        self._insert_line("Tip: For prediction, enter values in the SAME order as selected Features.", "title")

    def _init_metric_tags(self):
        try:
            self.metrics.tag_configure("good", foreground="#00ff66")
            self.metrics.tag_configure("bad", foreground="#ff4444")
            self.metrics.tag_configure("neutral", foreground=FG)
            self.metrics.tag_configure("title", foreground="#66ccff")
            self.metrics.tag_configure("header", foreground="#ffffff", font=("Consolas", 14, "bold"))
        except Exception:
            pass

    def _insert_line(self, text, tag="neutral"):
        self.metrics.insert(tk.END, text + "\n", tag)

    def _metric_tag(self, name, val):
        lname = name.lower()
        try:
            v = float(val)
        except Exception:
            return "neutral"

        if any(k in lname for k in ["accuracy", "precision", "recall", "f1"]):
            if v >= 0.80: return "good"
            if v < 0.50: return "bad"
            return "neutral"
        if "r2" in lname or "r²" in lname:
            if v >= 0.70: return "good"
            if v < 0.0: return "bad"
            return "neutral"
        return "neutral"

    def load_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not path:
            return
        try:
            self.df = pd.read_csv(path)
        except Exception as e:
            return messagebox.showerror("Error", f"Failed to load CSV: {e}")
        self._after_load()

    def load_excel(self):
        path = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xlsx *.xls")])
        if not path:
            return
        try:
            self.df = pd.read_excel(path)
        except Exception as e:
            return messagebox.showerror("Error", f"Failed to load Excel: {e}")
        self._after_load()

    def _after_load(self):
        self.preview.delete("1.0", tk.END)
        self.preview.insert(tk.END, str(self.df.head(25)))

        cols = list(self.df.columns.astype(str))
        self.target_menu["values"] = cols
        if cols:
            self.target_var.set(cols[-1])

        self.feature_list.delete(0, tk.END)
        for c in cols:
            self.feature_list.insert(tk.END, c)

        self.metrics.configure(state="normal")
        self.metrics.delete("1.0", tk.END)
        self._insert_line("Dataset loaded.", "header")
        self._insert_line("Select Target (Y) and Features (X), then click Train Model.", "neutral")
        self._insert_line("", "neutral")

    def _get_selected_features(self):
        idxs = self.feature_list.curselection()
        return [self.feature_list.get(i) for i in idxs]

    def train_model(self):
        if self.df is None:
            return messagebox.showerror("Error", "Please load a dataset first.")

        y_col = self.target_var.get()
        x_cols = self._get_selected_features()

        if not y_col:
            return messagebox.showerror("Error", "Please select Target (Y) column.")
        if not x_cols:
            return messagebox.showerror("Error", "Please select at least one Feature (X) column.")
        if y_col in x_cols:
            x_cols = [c for c in x_cols if c != y_col]
            if not x_cols:
                return messagebox.showerror("Error", "Features (X) cannot be only the target column.")

        data = self.df[x_cols + [y_col]].copy().dropna()
        if len(data) < 5:
            return messagebox.showerror("Error", "Not enough rows after removing missing values.")

        X = data[x_cols]
        y = data[y_col]

        X_enc = pd.get_dummies(X, drop_first=False)
        self._dummy_columns = X_enc.columns
        self._x_cols = x_cols
        self._y_col = y_col

        try:
            X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size=0.2, random_state=42)
        except Exception as e:
            return messagebox.showerror("Error", f"Train/test split failed: {e}")

        alg = self.alg_var.get()
        self.metrics.configure(state="normal")
        self.metrics.delete("1.0", tk.END)

        if alg == "Linear Regression":
            try:
                y_train_num = pd.to_numeric(y_train)
                y_test_num = pd.to_numeric(y_test)
            except Exception:
                return messagebox.showerror("Error", "Linear Regression requires numeric target (Y).")

            self.model = LinearRegression()
            self.model.fit(X_train, y_train_num)
            preds = self.model.predict(X_test)

            r2 = r2_score(y_test_num, preds)
            mse = mean_squared_error(y_test_num, preds)
            rmse = mse ** 0.5
            mae = mean_absolute_error(y_test_num, preds)

            self._insert_line("LINEAR REGRESSION REPORT", "header")
            self._insert_line("-" * 50, "neutral")
            self._insert_line(f"Rows used: {len(data)}", "neutral")
            self._insert_line(f"Features used (after encoding): {len(self._dummy_columns)}", "neutral")
            self._insert_line("", "neutral")

            self._insert_line(f"R² Score : {r2:.4f}", self._metric_tag("r2", r2))
            self._insert_line(f"MSE      : {mse:.4f}", "neutral")
            self._insert_line(f"RMSE     : {rmse:.4f}", "neutral")
            self._insert_line(f"MAE      : {mae:.4f}", "neutral")

            self._last = {"X_test": X_test, "y_test": y_test_num, "preds": preds, "alg": alg}

        else:
            if pd.Series(y).nunique() < 2:
                return messagebox.showerror("Error", "Target must have at least 2 classes for Logistic Regression.")

            self.model = LogisticRegression(max_iter=2000)
            try:
                self.model.fit(X_train, y_train)
            except Exception as e:
                return messagebox.showerror("Error", f"Model training failed: {e}")

            preds = self.model.predict(X_test)

            acc = accuracy_score(y_test, preds)
            avg = "binary" if len(np.unique(y_test)) == 2 else "macro"
            prec = precision_score(y_test, preds, zero_division=0, average=avg)
            rec = recall_score(y_test, preds, zero_division=0, average=avg)
            f1 = f1_score(y_test, preds, zero_division=0, average=avg)
            cm = confusion_matrix(y_test, preds)
            classes = list(np.unique(y_test))

            self._insert_line("LOGISTIC REGRESSION REPORT", "header")
            self._insert_line("-" * 50, "neutral")
            self._insert_line(f"Rows used: {len(data)}", "neutral")
            self._insert_line(f"Classes : {classes}", "neutral")
            self._insert_line("", "neutral")

            self._insert_line(f"Accuracy  : {acc:.4f}", self._metric_tag("accuracy", acc))
            self._insert_line(f"Precision : {prec:.4f}", self._metric_tag("precision", prec))
            self._insert_line(f"Recall    : {rec:.4f}", self._metric_tag("recall", rec))
            self._insert_line(f"F1 Score  : {f1:.4f}", self._metric_tag("f1", f1))
            self._insert_line("", "neutral")

            self._insert_line("Confusion Matrix (rows=Actual, cols=Predicted):", "title")
            self._insert_line("-" * 50, "neutral")
            try:
                w = max(len(str(v)) for v in cm.flatten()) + 2
                for row in cm:
                    self._insert_line("".join(f"{int(v):>{w}d}" for v in row), "neutral")
            except Exception:
                for row in cm:
                    self._insert_line("   ".join(str(v) for v in row), "neutral")

            self._last = {"X_test": X_test, "y_test": y_test, "preds": preds, "alg": alg, "cm": cm, "classes": classes}

    def predict_value(self):
        if self.model is None or self.df is None or self._dummy_columns is None:
            return messagebox.showerror("Error", "Train a model first.")

        raw = self.predict_entry.get().strip()
        if not raw:
            return messagebox.showerror("Error", "Enter feature values (comma-separated).")

        parts = [p.strip() for p in raw.split(",")]
        if len(parts) != len(self._x_cols):
            return messagebox.showerror("Error", f"Expected {len(self._x_cols)} values for features: {self._x_cols}")

        row = {}
        for c, val in zip(self._x_cols, parts):
            try:
                row[c] = float(val)
            except Exception:
                row[c] = val

        X_one = pd.DataFrame([row])
        X_one_enc = pd.get_dummies(X_one, drop_first=False)

        for col in self._dummy_columns:
            if col not in X_one_enc.columns:
                X_one_enc[col] = 0
        X_one_enc = X_one_enc[self._dummy_columns]

        try:
            pred = self.model.predict(X_one_enc)[0]
        except Exception as e:
            return messagebox.showerror("Error", f"Prediction failed: {e}")

        messagebox.showinfo("Prediction", f"Predicted value/class: {pred}")

    def show_feature_importance(self):
        if self.model is None or self._dummy_columns is None:
            return messagebox.showerror("Error", "Train a model first.")

        try:
            if hasattr(self.model, "coef_"):
                coef = np.array(self.model.coef_)
                if coef.ndim == 2:
                    coef = coef[0]
            elif hasattr(self.model, "feature_importances_"):
                coef = np.array(self.model.feature_importances_)
            else:
                return messagebox.showerror("Error", "This model does not provide feature importances.")
        except Exception as e:
            return messagebox.showerror("Error", f"Cannot compute feature importance: {e}")

        cols = list(self._dummy_columns)
        pairs = list(zip(cols, coef))
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)

        top = pairs[:15]
        lines = ["Top feature weights (sorted by |weight|):", ""]
        for name, w in top:
            lines.append(f"{name:30s}  {w: .6g}")

        win = tk.Toplevel(self.parent)
        win.title("Feature Importance")
        f, t = make_scrolled_text(win, height=18, width=110)
        t.configure(wrap="none")
        t.insert(tk.END, "\n".join(lines))
        t.configure(state="disabled")

    def export_report(self):
        if self.df is None:
            return messagebox.showerror("Error", "Load a dataset first.")

        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text file", "*.txt"), ("All files", "*.*")]
        )
        if not path:
            return

        try:
            preview_text = self.preview.get("1.0", tk.END).strip()
            metric_text = self.metrics.get("1.0", tk.END).strip()

            with open(path, "w", encoding="utf-8") as f:
                f.write("=== DATA PREVIEW (head) ===\n")
                f.write(preview_text + "\n\n")
                f.write("=== MODEL REPORT ===\n")
                f.write(metric_text + "\n")
        except Exception as e:
            return messagebox.showerror("Error", f"Failed to export: {e}")

        messagebox.showinfo("Saved", f"Report saved to: {path}")

    def plot_confusion_heatmap(self):
        if self.model is None or self._last.get("alg") != "Logistic Regression":
            return messagebox.showerror("Error", "Train Logistic Regression first.")
        cm = self._last.get("cm")
        classes = self._last.get("classes")
        if cm is None:
            y_test = self._last.get("y_test"); preds = self._last.get("preds")
            if y_test is None or preds is None:
                return
            cm = confusion_matrix(y_test, preds)
            classes = list(np.unique(y_test))

        fig = plt.figure(figsize=(6, 4.5))
        ax = fig.add_subplot(111)
        im = ax.imshow(cm)
        ax.set_title("Confusion Matrix Heatmap")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        win = tk.Toplevel(self.parent)
        win.title("Confusion Matrix Heatmap")
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

class FinanceTab:
    def __init__(self, parent):
        self.parent = parent
        tk.Label(parent, text="Finance & Business Analytics", bg=BG, fg=FG, font=FONT_TITLE).pack(pady=10)

        frame = tk.Frame(parent, bg=BG)
        frame.pack(pady=5)

        labels = [
            ("Principal / Value:", "principal"),
            ("Rate (%):", "rate"),
            ("Time (Years):", "time"),
            ("Cash Flows (comma-separated):", "cashflows"),
        ]
        self.entries = {}
        for r, (txt, key) in enumerate(labels):
            tk.Label(frame, text=txt, bg=BG, fg=FG).grid(row=r, column=0, sticky="w", pady=4)
            e = tk.Entry(frame, width=50)
            e.grid(row=r, column=1, padx=10, pady=4)
            self.entries[key] = e

        ops = tk.Frame(parent, bg=BG)
        ops.pack(pady=10)
        buttons = [
            ("EMI", self.calc_emi),
            ("Future Value (FV)", self.future_value),
            ("Present Value (PV)", self.present_value),
            ("NPV", self.calc_npv),
            ("IRR", self.calc_irr),
            ("Payback Period", self.calc_payback),
            ("CAGR", self.calc_cagr),
            ("Break-even", self.break_even),
            ("What-if", self.what_if),
        ]
        for label, func in buttons:
            tk.Button(ops, text=label, width=16, bg=BTN, fg=FG, command=func).pack(side="left", padx=4, pady=4)

        self.output_frame = tk.Frame(parent, bg=BG)
        self.output_frame.pack(pady=10, fill='both', expand=True)

        self.output = tk.Text(self.output_frame,width=120, height=14, bg="#000000", fg=FG,font=("Consolas", 13))
        self.output.pack(side='left', fill='both', expand=True)

        yscroll = tk.Scrollbar(self.output_frame, orient='vertical', command=self.output.yview)
        xscroll = tk.Scrollbar(self.output_frame, orient='horizontal', command=self.output.xview)
        self.output.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        yscroll.pack(side='right', fill='y')
        xscroll.pack(side='bottom', fill='x')

        # hints
        self.entries["principal"].insert(0, "100000")
        self.entries["rate"].insert(0, "10")
        self.entries["time"].insert(0, "5")
        self.entries["cashflows"].insert(0, "-1000, 200, 300, 400, 500, 600")

    def _show(self, text):
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, str(text))

    def _get_float(self, key):
        try:
            return float(self.entries[key].get())
        except Exception:
            return None

    def _get_cashflows(self):
        try:
            return [float(x.strip()) for x in self.entries["cashflows"].get().split(",")]
        except Exception:
            return None

    def calc_emi(self):
        P = self._get_float("principal")
        r = self._get_float("rate")
        t = self._get_float("time")
        if None in (P, r, t): return self._show("Invalid inputs")
        monthly = r / (12 * 100)
        n = int(round(t * 12))
        try:
            emi = (P * monthly * (1 + monthly) ** n) / ((1 + monthly) ** n - 1)
            self._show(f"EMI = {emi:.3f} per month\nTotal paid ≈ {emi*n:.3f}")
        except Exception:
            self._show("Error computing EMI")

    def future_value(self):
        P = self._get_float("principal")
        r = self._get_float("rate")
        t = self._get_float("time")
        if None in (P, r, t): return self._show("Invalid inputs")
        FV = P * (1 + r / 100) ** t
        self._show(f"Future Value (FV) = {FV:.3f}")

    def present_value(self):
        FV = self._get_float("principal")
        r = self._get_float("rate")
        t = self._get_float("time")
        if None in (FV, r, t): return self._show("Invalid inputs")
        PV = FV / ((1 + r / 100) ** t)
        self._show(f"Present Value (PV) = {PV:.3f}")

    def calc_npv(self):
        r = self._get_float("rate")
        cf = self._get_cashflows()
        if r is None or cf is None: return self._show("Invalid inputs")
        rate = r / 100
        npv = sum(c / ((1 + rate) ** i) for i, c in enumerate(cf))
        self._show(f"NPV = {npv:.3f}")

    def calc_irr(self):
        cf = self._get_cashflows()
        if cf is None: return self._show("Invalid cashflows")
        # Try numpy_financial if available; else do a simple bisection
        try:
            import numpy_financial as npf
            irr = npf.irr(cf)
        except Exception:
            irr = self._irr_bisection(cf)
        if irr is None:
            self._show("IRR could not be computed")
        else:
            self._show(f"IRR ≈ {irr * 100:.3f}%")

    def _irr_bisection(self, cf, lo=-0.99, hi=5.0, iters=200):
        def npv(rate):
            return sum(c / ((1 + rate) ** i) for i, c in enumerate(cf))
        f_lo = npv(lo)
        f_hi = npv(hi)
        if f_lo == 0: return lo
        if f_hi == 0: return hi
        if f_lo * f_hi > 0:
            return None
        for _ in range(iters):
            mid = (lo + hi) / 2
            f_mid = npv(mid)
            if abs(f_mid) < 1e-9:
                return mid
            if f_lo * f_mid < 0:
                hi = mid; f_hi = f_mid
            else:
                lo = mid; f_lo = f_mid
        return (lo + hi) / 2

    def calc_payback(self):
        cf = self._get_cashflows()
        if cf is None: return self._show("Invalid cashflows")
        cum = 0.0
        for i, c in enumerate(cf):
            cum += c
            if cum >= 0:
                self._show(f"Payback Period ≈ {i} period(s) (0-indexed cashflows)")
                return
        self._show("Project never pays back within provided cashflows")

    def calc_cagr(self):
        P = self._get_float("principal")
        FV = self._get_float("rate")  
        t = self._get_float("time")
        if None in (P, FV, t) or P <= 0 or t <= 0: return self._show("Invalid inputs (principal>0, time>0)")
        cagr = ((FV / P) ** (1 / t) - 1) * 100
        self._show(f"CAGR = {cagr:.2f}%\n(Here: principal=PV, rate=FV, time=years)")

    def break_even(self):
        try:
            fixed_cost = float(self.entries["principal"].get())
            price = float(self.entries["rate"].get())
            var_cost = float(self.entries["time"].get())
        except Exception:
            return self._show("Enter: principal=fixed cost, rate=price, time=variable cost")
        contrib = price - var_cost
        if contrib <= 0:
            return self._show("Contribution must be positive (price > variable cost)")
        bep_units = fixed_cost / contrib
        bep_revenue = bep_units * price
        self._show(f"Break-even Units = {bep_units:.2f}\nBreak-even Revenue = {bep_revenue:.2f}")

    def what_if(self):
        try:
            price = float(self.entries["rate"].get())
            volume = float(self.entries["principal"].get())
            cost = float(self.entries["time"].get())
        except Exception:
            return self._show("Enter: rate=price, principal=volume, time=unit cost")
        out = ["Price | Profit"]
        out.append("-" * 24)
        for p in np.linspace(price * 0.8, price * 1.2, 10):
            profit = (p - cost) * volume
            out.append(f"{p:7.2f} | {profit:10.2f}")
        self._show("\n".join(out))


class ForecastTab:
    def __init__(self, parent):
        self.parent = parent
        self.df = None

        tk.Label(parent, text="Time-Series Forecasting", bg=BG, fg=FG, font=FONT_TITLE).pack(pady=10)

        load_frame = tk.Frame(parent, bg=BG)
        load_frame.pack(pady=5)
        tk.Button(load_frame, text="Load CSV", bg=BTN, fg=FG, command=self.load_csv).pack(side="left", padx=5)
        tk.Button(load_frame, text="Load Excel", bg=BTN, fg=FG, command=self.load_excel).pack(side="left", padx=5)

        col_frame = tk.Frame(parent, bg=BG)
        col_frame.pack(pady=5)
        tk.Label(col_frame, text="Numeric Column:", bg=BG, fg=FG).pack(side="left")
        self.col_var = ttk.Combobox(col_frame, width=30)
        self.col_var.pack(side="left", padx=8)

        method_frame = tk.Frame(parent, bg=BG)
        method_frame.pack(pady=5)
        tk.Label(method_frame, text="Method:", bg=BG, fg=FG).pack(side="left")
        self.method = ttk.Combobox(
            method_frame, width=28,
            values=["Simple Moving Average", "Weighted Moving Average", "Exponential Smoothing", "Linear Trend Forecast"]
        )
        self.method.pack(side="left", padx=8)
        self.method.current(0)

        tk.Label(method_frame, text="Window / Alpha / Periods:", bg=BG, fg=FG).pack(side="left")
        self.param_entry = tk.Entry(method_frame, width=12)
        self.param_entry.pack(side="left", padx=8)
        self.param_entry.insert(0, "5")

        tk.Button(method_frame, text="Run Forecast", bg=ACCENT, fg="black", command=self.run_forecast)\
            .pack(side="left", padx=12)

        self.output_frame = tk.Frame(parent, bg=BG)
        self.output_frame.pack(pady=10, fill='both', expand=True)

        self.output = tk.Text(self.output_frame,width=120, height=14, bg="#000000", fg=FG)
        self.output.pack(side='left', fill='both', expand=True)

        yscroll = tk.Scrollbar(self.output_frame, orient='vertical', command=self.output.yview)
        xscroll = tk.Scrollbar(self.output_frame, orient='horizontal', command=self.output.xview)
        self.output.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        yscroll.pack(side='right', fill='y')
        xscroll.pack(side='bottom', fill='x')

    def load_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not path: return
        self.df = pd.read_csv(path)
        self._after_load()

    def load_excel(self):
        path = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xlsx")])
        if not path: return
        self.df = pd.read_excel(path)
        self._after_load()

    def _after_load(self):
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, self.df.head().to_string())
        self.col_var["values"] = list(self.df.columns)

    def run_forecast(self):
        if self.df is None:
            return messagebox.showerror("Error", "Load a dataset first.")
        col = self.col_var.get().strip()
        if not col:
            return messagebox.showerror("Error", "Select a column.")
        try:
            series = pd.to_numeric(self.df[col], errors="coerce").dropna().reset_index(drop=True)
        except Exception:
            return messagebox.showerror("Error", "Column must be numeric.")
        if len(series) < 10:
            return messagebox.showerror("Error", "Need at least 10 numeric points")

        method = self.method.get()
        try:
            param = float(self.param_entry.get())
        except Exception:
            return messagebox.showerror("Error", "Enter numeric parameter")

        if method == "Simple Moving Average":
            self.simple_moving_average(series, int(param))
        elif method == "Weighted Moving Average":
            self.weighted_moving_average(series, int(param))
        elif method == "Exponential Smoothing":
            self.exponential_smoothing(series, float(param))
        else:
            self.linear_trend(series, int(param))

    def _show(self, text):
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, str(text))

    def _show_plot(self, series, line, title, future_x=None, future_values=None):
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        ax.plot(series.values, label="Original")
        ax.plot(line.values if isinstance(line, pd.Series) else line, label=title)
        if future_x is not None:
            ax.plot(future_x, future_values, label="Forecast")
        ax.set_title(title)
        ax.legend()

        win = tk.Toplevel(self.parent)
        win.title(title)
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def simple_moving_average(self, series, window):
        window = max(2, int(window))
        sma = series.rolling(window).mean()
        forecast = float(sma.iloc[-1])
        errors = (series[window:] - sma[window:]).abs()
        mae = float(errors.mean())
        den = series[window:].replace(0, np.nan)
        mape = float((errors / den).mean() * 100)
        self._show(
            f"Simple Moving Average (window={window})\nForecast (next) ≈ {forecast}\nMAE={mae}\nMAPE={mape}%"
        )
        self._show_plot(series, sma, "SMA")

    def weighted_moving_average(self, series, window):
        window = max(2, int(window))
        weights = np.arange(1, window + 1)
        wma_vals = [None] * (window - 1)
        for i in range(window - 1, len(series)):
            v = series.iloc[i - window + 1:i + 1].values
            wma_vals.append(float(np.dot(v, weights) / weights.sum()))
        wma = pd.Series(wma_vals)
        forecast = float(wma.dropna().iloc[-1])
        errors = (series[window:] - wma[window:]).abs()
        mae = float(errors.mean())
        den = series[window:].replace(0, np.nan)
        mape = float((errors / den).mean() * 100)
        self._show(
            f"Weighted Moving Average (window={window})\nForecast (next) ≈ {forecast}\nMAE={mae}\nMAPE={mape}%"
        )
        self._show_plot(series, wma, "WMA")

    def exponential_smoothing(self, series, alpha):
        alpha = float(alpha)
        if not (0 < alpha <= 1):
            return messagebox.showerror("Error", "alpha must be in (0,1]")
        smoothed = [float(series.iloc[0])]
        for t in range(1, len(series)):
            smoothed.append(alpha * float(series.iloc[t]) + (1 - alpha) * smoothed[t - 1])
        smoothed = pd.Series(smoothed)
        forecast = float(smoothed.iloc[-1])
        errors = (series - smoothed).abs()
        mae = float(errors.mean())
        den = series.replace(0, np.nan)
        mape = float((errors / den).mean() * 100)
        self._show(f"Exponential Smoothing (alpha={alpha})\nForecast ≈ {forecast}\nMAE={mae}\nMAPE={mape}%")
        self._show_plot(series, smoothed, "Exp Smoothing")

    def linear_trend(self, series, periods):
        periods = max(1, int(periods))
        x = np.arange(len(series))
        y = series.values.astype(float)
        coef = np.polyfit(x, y, 1)
        trend = coef[0] * x + coef[1]
        future_x = np.arange(len(series), len(series) + periods)
        future_vals = coef[0] * future_x + coef[1]
        self._show(f"Linear Trend\nSlope={coef[0]}\nIntercept={coef[1]}\nNext {periods}: {future_vals}")
        self._show_plot(series, trend, "Linear Trend", future_x, future_vals)


class PlotTab:
    def __init__(self, parent):
        self.parent = parent
        self.df = None

        tk.Label(parent, text="Visualization & Plotting Tools", bg=BG, fg=FG, font=FONT_TITLE).pack(pady=10)

        load_frame = tk.Frame(parent, bg=BG)
        load_frame.pack(pady=5)
        tk.Button(load_frame, text="Load CSV", bg=BTN, fg=FG, command=self.load_csv).pack(side="left", padx=5)
        tk.Button(load_frame, text="Load Excel", bg=BTN, fg=FG, command=self.load_excel).pack(side="left", padx=5)

        col_frame = tk.Frame(parent, bg=BG)
        col_frame.pack(pady=5)
        tk.Label(col_frame, text="Column X:", bg=BG, fg=FG).pack(side="left")
        self.col_x = ttk.Combobox(col_frame, width=20); self.col_x.pack(side="left", padx=8)
        tk.Label(col_frame, text="Column Y:", bg=BG, fg=FG).pack(side="left")
        self.col_y = ttk.Combobox(col_frame, width=20); self.col_y.pack(side="left", padx=8)

        btn_frame = tk.Frame(parent, bg=BG)
        btn_frame.pack(pady=8)
        for label, fn in [
            ("Scatter Plot", self.scatter_plot),
            ("Line Plot", self.line_plot),
            ("Histogram", self.histogram),
            ("Boxplot", self.boxplot),
            ("Bar Chart", self.bar_chart),
            ("Multi-Series Plot", self.multi_series_plot),
        ]:
            tk.Button(btn_frame, text=label, width=18, bg=BTN, fg=FG, command=fn).pack(side="left", padx=5)

        self.output_frame = tk.Frame(parent, bg=BG)
        self.output_frame.pack(pady=10, fill='both', expand=True)

        self.output = tk.Text(self.output_frame,width=120, height=14, bg="#000000", fg=FG)
        self.output.pack(side='left', fill='both', expand=True)

        yscroll = tk.Scrollbar(self.output_frame, orient='vertical', command=self.output.yview)
        xscroll = tk.Scrollbar(self.output_frame, orient='horizontal', command=self.output.xview)
        self.output.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        yscroll.pack(side='right', fill='y')
        xscroll.pack(side='bottom', fill='x')

    def load_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not path: return
        self.df = pd.read_csv(path)
        self._after_load()

    def load_excel(self):
        path = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xlsx")])
        if not path: return
        self.df = pd.read_excel(path)
        self._after_load()

    def _after_load(self):
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, self.df.head().to_string())
        cols = list(self.df.columns)
        self.col_x["values"] = cols
        self.col_y["values"] = cols

    def _show_fig(self, fig, title):
        win = tk.Toplevel(self.parent)
        win.title(title)
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def scatter_plot(self):
        if self.df is None: return
        x, y = self.col_x.get(), self.col_y.get()
        if not x or not y: return
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        ax.scatter(self.df[x], self.df[y])
        ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title("Scatter Plot")
        self._show_fig(fig, "Scatter Plot")

    def line_plot(self):
        if self.df is None: return
        x, y = self.col_x.get(), self.col_y.get()
        if not x or not y: return
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        ax.plot(self.df[x], self.df[y])
        ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title("Line Plot")
        self._show_fig(fig, "Line Plot")

    def histogram(self):
        if self.df is None: return
        x = self.col_x.get()
        if not x: return
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        ax.hist(pd.to_numeric(self.df[x], errors="coerce").dropna(), bins=20)
        ax.set_xlabel(x); ax.set_title("Histogram")
        self._show_fig(fig, "Histogram")

    def boxplot(self):
        if self.df is None: return
        x = self.col_x.get()
        if not x: return
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        ax.boxplot(pd.to_numeric(self.df[x], errors="coerce").dropna())
        ax.set_title(f"Boxplot of {x}")
        self._show_fig(fig, "Boxplot")

    def bar_chart(self):
        if self.df is None: return
        x, y = self.col_x.get(), self.col_y.get()
        if not x or not y: return
        fig = plt.figure(figsize=(7, 4))
        ax = fig.add_subplot(111)
        ax.bar(self.df[x].astype(str), pd.to_numeric(self.df[y], errors="coerce"))
        ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title("Bar Chart")
        self._show_fig(fig, "Bar Chart")

    def multi_series_plot(self):
        if self.df is None: return
        fig = plt.figure(figsize=(7, 4))
        ax = fig.add_subplot(111)
        for col in self.df.columns:
            s = pd.to_numeric(self.df[col], errors="coerce")
            if s.notna().sum() > 0:
                ax.plot(s.values, label=str(col))
        ax.legend()
        ax.set_title("Multi-Series Plot (numeric columns)")
        self._show_fig(fig, "Multi-Series Plot")


class ConverterTab:
    def __init__(self, parent):
        self.parent = parent
        tk.Label(parent, text="Unit Converter & Date Calculator", bg=BG, fg=FG, font=FONT_TITLE).pack(pady=10)

        unit_frame = tk.LabelFrame(parent, text="Unit Converter", bg=BG, fg=FG, font=FONT_MED)
        unit_frame.pack(pady=10)

        tk.Label(unit_frame, text="Value:", bg=BG, fg=FG).grid(row=0, column=0, sticky="w")
        self.unit_value = tk.Entry(unit_frame, width=15)
        self.unit_value.grid(row=0, column=1, padx=10, pady=5)

        units = [
            "meter", "kilometer", "mile",
            "gram", "kilogram", "pound",
            "celsius", "fahrenheit", "kelvin",
            "byte", "kilobyte", "megabyte", "gigabyte"
        ]

        tk.Label(unit_frame, text="From:", bg=BG, fg=FG).grid(row=1, column=0, sticky="w")
        self.from_unit = ttk.Combobox(unit_frame, values=units, width=15)
        self.from_unit.grid(row=1, column=1, padx=10, pady=5)

        tk.Label(unit_frame, text="To:", bg=BG, fg=FG).grid(row=1, column=2, sticky="w")
        self.to_unit = ttk.Combobox(unit_frame, values=units, width=15)
        self.to_unit.grid(row=1, column=3, padx=10, pady=5)

        tk.Button(unit_frame, text="Convert", bg=BTN, fg=FG, command=self.convert_unit)\
            .grid(row=2, column=1, pady=10)

        self.unit_output = tk.Label(unit_frame, text="", bg=BG, fg=ACCENT, font=FONT_MED)
        self.unit_output.grid(row=3, column=0, columnspan=4, pady=5)

        date_frame = tk.LabelFrame(parent, text="Date Calculator", bg=BG, fg=FG, font=FONT_MED)
        date_frame.pack(pady=10)

        tk.Label(date_frame, text="Date 1 (YYYY-MM-DD):", bg=BG, fg=FG).grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.date1 = tk.Entry(date_frame, width=15); self.date1.grid(row=0, column=1, padx=10, pady=5)

        tk.Label(date_frame, text="Date 2 (YYYY-MM-DD) OR Days:", bg=BG, fg=FG).grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.date2 = tk.Entry(date_frame, width=15); self.date2.grid(row=1, column=1, padx=10, pady=5)

        tk.Button(date_frame, text="Difference", bg=BTN, fg=FG, command=self.date_difference).grid(row=0, column=2, padx=10)
        self.date_output = tk.Label(date_frame, text="", bg=BG, fg=ACCENT, font=FONT_MED)
        self.date_output.grid(row=2, column=0, columnspan=3, pady=5)

        # defaults
        self.unit_value.insert(0, "100")
        self.from_unit.set("meter")
        self.to_unit.set("kilometer")
        self.date1.insert(0, "2025-01-01")
        self.date2.insert(0, "2025-12-31")

    def convert_unit(self):
        try:
            value = float(self.unit_value.get())
        except Exception:
            self.unit_output.config(text="Invalid number")
            return

        from_u = self.from_unit.get()
        to_u = self.to_unit.get()

        length = {"meter": 1, "kilometer": 1000, "mile": 1609.34}
        weight = {"gram": 1, "kilogram": 1000, "pound": 453.592}
        digital = {"byte": 1, "kilobyte": 1024, "megabyte": 1024 ** 2, "gigabyte": 1024 ** 3}

        if from_u in length and to_u in length:
            converted = value * (length[from_u] / length[to_u])
            self.unit_output.config(text=f"Converted: {converted}")
            return
        if from_u in weight and to_u in weight:
            converted = value * (weight[from_u] / weight[to_u])
            self.unit_output.config(text=f"Converted: {converted}")
            return
        if from_u in digital and to_u in digital:
            converted = value * (digital[from_u] / digital[to_u])
            self.unit_output.config(text=f"Converted: {converted}")
            return

        try:
            if from_u == "celsius":
                if to_u == "fahrenheit": converted = value * 9/5 + 32
                elif to_u == "kelvin": converted = value + 273.15
                else: converted = value
            elif from_u == "fahrenheit":
                if to_u == "celsius": converted = (value - 32) * 5/9
                elif to_u == "kelvin": converted = (value - 32) * 5/9 + 273.15
                else: converted = value
            elif from_u == "kelvin":
                if to_u == "celsius": converted = value - 273.15
                elif to_u == "fahrenheit": converted = (value - 273.15) * 9/5 + 32
                else: converted = value
            else:
                converted = "Unsupported units"
            self.unit_output.config(text=f"Converted: {converted}")
        except Exception:
            self.unit_output.config(text="Conversion error")

    def date_difference(self):
        try:
            d1 = _parse_date_flexible(self.date1.get())
            d2 = _parse_date_flexible(self.date2.get())
            diff = abs((d2 - d1).days)
            self.date_output.config(text=f"Difference: {diff} days")
        except Exception:
            self.date_output.config(text="Invalid date (use YYYY-MM-DD, DD-MM-YYYY, DD/MM/YYYY)")

    def date_add(self):
        try:
            d1 = _parse_date_flexible(self.date1.get())
            days = int(float(self.date2.get().strip()))
            new_date = d1 + datetime.timedelta(days=days)
            self.date_output.config(text=f"New Date: {new_date.date()}")
        except Exception:
            self.date_output.config(text="Invalid input (Date2 should be days)")


class FormulaEditorTab:
    def __init__(self, parent):
        self.parent = parent
        self.formulas = {}

        tk.Label(parent, text="Custom Formula Editor", bg=BG, fg=FG, font=FONT_TITLE).pack(pady=10)

        frame = tk.Frame(parent, bg=BG)
        frame.pack(pady=8)

        tk.Label(frame, text="Formula Name:", bg=BG, fg=FG).grid(row=0, column=0, sticky="w")
        self.fname = tk.Entry(frame, width=25); self.fname.grid(row=0, column=1, padx=10)

        tk.Label(frame, text="Expression (use x,y,z):", bg=BG, fg=FG).grid(row=1, column=0, sticky="w")
        self.fexpr = tk.Entry(frame, width=55); self.fexpr.grid(row=1, column=1, padx=10)

        tk.Button(frame, text="Save Formula", bg=BTN, fg=FG, command=self.save_formula)\
            .grid(row=2, column=0, columnspan=2, pady=8)

        exec_frame = tk.Frame(parent, bg=BG)
        exec_frame.pack(pady=5)

        tk.Label(exec_frame, text="Values for x,y,z:", bg=BG, fg=FG).pack(side="left")
        self.fvals = tk.Entry(exec_frame, width=35); self.fvals.pack(side="left", padx=10)
        tk.Button(exec_frame, text="Run Formula", bg=BTN, fg=FG, command=self.run_formula).pack(side="left")

        self.output_frame = tk.Frame(parent, bg=BG)
        self.output_frame.pack(pady=10, fill='both', expand=True)

        self.output = tk.Text(self.output_frame,width=120, height=14, bg="#000000", fg=FG)
        self.output.pack(side='left', fill='both', expand=True)

        yscroll = tk.Scrollbar(self.output_frame, orient='vertical', command=self.output.yview)
        xscroll = tk.Scrollbar(self.output_frame, orient='horizontal', command=self.output.xview)
        self.output.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        yscroll.pack(side='right', fill='y')
        xscroll.pack(side='bottom', fill='x')

        # defaults
        self.fname.insert(0, "profit")
        self.fexpr.insert(0, "x - y")
        self.fvals.insert(0, "1000, 650, 0")

    def save_formula(self):
        name = self.fname.get().strip()
        expr = self.fexpr.get().strip()
        if not name or not expr:
            self.output.insert(tk.END, "Invalid formula.\n")
            return
        self.formulas[name] = expr
        self.output.insert(tk.END, f"Saved: {name} = {expr}\n")

    def run_formula(self):
        name = self.fname.get().strip()
        if name not in self.formulas:
            self.output.insert(tk.END, "Formula not found.\n")
            return
        expr = self.formulas[name]
        try:
            vals = [float(v.strip()) for v in self.fvals.get().split(",")]
            while len(vals) < 3:
                vals.append(0.0)
            x, y, z = vals[:3]
            # safe-ish eval: only x,y,z and math/np
            result = eval(expr, {"__builtins__": {}}, {"x": x, "y": y, "z": z, "math": math, "np": np})
            self.output.insert(tk.END, f"{name}({x},{y},{z}) = {result}\n")
        except Exception as e:
            self.output.insert(tk.END, f"Error: {e}\n")


class ReportTab:
    def __init__(self, parent):
        self.parent = parent
        self.df = None
        self.image_sections = []

        tk.Label(parent, text="Report Generator (HTML)", bg=BG, fg=FG, font=FONT_TITLE).pack(pady=10)

        load_frame = tk.Frame(parent, bg=BG)
        load_frame.pack(pady=5)
        tk.Button(load_frame, text="Load CSV", bg=BTN, fg=FG, command=self.load_csv).pack(side="left", padx=5)
        tk.Button(load_frame, text="Load Excel", bg=BTN, fg=FG, command=self.load_excel).pack(side="left", padx=5)

        col_frame = tk.Frame(parent, bg=BG)
        col_frame.pack(pady=5)
        tk.Label(col_frame, text="Column:", bg=BG, fg=FG).pack(side="left")
        self.col_var = ttk.Combobox(col_frame, width=30)
        self.col_var.pack(side="left", padx=8)

        tk.Button(col_frame, text="Add Histogram", bg=BTN, fg=FG, command=self.add_histogram).pack(side="left", padx=8)
        tk.Button(col_frame, text="Add Line Plot", bg=BTN, fg=FG, command=self.add_lineplot).pack(side="left", padx=8)
        tk.Button(col_frame, text="Clear Plots", bg=BTN, fg=FG, command=self.clear_plots).pack(side="left", padx=8)

        tk.Button(parent, text="Generate HTML Report", bg=ACCENT, fg="black", width=25, command=self.generate_report)\
            .pack(pady=10)

        self.output_frame = tk.Frame(parent, bg=BG)
        self.output_frame.pack(pady=10, fill='both', expand=True)

        self.output = tk.Text(self.output_frame,width=120, height=14, bg="#000000", fg=FG)
        self.output.pack(side='left', fill='both', expand=True)

        yscroll = tk.Scrollbar(self.output_frame, orient='vertical', command=self.output.yview)
        xscroll = tk.Scrollbar(self.output_frame, orient='horizontal', command=self.output.xview)
        self.output.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        yscroll.pack(side='right', fill='y')
        xscroll.pack(side='bottom', fill='x')

    def load_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not path: return
        self.df = pd.read_csv(path)
        self._after_load()

    def load_excel(self):
        path = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xlsx")])
        if not path: return
        self.df = pd.read_excel(path)
        self._after_load()

    def _after_load(self):
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, self.df.head().to_string())
        self.col_var["values"] = list(self.df.columns)
        self.image_sections.clear()

    def clear_plots(self):
        self.image_sections.clear()
        self.output.insert(tk.END, "\nCleared saved plots for report.\n")

    def _fig_to_b64(self, fig):
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()
        return encoded

    def add_histogram(self):
        if self.df is None: return
        col = self.col_var.get().strip()
        if not col: return
        s = pd.to_numeric(self.df[col], errors="coerce").dropna()
        if len(s) == 0: return messagebox.showerror("Error", "Selected column has no numeric data")

        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        ax.hist(s, bins=25)
        ax.set_title(f"Histogram of {col}")
        ax.set_xlabel(col)

        self.image_sections.append((f"Histogram of {col}", self._fig_to_b64(fig)))
        plt.close(fig)
        self.output.insert(tk.END, f"\nAdded Histogram for {col}\n")

    def add_lineplot(self):
        if self.df is None: return
        col = self.col_var.get().strip()
        if not col: return
        s = pd.to_numeric(self.df[col], errors="coerce").dropna().reset_index(drop=True)
        if len(s) == 0: return messagebox.showerror("Error", "Selected column has no numeric data")

        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        ax.plot(s.values)
        ax.set_title(f"Line Plot of {col}")
        ax.set_xlabel("Index")
        ax.set_ylabel(col)

        self.image_sections.append((f"Line Plot of {col}", self._fig_to_b64(fig)))
        plt.close(fig)
        self.output.insert(tk.END, f"\nAdded Line Plot for {col}\n")

    def generate_report(self):
        if self.df is None:
            return messagebox.showerror("Error", "Load data first")

        save_path = filedialog.asksaveasfilename(defaultextension=".html", filetypes=[("HTML Files", "*.html")])
        if not save_path: return

        try:
            summary_html = self.df.describe(include="all").to_html(classes="table")
        except Exception:
            summary_html = "<p>Summary failed.</p>"

        html = f"""
<html>
<head>
<title>Data Report</title>
<style>
body {{ background: #1e1e1e; color: #ffffff; font-family: Arial; }}
.table {{ background: #333333; color: #ffffff; border-collapse: collapse; }}
.table td, .table th {{ border: 1px solid #555; padding: 6px; }}
h1, h2 {{ color: #4ea1d3; }}
pre {{ background: #111; padding: 10px; border: 1px solid #333; overflow-x: auto; }}
</style>
</head>
<body>
<h1>Dataset Report</h1>

<h2>Dataset Preview</h2>
<pre>{self.df.head().to_string()}</pre>

<h2>Summary Statistics</h2>
{summary_html}
"""
        for title, img in self.image_sections:
            html += f"""
<h2>{title}</h2>
<img src="data:image/png;base64,{img}" width="700">
"""
        html += "</body></html>"

        try:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(html)
            self.output.insert(tk.END, f"\nReport Saved: {save_path}\n")
            webbrowser.open(save_path)
        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    root.update_idletasks()
    root.state("zoomed")   
    app = AdvancedCalculatorApp(root)
    root.mainloop()
