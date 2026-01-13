#!/usr/bin/env python3
"""
IAthon – Iteration 13 (All Fixes Applied)
- Live pricing fetch from OpenAI with fallbacks
- Prevents file reading errors
- Generates comprehensive Markdown reports
- Creates multiple diverse visualizations
- Performs statistical tests
- Always shows cost summary
"""

import argparse
import os
import shutil      
import subprocess
import re
import time
import warnings
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional
from datetime import datetime

warnings.filterwarnings("ignore")

# --- Core scientific stack ---
import pandas as pd
import numpy as np
from scipy import stats

# --- Plotting (headless-safe) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns  
import plotly.express as px
import plotly.graph_objects as go

from openai import OpenAI

# For web scraping (optional - will fallback if not available)
try:
    import requests
    from bs4 import BeautifulSoup
    WEB_SCRAPING_AVAILABLE = True
except ImportError:
    WEB_SCRAPING_AVAILABLE = False
    print("⚠️  requests/beautifulsoup4 not available. Install for live pricing: pip install requests beautifulsoup4")

# Try to import python-pptx
try:
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.enum.text import PP_ALIGN
    from pptx.dml.color import RGBColor
    PPTX_AVAILABLE = True
except ImportError:
    PPTX_AVAILABLE = False

# ======================================================
# FALLBACK PRICING (if web fetch fails)
# Updated January 2025
# ======================================================
FALLBACK_PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "o1": {"input": 15.00, "output": 60.00},
    "o1-mini": {"input": 3.00, "output": 12.00},
    "o3-mini": {"input": 1.10, "output": 4.40},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "dall-e-3": {"standard_1024": 0.040, "standard_1792": 0.080, "hd_1024": 0.080, "hd_1792": 0.120},
    "dall-e-2": {"1024": 0.020, "512": 0.018, "256": 0.016},
}

# ======================================================
# Token Usage Tracker with Live Pricing
# ======================================================
@dataclass
class TokenUsage:
    """Track token usage per model"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    def add(self, prompt: int, completion: int):
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += (prompt + completion)

@dataclass
class UsageTracker:
    """Central tracker with live pricing fetch"""
    models: Dict[str, TokenUsage] = field(default_factory=dict)
    pricing_cache: Dict[str, Dict[str, float]] = field(default_factory=dict)
    pricing_source: str = "unknown"
    cache_file: Path = field(default_factory=lambda: Path.home() / ".cache" / "iathlon" / "openai_pricing.json")
    
    def __post_init__(self):
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.fetch_pricing()
    
    def _fetch_live_pricing_from_web(self) -> Optional[Dict]:
        """Attempt to fetch pricing from OpenAI's website"""
        if not WEB_SCRAPING_AVAILABLE:
            return None
        
        print("🌐 Fetching live pricing from OpenAI...")
        
        try:
            response = requests.get(
                "https://openai.com/api/pricing/",
                timeout=15,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                }
            )
            
            if response.status_code == 200:
                text = response.text
                extracted = {}
                
                # Try to find JSON pricing data
                json_patterns = [r'"pricing":\s*({[^}]+})', r'pricing:\s*({[^}]+})']
                for pattern in json_patterns:
                    for match in re.finditer(pattern, text):
                        try:
                            pricing_json = json.loads(match.group(1))
                            if isinstance(pricing_json, dict):
                                extracted.update(pricing_json)
                        except:
                            continue
                
                # Fallback: regex patterns
                price_patterns = [
                    r'(gpt-[\w.-]+|o\d+(?:-[\w]+)?)\s*[^\d]*\$\s*([\d.]+)\s*(?:/|per)\s*(?:1M|million)[^\$]*\$\s*([\d.]+)\s*(?:/|per)\s*(?:1M|million)',
                ]
                
                for pattern in price_patterns:
                    for match in re.finditer(pattern, text, re.IGNORECASE):
                        try:
                            model = match.group(1).strip().lower()
                            input_price = float(match.group(2))
                            output_price = float(match.group(3))
                            extracted[model] = {"input": input_price, "output": output_price}
                        except:
                            continue
                
                if extracted:
                    print(f"✅ Fetched pricing for {len(extracted)} models")
                    cache_data = {
                        'timestamp': datetime.now().isoformat(),
                        'source': 'https://openai.com/api/pricing/',
                        'pricing': extracted
                    }
                    with open(self.cache_file, 'w') as f:
                        json.dump(cache_data, f, indent=2)
                    return extracted
                else:
                    print(f"⚠️  Page loaded but no pricing extracted (JavaScript-rendered content)")
        except Exception as e:
            print(f"⚠️  Web fetch failed: {str(e)[:80]}")
        
        return None
    
    def _load_from_cache(self) -> Optional[Dict]:
        """Load from cache"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                timestamp = cache_data.get('timestamp', 'unknown')
                pricing = cache_data.get('pricing', {})
                if pricing:
                    print(f"💾 Loaded cached pricing from {timestamp}")
                    return pricing
        except Exception as e:
            print(f"⚠️  Cache load failed: {e}")
        return None
    
    def fetch_pricing(self):
        """Fetch with fallback chain"""
        live_pricing = self._fetch_live_pricing_from_web()
        if live_pricing:
            self.pricing_cache = live_pricing
            self.pricing_source = "live"
            return
        
        cached_pricing = self._load_from_cache()
        if cached_pricing:
            self.pricing_cache = cached_pricing
            self.pricing_source = "cache"
            return
        
        print("📋 Using fallback pricing (January 2025)")
        self.pricing_cache = FALLBACK_PRICING.copy()
        self.pricing_source = "fallback"
    
    def _get_model_pricing(self, model_name: str) -> Dict[str, float]:
        model_lower = model_name.lower()
        if model_name in self.pricing_cache:
            return self.pricing_cache[model_name]
        
        # Priority matching for known tiers
        if "o1" in model_lower or "gpt-4" in model_lower and "mini" not in model_lower:
            return FALLBACK_PRICING.get("gpt-4o", {"input": 2.50, "output": 10.00})
        if "mini" in model_lower or "gpt-3.5" in model_lower:
            return FALLBACK_PRICING.get("gpt-4o-mini", {"input": 0.15, "output": 0.60})
            
        return {"input": 2.50, "output": 10.00} # Default to standard 4o rates
        
    def record(self, model: str, usage):
        """Record usage from OpenAI response"""
        if model not in self.models:
            self.models[model] = TokenUsage()
        
        prompt = 0
        completion = 0
        
        for prompt_attr in ['prompt_tokens', 'input_tokens', 'total_input_tokens']:
            if hasattr(usage, prompt_attr):
                prompt = getattr(usage, prompt_attr)
                break
        
        for completion_attr in ['completion_tokens', 'output_tokens', 'total_output_tokens']:
            if hasattr(usage, completion_attr):
                completion = getattr(usage, completion_attr)
                break
        
        if prompt == 0 and completion == 0:
            try:
                if hasattr(usage, '__dict__'):
                    usage_dict = usage.__dict__
                elif hasattr(usage, 'model_dump'):
                    usage_dict = usage.model_dump()
                else:
                    usage_dict = dict(usage)
                prompt = usage_dict.get('prompt_tokens', 0) or usage_dict.get('input_tokens', 0)
                completion = usage_dict.get('completion_tokens', 0) or usage_dict.get('output_tokens', 0)
            except:
                pass
        
        self.models[model].add(prompt, completion)
    
    def record_image(self, model: str, num_images: int = 1, size: str = "1024x1024", quality: str = "standard"):
        """Record image generation"""
        if model not in self.models:
            self.models[model] = TokenUsage()
        self.models[model].add(num_images, 0)
        if not hasattr(self.models[model], 'image_specs'):
            self.models[model].image_specs = []
        self.models[model].image_specs.append({'size': size, 'quality': quality})
    
    def print_summary(self):
        """Print cost summary"""
        print("\n" + "="*80)
        print("💰 TOKEN USAGE & COST SUMMARY")
        print("="*80)
        
        source_indicators = {
            'live': '✅ Live from OpenAI',
            'cache': '💾 Cached pricing',
            'fallback': '📋 Fallback pricing'
        }
        print(f"📊 Pricing: {source_indicators.get(self.pricing_source, 'Unknown')}")
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        total_cost = 0.0
        
        for model, usage in self.models.items():
            print(f"\n🤖 {model}")
            
            if 'dall-e' in model.lower():
                num_images = usage.prompt_tokens
                print(f"   Images: {num_images}")
                if hasattr(usage, 'image_specs'):
                    model_cost = 0
                    pricing = self._get_model_pricing(model)
                    for spec in usage.image_specs:
                        size, quality = spec['size'], spec['quality']
                        if 'dall-e-3' in model.lower():
                            key = f"{'hd' if quality == 'hd' else 'standard'}_{size.split('x')[0]}"
                            cost = pricing.get(key, 0.040)
                        else:
                            cost = pricing.get(size.split('x')[0], 0.020)
                        model_cost += cost
                    total_cost += model_cost
                    print(f"   Cost: ${model_cost:.6f}")
            else:
                print(f"   Input:  {usage.prompt_tokens:>10,} tokens")
                print(f"   Output: {usage.completion_tokens:>10,} tokens")
                print(f"   Total:  {usage.total_tokens:>10,} tokens")
                
                pricing = self._get_model_pricing(model)
                input_cost = (usage.prompt_tokens / 1_000_000) * pricing["input"]
                output_cost = (usage.completion_tokens / 1_000_000) * pricing["output"]
                model_cost = input_cost + output_cost
                total_cost += model_cost
                
                print(f"   Rate: ${pricing['input']:.3f}/${pricing['output']:.3f} per 1M")
                print(f"   Cost: ${model_cost:.6f}")
        
        print(f"\n{'='*80}")
        print(f"💵 TOTAL COST: ${total_cost:.6f}")
        print("="*80)
        
        return total_cost
    
    def save_log(self, output_dir: Path):
        """Save log file"""
        log_path = output_dir / "token_usage.log"
        with open(log_path, 'w') as f:
            f.write(f"TOKEN USAGE LOG\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Pricing: {self.pricing_source}\n\n")
            
            total = 0.0
            for model, usage in self.models.items():
                f.write(f"{model}:\n")
                if 'dall-e' not in model.lower():
                    pricing = self._get_model_pricing(model)
                    cost = (usage.prompt_tokens / 1_000_000) * pricing["input"] + \
                           (usage.completion_tokens / 1_000_000) * pricing["output"]
                    f.write(f"  Tokens: {usage.total_tokens:,}\n")
                    f.write(f"  Cost: ${cost:.6f}\n")
                    total += cost
                f.write("\n")
            f.write(f"TOTAL: ${total:.6f}\n")
        print(f"📄 Log saved: {log_path}")

# ======================================================
# Utilities
# ======================================================
def get_api_key(cli_key=None):
    return cli_key or os.getenv("OPENAI_API_KEY")

def robust_rmtree(path: Path):
    """Delete directory with retry"""
    if not path.exists():
        return
    for _ in range(3):
        try:
            shutil.rmtree(path)
            return
        except PermissionError:
            time.sleep(1)
    for item in path.iterdir():
        try:
            if item.is_file(): item.unlink()
            elif item.is_dir(): shutil.rmtree(item)
        except:
            pass

# ======================================================
# Data Loading
# ======================================================
def load_and_clean(path: Path):
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, on_bad_lines='skip')
    else:
        df = pd.read_excel(path)

    df = df.dropna(axis=1, thresh=len(df) * 0.5)
    df = df.loc[:, df.nunique() > 1]

    num_features = df.select_dtypes(include=np.number).columns.tolist()
    cat_features = df.select_dtypes(include=["object", "category"]).columns.tolist()

    return df, num_features, cat_features

# ======================================================
# Safe Execution Environment
# ======================================================
class SafeRunner:
    def __init__(self, df, num_features, cat_features, output_dir: Path, verbose=False):
        self.df = df
        self.num_features = num_features
        self.cat_features = cat_features
        self.output_dir = output_dir.resolve() 
        self.verbose = verbose
        self.figures_dir = self.output_dir / "figures"
        
        if self.figures_dir.exists():
            if self.verbose: print(f"🧹 Cleaning old figures...")
            robust_rmtree(self.figures_dir)
        
        self.figures_dir.mkdir(parents=True, exist_ok=True)
    
    def run(self, code):
        assert "exec(" not in code.lower(), "Generated code must not call exec()"

        if "```" in code:
            match = re.search(r"```(?:python)?\n?(.*?)\n?```", code, re.DOTALL)
            if match:
                code = match.group(1)
            else:
                code = "\n".join([line for line in code.split("\n") if "```" not in line])

        
        # Block file reading
        forbidden = [
            (r'pd\.read_csv\s*\(', "pd.read_csv()"),
            (r'pd\.read_excel\s*\(', "pd.read_excel()"),
            (r'pd\.read_json\s*\(', "pd.read_json()"),
        ]
        
        for pattern, func in forbidden:
            if re.search(pattern, code, re.IGNORECASE):
                raise RuntimeError(f"❌ Attempted {func} - use existing 'df' variable instead")

        df_copy = self.df.copy()

        globals_safe = {
            "df": df_copy,
            "num_features": self.num_features, 
            "cat_features": self.cat_features, 
            "plt": plt, "sns": sns, "px": px, "go": go, 
            "stats": stats, "np": np, "pd": pd,
            "FIGURES_DIR": str(self.figures_dir),
            "os": os
        }
        
        code = re.sub(
            r"plt\.savefig\(\s*['\"]([^'\"]+\.png)['\"]\s*\)",
            r"plt.savefig(os.path.join(FIGURES_DIR, '\1'))",
            code)


        old_cwd = os.getcwd()
        os.chdir(self.figures_dir)

        try:
            exec(code, globals_safe)

        except NameError as e:
            raise RuntimeError(
                f"Generated code referenced an undefined variable: {e}. "
                "Ensure all intermediate variables (e.g., df_clean) are defined "
                "before being used."
            )

        except Exception as e:
            raise RuntimeError(str(e))

        finally:
            os.chdir(old_cwd)

        pngs = list(self.figures_dir.glob("*.png"))
        if not pngs:
            raise RuntimeError(
                "No figures were generated. Ensure plots are saved to FIGURES_DIR.")


# ======================================================
# Decision Maker
# ======================================================
class DecisionMaker:
    def __init__(self, api_key, tracker, custom_prompt=None, verbose=False):
        self.client = OpenAI(api_key=api_key)
        self.tracker = tracker
        self.custom_prompt = custom_prompt
        self.verbose = verbose
        self.model = "gpt-5-mini"
    
    def decide(self, observations: str, step_num: int, max_steps: int, 
               num_features: list, cat_features: list, data_preview: str) -> str:
        
        context = ""
        if self.custom_prompt:
            context = f"USER CONTEXT: {self.custom_prompt}\n\n"
       

        prompt = f"""{context}DATA PREVIEW:
{data_preview}

NUMERIC: {num_features}
CATEGORICAL: {cat_features}

STEP {step_num + 1}/{max_steps}
OBSERVATIONS: {observations}

STRATEGY - Create diverse, comprehensive analysis:
1. Distribution plots (histograms, KDE)
2. Correlation heatmaps
3. Scatter plots for relationships
4. Box plots for group comparisons
5. Statistical tests (t-tests, ANOVA, chi-square, correlations)
6. Time series if applicable
7. Outlier detection
8. MANDATORY: Kaplan-Meier Survival Analysis and Log-rank tests IF variables like 'death', 'status', or 'time' are detected in the preview.
9. Log-rank or Cox regression tests to compare survival between groups

GOALS:
- Generate 8-10 different visualizations
- Perform 3+ statistical tests  
- Use varied plot types
- Focus on most interesting patterns

Choose ONE specific next action. Be concrete about variables.
If {step_num + 1} >= 15, respond "STOP"."""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            # temperature=0.4,
        )
        
        if hasattr(response, 'usage'):
            self.tracker.record(self.model, response.usage)
        
        return response.choices[0].message.content.strip()
    



    def synthesize_report(self, analysis_log, original_prompt, figures_dir, data_preview):
        if self.verbose: print("✍️ Synthesizing Final Report...")
        
        figs = sorted(figures_dir.glob("*.png"))
        fig_list = "\n".join([f"figures/{f.name}" for f in figs])
        context = f"USER CONTEXT: {self.custom_prompt}\n\n" if self.custom_prompt else ""

        prompt = f"""{context}
You are a professional presentation designer. Create a Markdown document optimized for PPTX.

AVAILABLE FIGURES: {fig_list}
ANALYSIS RESULTS: {analysis_log}

STRICT SLIDE REQUIREMENTS:
1. Each H1 (#) or H2 (##) starts a NEW slide.
2. NO ADVICE OR RECOMMENDATIONS on "what to do next". Stick to the data results.
3. TABLE SPACING: Include TWO empty lines between a table caption and the table itself.
4. Use the EXACT filenames from the AVAILABLE FIGURES list.
5. Create a Title slide, an Executive Summary slide, and several Results slides.
"""
        response = self.client.chat.completions.create(
            model=self.model, 
            messages=[
                {"role": "system", "content": "You are a slide generation engine. No meta-talk."},
                {"role": "user", "content": prompt}
            ]
            # temperature=0.2 REMOVED
        )



    def synthesize_report(self, analysis_log, original_prompt, figures_dir, data_preview):
        if self.verbose: print("✍️ Synthesizing Final Report...")
        
        # Get actual files on disk to prevent hallucination
        figs = sorted(figures_dir.glob("*.png"))
        fig_list = "\n".join([f"figures/{f.name}" for f in figs])

        context = f"USER CONTEXT: {self.custom_prompt}\n\n" if self.custom_prompt else ""

        # Use a VERY strict system-style prompt to remove conversational "filler"
        prompt = f"""{context}
You are a technical reporting engine. Write a professional, publication-ready Data Analysis 
Report in Markdown format.

DATA PREVIEW:
{data_preview}

ANALYSIS STEPS & RESULTS:
{analysis_log}

AVAILABLE FIGURES (USE ONLY THESE FILENAMES):
{fig_list}

STRICT REQUIREMENTS:
1. DO NOT include conversational filler like "What would you like to do next?" or "I can do these steps".
2. DO NOT offer to re-run analyses.
3. INTEGRATE figures: Use ONLY the filenames listed in "AVAILABLE FIGURES".
        Place each ![Description](figures/filename.png) tag immediately after the paragraph discussing that specific result.
        Embed figures using: ![Description](figures/filename.png)
4. Use professional headers (#, ##, ###).
5. Include a "Statistical Summary" section with exact p-values and means found in the analysis.
6. END the report with a "Conclusions" section. Do NOT add anything after it.
7. Every table caption MUST be followed by TWO empty lines to ensure proper rendering.

REPORT STRUCTURE:
# [Title]
## Executive Summary
## Methodology
## Analysis & Results (Embed figures here)
## Statistical Tables
## Conclusions and Recommendations
"""
        response = self.client.chat.completions.create(
            model=self.model, 
            messages=[
                {"role": "system", "content": "You are a professional medical and data science reporter. You output only the report text, no meta-talk."},
                {"role": "user", "content": prompt}
            ],
            # temperature=0.2, # Lower temperature for less "creativity" / chatter
        )
        
        if hasattr(response, 'usage'):
            self.tracker.record(self.model, response.usage)
        
        return response.choices[0].message.content

    
# ======================================================
# Code Generator
# ======================================================
class CodexGenerator:
    def __init__(self, api_key, tracker: UsageTracker, verbose=False):
        self.client = OpenAI(api_key=api_key)
        self.tracker = tracker
        self.verbose = verbose
        self.model = "gpt-5.1-codex-mini"

    def generate(self, instruction, num_features, cat_features, data_preview, error_context = None) -> str:
        error_block = ""
        if error_context:
            error_block = f"""
        PREVIOUS EXECUTION ERROR (FIX THIS):
        {error_context}

        RULE:
        - Do NOT repeat the same mistake
        - Ensure all variables are defined before use
        """
        prompt = f"""You are an expert data scientist. Generate Python code.

                DATA:
                {data_preview}

                NUMERIC: {num_features}
                CATEGORICAL: {cat_features}

                {error_block}

                TASK: {instruction}

                CRITICAL RULES:
                1. DataFrame 'df' is ALREADY LOADED - NEVER use pd.read_csv/excel
                2. Variables: df, num_features, cat_features, plt, sns, px, go, stats, np, pd, FIGURES_DIR
                3. Save plots: os.path.join(FIGURES_DIR, 'name.png')
                4. ALWAYS plt.close() after plt.savefig()
                5. Every code block MUST BE SELF-CONTAINED. 
                6. If you need a filtered dataframe (like df_clean), you MUST define it 
                inside the current code block using 'df'.
                7. Do not assume variables from previous steps exist.

                ERROR PREVENTION:
                - Check dtypes: df[col].dtype
                - Convert times: pd.to_timedelta(df[col]).dt.total_seconds()
                - Ensure numeric: df[col] = pd.to_numeric(df[col], errors='coerce')
                - Handle NaN: df.dropna(subset=[col])
                - Use pd.isna() not .isnull() on scalars
                - Check column exists: if col in df.columns

                QUALITY:
                - Descriptive titles, labels, legends
                - Appropriate plot types
                - Good font sizes, DPI=150
                - Statistical tests with p-values
                - Print results

                OUTPUT: Pure Python only. NO markdown, NO backticks."""
        response = self.client.responses.create(
            model=self.model,
            input=prompt,)
        
        # Track usage - try to access usage attribute safely
        if hasattr(response, 'usage') and response.usage is not None:
            self.tracker.record(self.model, response.usage)
        
        return response.output_text.strip()

# ======================================================
# ReAct Orchestrator
# ======================================================
class ReActAnalyzer:
    def __init__(self, runner, decider, coder, tracker, max_steps=20, verbose=False):
        self.runner = runner
        self.decider = decider
        self.coder = coder
        self.tracker = tracker
        self.max_steps = max_steps
        self.verbose = verbose
        self.observations = []
        self.analysis_log = []
        self.error_history = []
        self.data_preview = self.runner.df.head(10).to_string()

    def observe(self):
        figs = sorted(self.runner.figures_dir.glob("*.png"))
        obs = f"Figures: {len(figs)} - " + ", ".join([f.name for f in figs]) if figs else "No figures"
        self.observations.append(obs)

    def run(self, user_requirements): 
        print(f"\n🔬 STARTING ANALYSIS\n{'='*80}")
        self.observe()
        
        for step in range(self.max_steps):
            print(f"\n🔄 STEP {step + 1}/{self.max_steps}")
            
            error_context = ""
            if self.error_history:
                recent = self.error_history[-3:]
                error_context = "\n\nRECENT ERRORS:\n" + "\n".join([
                    f"- {e['error']}\n  From: {e['action'][:100]}..."
                    for e in recent
                ])
            
            decision = self.decider.decide(
                "\n".join(self.observations) + error_context, 
                step, self.max_steps, 
                self.runner.num_features, 
                self.runner.cat_features, 
                self.data_preview
            )
            
            if "STOP" in decision.upper(): 
                print("🛑 Analysis complete")
                break
                        
            last_error = None
            if self.error_history:
                last_error = self.error_history[-1]["error"]

            code = self.coder.generate(
                decision,
                self.runner.num_features,
                self.runner.cat_features,
                self.data_preview,
                error_context=last_error
            )

            try:
                self.runner.run(code)
                if self.verbose: print("✅ Success")
                self.observe()
                self.analysis_log.append({
                    "step": step + 1, 
                    "thought": decision, 
                    "result": self.observations[-1],
                    "status": "success"
                })
            except Exception as e:
                error_msg = str(e)
                print(f"❌ Error: {error_msg}")
                
                self.error_history.append({
                    "step": step + 1,
                    "action": decision,
                    "error": error_msg
                })
                
                error_obs = f"ERROR: {error_msg}"
                self.observations.append(error_obs)
                
                self.analysis_log.append({
                    "step": step + 1, 
                    "thought": decision, 
                    "result": error_obs,
                    "status": "error"
                })

        report_text = self.decider.synthesize_report(
            self.analysis_log, 
            user_requirements, 
            self.runner.figures_dir, 
            self.data_preview
        )
        
        report_path = self.runner.output_dir / "report.md" 
        report_path.write_text(report_text, encoding='utf-8')
        print(f"✨ Report saved: {report_path}")
        
        return report_path, report_text


# ======================================================
# 6. CLI
# ======================================================
def main():
    parser = argparse.ArgumentParser(
        description="IAthon - Intelligent Automated Data Analysis with customizable reports"
    )
    parser.add_argument("-i", "--input", required=True, help="Input data file (CSV or Excel)")
    parser.add_argument("-o", "--output", required=True, help="Output file path")
    parser.add_argument("-f", "--format", choices=["pdf", "docx", "pptx"], help="Output format")
    parser.add_argument("--api-key", required=False, help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "-p", "--prompt", 
        required=False, 
        help="Custom prompt describing data context and report requirements (e.g., 'This is triathlon race data. Create a concise executive summary with focus on performance trends.')"
    )
    parser.add_argument(
        "--prompt-file",
        required=False,
        help="Path to text file containing custom prompt"
    )
    args = parser.parse_args()

    api_key = get_api_key(args.api_key)
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load custom prompt if provided
    custom_prompt = None
    if args.prompt_file:
        prompt_file = Path(args.prompt_file)
        if prompt_file.exists():
            custom_prompt = prompt_file.read_text(encoding='utf-8')
            print(f"📝 Loaded custom prompt from: {prompt_file}")
        else:
            print(f"⚠️  Prompt file not found: {prompt_file}")
    elif args.prompt:
        custom_prompt = args.prompt
        print(f"📝 Using custom prompt: {custom_prompt[:100]}...")

    # Initialize usage tracker
    tracker = UsageTracker()

    df, num_features, cat_features = load_and_clean(input_path)
    runner = SafeRunner(df, num_features, cat_features, output_dir, verbose=args.verbose)
    decider = DecisionMaker(api_key, tracker, custom_prompt=custom_prompt, verbose=args.verbose)
    coder = CodexGenerator(api_key, tracker, verbose=args.verbose)

    analyzer = ReActAnalyzer(runner, decider, coder, tracker, max_steps=15, verbose=args.verbose)
    
    # Use custom prompt if provided, otherwise use default
    if custom_prompt:
        user_requirements = custom_prompt
    else:
        user_requirements = "Comprehensive domain-specific report with embedded figures."
    
    report_md_path, report_text = analyzer.run(user_requirements) 

    if args.format:
        print(f"📦 Converting to {args.format}...")
        out_file = output_path.with_suffix(f".{args.format}")
        
        if args.format == "pptx":
            # ---- Presentation synthesis ----
            if hasattr(decider, "synthesize_presentation"):
                pptx_content, domain_context = decider.synthesize_presentation(
                    analyzer.analysis_log,
                    runner.figures_dir,
                    analyzer.data_preview
                )
            else:
                print("⚠️  synthesize_presentation() not implemented, using report.md")
                pptx_content = report_text
                domain_context = "Automated data analysis presentation"

            # ---- Optional title image ----
            generated_image = False
            if hasattr(decider, "generate_title_image"):
                title_image_path = runner.figures_dir / "title_image.png"
                generated_image = decider.generate_title_image(domain_context, title_image_path)

            # ---- Optional reference doc ----
            ref_path = None
            if "create_styled_reference" in globals():
                ref_path = create_styled_reference(output_dir, runner.figures_dir)

            # Build Pandoc command
            pandoc_cmd = ["pandoc"]
            
            # Input file (must be relative to cwd)
            pandoc_cmd.extend([str(report_md_path.name)])
            
            # Output format and file
            pandoc_cmd.extend(["-t", "pptx", "-o", str(out_file.name)])
            
            # Add reference doc if template was created successfully
            if ref_path and ref_path.exists():
                pandoc_cmd.extend([f"--reference-doc={ref_path.name}"])
            
            # Resource path for images (absolute path is safer)
            res_paths = [".", "figures", str(runner.figures_dir.absolute())]
            pandoc_cmd.append(f"--resource-path={os.pathsep.join(res_paths)}")
            
            # Add standalone flag to ensure complete document
            pandoc_cmd.append("--standalone")
            
            print(f"🔧 Running: {' '.join(pandoc_cmd)}")
            
            try:
                result = subprocess.run(
                    pandoc_cmd, 
                    cwd=output_dir, 
                    check=True,
                    capture_output=True,
                    text=True
                )
                print(f"✅ Presentation saved to {out_file}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Pandoc error: {e}")
                print(f"   stdout: {e.stdout}")
                print(f"   stderr: {e.stderr}")
                print(f"   Trying without reference doc...")
                
                # Fallback: try without reference doc
                pandoc_cmd_simple = [
                    "pandoc",
                    str(report_md_path.name),
                    "-t", "pptx",
                    "-o", str(out_file.name),
                    f"--resource-path={os.pathsep.join(['.', 'figures', str(runner.figures_dir.absolute())])}",
                    "--standalone"
                ]
                subprocess.run(pandoc_cmd_simple, cwd=output_dir, check=True)
                print(f"✅ Presentation saved to {out_file} (with default styling)")
        else:
            # For PDF and DOCX, use the full report
            try:
                subprocess.run(
                    ["pandoc", report_md_path.name, "-o", out_file.name],
                    cwd=output_dir,
                    check=True
                )
                print(f"✅ Document saved to {out_file}")
            except FileNotFoundError:
                print("❌ Pandoc not installed or not in PATH")
            except subprocess.CalledProcessError as e:
                print("❌ Pandoc failed:", e)
            
    # Print token usage LAST (after all processing is complete)
    print("\n" + "="*70)
    tracker.print_summary()
    tracker.save_log(output_dir)

if __name__ == "__main__":
    main()