"""Report generation module for portfolio optimization analysis"""

import os
import json
from datetime import datetime


class ReportGenerator:
    """Generates and saves analysis reports"""
    
    def __init__(self, report_dir="Reports"):
        self.report_dir = report_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_path = os.path.join(report_dir, f"Report_{self.timestamp}")
        self.terminal_output = []
        
        # Create directories
        os.makedirs(self.report_path, exist_ok=True)
        os.makedirs(os.path.join(self.report_path, "charts"), exist_ok=True)
        
        print(f"\nReport directory created: {os.path.abspath(self.report_path)}")
    
    def log(self, message):
        """Log message to both console and file"""
        print(message)
        self.terminal_output.append(message)
    
    def save_chart(self, fig, filename):
        """Save Plotly chart as HTML"""
        filepath = os.path.join(self.report_path, "charts", f"{filename}.html")
        fig.write_html(filepath)
        self.log(f"✓ Saved chart: {filename}.html")
        return filepath
    
    def save_terminal_output(self):
        """Save all terminal output to text file"""
        filepath = os.path.join(self.report_path, "terminal_output.txt")
        with open(filepath, 'w') as f:
            f.write('\n'.join(self.terminal_output))
        self.log(f"✓ Saved terminal output: terminal_output.txt")
        return filepath
    
    def save_metrics(self, metrics_dict):
        """Save metrics as JSON"""
        filepath = os.path.join(self.report_path, "metrics.json")
        with open(filepath, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        self.log(f"✓ Saved metrics: metrics.json")
        return filepath
    
    def save_summary(self, summary_dict):
        """Save summary as JSON"""
        filepath = os.path.join(self.report_path, "summary.json")
        with open(filepath, 'w') as f:
            json.dump(summary_dict, f, indent=2)
        self.log(f"✓ Saved summary: summary.json")
        return filepath
    
    def save_risk_analysis(self, risk_dict):
        """Save risk analysis as JSON"""
        filepath = os.path.join(self.report_path, "risk_analysis.json")
        with open(filepath, 'w') as f:
            json.dump(risk_dict, f, indent=2, default=str)
        self.log(f"✓ Saved risk analysis: risk_analysis.json")
        return filepath
    
    def create_index(self):
        """Create HTML index for easy navigation"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Portfolio Optimization Report - {self.timestamp}</title>
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                body {{ 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%);
                    color: #fff;
                    padding: 20px;
                }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                h1 {{ color: #00ff00; margin-bottom: 10px; font-size: 2.5em; }}
                .timestamp {{ color: #888; margin-bottom: 30px; }}
                h2 {{ color: #ffff00; margin-top: 30px; margin-bottom: 15px; font-size: 1.5em; }}
                .section {{ 
                    background-color: #2d2d2d; 
                    padding: 20px; 
                    margin: 15px 0; 
                    border-radius: 8px;
                    border-left: 4px solid #00ff00;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.3);
                }}
                .section:hover {{ background-color: #3d3d3d; transition: 0.3s; }}
                a {{ 
                    color: #00ccff; 
                    text-decoration: none;
                    padding: 10px 15px;
                    display: inline-block;
                    margin: 8px 0;
                    background-color: #1e1e1e;
                    border-radius: 4px;
                    transition: 0.3s;
                }}
                a:hover {{ 
                    background-color: #00ccff;
                    color: #000;
                }}
                .chart-link {{ display: block; margin: 10px 0; }}
                .metric {{ 
                    display: inline-block; 
                    margin: 10px 20px 10px 0; 
                    padding: 15px; 
                    background-color: #3d3d3d; 
                    border-radius: 5px;
                    border: 1px solid #00ff00;
                }}
                .path {{ 
                    background-color: #1e1e1e;
                    padding: 15px;
                    border-radius: 4px;
                    word-break: break-all;
                    font-family: monospace;
                    color: #00ff00;
                }}
                .emoji {{ font-size: 1.2em; margin-right: 5px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1><span class="emoji">📊</span>Portfolio Optimization Analysis Report</h1>
                <p class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                
                <div class="section">
                    <h2><span class="emoji"></span>Interactive Charts</h2>
                    <a href="charts/portfolio_dashboard.html" target="_blank" class="chart-link">
                        <span class="emoji"></span>Main Dashboard (Interactive)
                    </a>
                </div>
                
                <div class="section">
                    <h2><span class="emoji"></span>Analysis Reports</h2>
                    <a href="terminal_output.txt" target="_blank" class="chart-link">
                        <span class="emoji"></span>Terminal Output (All Console Logs)
                    </a>
                    <a href="metrics.json" target="_blank" class="chart-link">
                        <span class="emoji"></span>Performance Metrics (JSON)
                    </a>
                    <a href="summary.json" target="_blank" class="chart-link">
                        <span class="emoji"></span>Portfolio Summary (JSON)
                    </a>
                    <a href="risk_analysis.json" target="_blank" class="chart-link">
                        <span class="emoji"></span>Risk Analysis (JSON)
                    </a>
                </div>
                
                <div class="section">
                    <h2><span class="emoji"></span>Report Location</h2>
                    <div class="path">{os.path.abspath(self.report_path)}</div>
                </div>
                
                <div class="section">
                    <h2><span class="emoji"></span>How to Use</h2>
                    <p>
                        1. Open <strong>index.html</strong> (this file) in your browser<br>
                        2. Click on <strong>Main Dashboard</strong> to view interactive charts<br>
                        3. Download <strong>Terminal Output</strong> for detailed logs<br>
                        4. Export <strong>JSON files</strong> for further analysis<br>
                    </p>
                </div>
            </div>
        </body>
        </html>
        """
        
        filepath = os.path.join(self.report_path, "index.html")
        with open(filepath, 'w') as f:
            f.write(html_content)
        self.log(f"✓ Created index: index.html")
        self.log(f"\nReport generation complete!")
        self.log(f"Open this file to view the report: {os.path.abspath(filepath)}")
        return filepath
    
    def get_report_path(self):
        """Get the report directory path"""
        return os.path.abspath(self.report_path)
