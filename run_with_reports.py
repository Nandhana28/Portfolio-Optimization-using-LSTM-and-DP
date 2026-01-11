"""
Wrapper script to run portfolio optimization with automatic report generation
"""

import subprocess
import sys
import os
from report_generator import ReportGenerator
from datetime import datetime

def main():
    # Initialize report generator
    report = ReportGenerator()
    
    print("\n" + "="*70)
    print("PORTFOLIO OPTIMIZATION WITH REPORT GENERATION")
    print("="*70)
    print(f" Reports will be saved to: {report.get_report_path()}\n")
    
    # Run the main analysis
    print("Starting analysis...\n")
    
    try:
        # Import and run main analysis
        from main import main as run_analysis
        
        # Capture output
        import io
        from contextlib import redirect_stdout
        
        # Run analysis
        run_analysis()
        
        # Save terminal output
        report.save_terminal_output()
        
        # Create index
        report.create_index()
        
        print("\n" + "="*70)
        print("REPORT GENERATION COMPLETE")
        print("="*70)
        print(f"Open this file to view your report:")
        print(f"   {os.path.join(report.get_report_path(), 'index.html')}")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
