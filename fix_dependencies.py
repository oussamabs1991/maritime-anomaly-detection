#!/usr/bin/env python3
"""
Quick fix script to install the correct dependencies for Pydantic v2
Run this script to fix the import issues.
"""

import subprocess
import sys
import os

def run_command(command):
    """Run a command and return the result"""
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {command}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {command}")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("🔧 Fixing Pydantic dependencies...")
    print()
    
    # Check if we're in a virtual environment
    if sys.prefix == sys.base_prefix:
        print("⚠️  Warning: You don't appear to be in a virtual environment.")
        print("   It's recommended to use a virtual environment.")
        response = input("   Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Exiting. Please activate your virtual environment and try again.")
            return
    
    commands = [
        # Uninstall old pydantic if needed
        "pip uninstall -y pydantic",
        
        # Install correct versions
        "pip install pydantic>=2.0.0",
        "pip install pydantic-settings>=2.0.0",
        
        # Reinstall other requirements
        "pip install -r requirements.txt",
        
        # Install the package in development mode
        "pip install -e ."
    ]
    
    for command in commands:
        if not run_command(command):
            print(f"\n❌ Failed to execute: {command}")
            print("Please run this command manually and fix any issues.")
            return
    
    print("\n🎉 Dependencies fixed successfully!")
    print("\nNow try running:")
    print("   python main.py --smoke-tests-only")

if __name__ == "__main__":
    main()